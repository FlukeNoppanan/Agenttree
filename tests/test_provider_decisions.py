"""Offline structured decisions, trust boundaries, and full pipeline tests."""

from copy import deepcopy
import json
import subprocess
import sys
from typing import Any

import pytest

from agenttree.agents import ManagerAgent, RootAgent, SpecialistAgent
from agenttree.core import (
    BaseFinalReviewer, BaseManagerReviewer, BaseTaskDecomposer, BaseTaskTriage,
    DecisionOutputError, DecisionParseError, ProviderFinalReviewer,
    ProviderManagerReviewer, ProviderSpecialistExecutor, ProviderTaskDecomposer,
    ProviderTaskTriage,
)
from agenttree.core.structured_output import parse_decision_output
from agenttree.models import (
    AgentResult, ExecutionTrace, ReviewDecision, Subtask, Task, TaskContext,
    TriageResult,
)
from agenttree.orchestration import (
    ExecutionStatus, FinalStatus, ManagerReviewStatus, OrchestrationEngine,
    SpecialistExecution, TaskManagerReviewResult,
    FinalResult,
)
from agenttree.providers import (
    BaseProvider, MockProvider, ProviderConfig, ProviderRegistry,
    ProviderRequest, ProviderResponse, ProviderRuntimeError,
)
from agenttree.registry import CapabilityRegistry
from tests.test_provider_adapters import ADAPTERS, FakeClient, sdk_result


TRIAGE = {"objective": "Normalized work", "required_capabilities": ["manage"]}
DECOMPOSITION = {"subtasks": [
    {"objective": "Produce output", "required_capabilities": ["work"]},
]}
REVIEW = {"decision": "pass", "feedback": "Accepted"}
STRATEGIES = (
    ProviderTaskTriage, ProviderTaskDecomposer,
    ProviderManagerReviewer, ProviderFinalReviewer,
)


def mock(data: dict[str, Any], name: str = "mock") -> MockProvider:
    return MockProvider(
        ProviderConfig(provider_name=name, model="fixture-model"),
        response_content=json.dumps(data),
    )


def inputs() -> tuple[Task, ManagerAgent, RootAgent, Subtask, TaskManagerReviewResult]:
    task = Task(
        objective="Complete work", context=TaskContext(data={"nested": [1]}),
        metadata={"final_revision": {"feedback": "Check completeness"}},
    )
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    root = RootAgent(name="Root")
    subtask = Subtask(
        parent_task_id=task.id, manager_id=manager.id, objective="Work",
        metadata={"revision": {"feedback": "Check evidence"}},
    )
    reviewed = TaskManagerReviewResult(
        task_id=task.id, manager_results=(), status=ManagerReviewStatus.PASSED,
        trace=ExecutionTrace(task_id=task.id),
    )
    return task, manager, root, subtask, reviewed


def invoke(strategy: type, provider: BaseProvider) -> Any:
    task, manager, root, subtask, reviewed = inputs()
    component = strategy(provider=provider)
    if strategy is ProviderTaskTriage:
        return component.triage(task)
    if strategy is ProviderTaskDecomposer:
        return component.decompose(task, manager, TriageResult(task.id, task.objective))
    if strategy is ProviderManagerReviewer:
        return component.review(task, subtask, manager, ())
    return component.review(task, root, reviewed)


@pytest.mark.parametrize("wrapper", ["{}", "  {}\n", "```json\n{}\n```", "```\n{}\n```"])
def test_json_object_and_conservative_fences(wrapper: str) -> None:
    assert parse_decision_output(wrapper.format(json.dumps(TRIAGE))) == TRIAGE


@pytest.mark.parametrize("content", [
    "", "not JSON", '{"broken":', 'Here is JSON: {}', '```python\n{}\n```',
    '```json\n{}\n```\nextra', '{}\n{}', '{"x":1,"x":2}',
    '{"value":NaN}', '{"value":Infinity}', '{"value":1e999}',
])
def test_malformed_json_raises_parse_error(content: str) -> None:
    with pytest.raises(DecisionParseError):
        parse_decision_output(content)


def test_json_error_preserves_cause() -> None:
    with pytest.raises(DecisionParseError) as caught:
        parse_decision_output("{")
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize("content", ['[]', 'null', '"text"', '42'])
def test_non_object_json_is_invalid_structure(content: str) -> None:
    with pytest.raises(DecisionOutputError, match="object"):
        parse_decision_output(content)


def test_triage_normalization_identity_and_isolation() -> None:
    task, *_ = inputs()
    provider = mock({
        **TRIAGE, "objective": "  Normalized work ", "task_id": "untrusted",
        "required_capabilities": [" Analysis ", "analysis", "Writing", "WRITING"],
        "category": " generic ", "confidence": 0.8, "notes": " note ",
        "metadata": {"nested": [1]},
    })
    before = deepcopy(task)
    triage = ProviderTaskTriage(provider)
    result = triage.triage(task)
    assert isinstance(triage, BaseTaskTriage)
    assert result.task_id == task.id
    assert result.objective == "Normalized work"
    assert result.required_capabilities == ("Analysis", "Writing")
    assert (result.category, result.confidence, result.notes) == ("generic", 0.8, "note")
    result.metadata["nested"].append(2)
    assert triage.triage(task).metadata == {"nested": [1]}
    request = provider.requests[0]
    request.context["task"]["context"]["nested"].append(2)
    assert task == before
    assert request.metadata == {"strategy": "triage"}
    assert "JSON only" in request.system_prompt
    assert request.model is None  # The injected provider's config supplies defaults.


@pytest.mark.parametrize("data, field", [
    ({}, "objective"),
    ({"objective": "work"}, "required_capabilities"),
    ({**TRIAGE, "objective": " "}, "objective"),
    ({**TRIAGE, "objective": 1}, "objective"),
    ({**TRIAGE, "required_capabilities": "analysis"}, "required_capabilities"),
    ({**TRIAGE, "required_capabilities": ["ok", 2]}, "required_capabilities"),
    ({**TRIAGE, "required_capabilities": [""]}, "required_capabilities"),
    ({**TRIAGE, "required_capabilities": None}, "required_capabilities"),
    ({**TRIAGE, "confidence": True}, "confidence"),
    ({**TRIAGE, "confidence": "0.5"}, "confidence"),
    ({**TRIAGE, "confidence": -0.1}, "confidence"),
    ({**TRIAGE, "confidence": 10**500}, "confidence"),
    ({**TRIAGE, "category": []}, "category"),
    ({**TRIAGE, "notes": None}, "notes"),
    ({**TRIAGE, "metadata": []}, "metadata"),
])
def test_invalid_triage_fields(data: dict[str, Any], field: str) -> None:
    with pytest.raises(DecisionOutputError, match=field):
        ProviderTaskTriage(mock(data)).triage(Task(objective="work"))


def test_triage_optional_fields_and_empty_capabilities() -> None:
    result = ProviderTaskTriage(mock({**TRIAGE, "required_capabilities": []})).triage(
        Task(objective="work"),
    )
    assert result.required_capabilities == ()
    assert result.category is None and result.confidence is None
    assert result.notes == "" and result.metadata == {}


def test_decomposition_order_fresh_identities_and_context() -> None:
    task, manager, *_ = inputs()
    triage = TriageResult(task.id, "Normalized", ("manage",))
    provider = mock({"subtasks": [
        {"id": "fake", "parent_task_id": "fake", "manager_id": "fake",
         "objective": " First ", "required_capabilities": [" Work ", "work"],
         "metadata": {"nested": [1]}},
        {"objective": "Second", "required_capabilities": []},
    ]})
    decomposer = ProviderTaskDecomposer(provider)
    assert isinstance(decomposer, BaseTaskDecomposer)
    result = decomposer.decompose(task, manager, triage)
    assert [item.objective for item in result] == ["First", "Second"]
    assert result[0].required_capabilities == ("Work",)
    assert all(item.parent_task_id == task.id and item.manager_id == manager.id for item in result)
    later = decomposer.decompose(task, manager, triage)
    assert len({item.id for item in (*result, *later)}) == 4
    assert all(item.id != "fake" for item in result)
    result[0].metadata["nested"].append(2)
    assert later[0].metadata == {"nested": [1]}
    context = provider.requests[0].context
    assert context["triage"]["objective"] == "Normalized"
    assert context["manager"]["capabilities"] == ("manage",)
    assert "specialists" not in context["manager"]
    assert ProviderTaskDecomposer(mock({"subtasks": []})).decompose(task, manager, triage) == ()


@pytest.mark.parametrize("data", [
    {}, {"subtasks": {}}, {"subtasks": ["bad"]}, {"subtasks": [{}]},
    {"subtasks": [{"objective": "ok"}]},
    {"subtasks": [{"objective": None, "required_capabilities": []}]},
    {"subtasks": [{"objective": "ok", "required_capabilities": [False]}]},
    {"subtasks": [{"objective": "ok", "required_capabilities": [], "metadata": None}]},
])
def test_invalid_decomposition(data: dict[str, Any]) -> None:
    with pytest.raises(DecisionOutputError):
        invoke(ProviderTaskDecomposer, mock(data))


@pytest.mark.parametrize("strategy", [ProviderManagerReviewer, ProviderFinalReviewer])
@pytest.mark.parametrize("decision", list(ReviewDecision))
def test_review_decisions_and_framework_reviewer_identity(strategy: type, decision: ReviewDecision) -> None:
    provider = mock({"decision": decision.value, "feedback": " Feedback ",
                     "reviewer_id": "fake", "metadata": {"nested": [1]}})
    task, manager, root, subtask, reviewed = inputs()
    reviewer = strategy(provider)
    if strategy is ProviderManagerReviewer:
        assert isinstance(reviewer, BaseManagerReviewer)
        result = reviewer.review(task, subtask, manager, ())
        assert result.reviewer_id == manager.id
    else:
        assert isinstance(reviewer, BaseFinalReviewer)
        result = reviewer.review(task, root, reviewed)
        assert result.reviewer_id == root.id
    assert result.decision is decision
    assert result.feedback == "Feedback"
    result.metadata["nested"].append(2)
    assert invoke(strategy, provider).metadata == {"nested": [1]}


@pytest.mark.parametrize("strategy", [ProviderManagerReviewer, ProviderFinalReviewer])
@pytest.mark.parametrize("data", [
    {}, {"decision": "pass"}, {"decision": "PASS", "feedback": "x"},
    {"decision": " pass ", "feedback": "x"}, {"decision": [], "feedback": "x"},
    {"decision": "pass", "feedback": 42},
    {"decision": "fail", "feedback": "x", "metadata": []},
])
def test_invalid_review_fields(strategy: type, data: dict[str, Any]) -> None:
    with pytest.raises(DecisionOutputError):
        invoke(strategy, mock(data))


def test_manager_context_preserves_failed_executions_and_revision_feedback() -> None:
    task, manager, _, subtask, _ = inputs()
    failed = SpecialistExecution(
        subtask_id=subtask.id, specialist_id="specialist", status=ExecutionStatus.FAILED,
        agent_result=AgentResult(agent_id="specialist", success=False,
                                 output={"partial": [1]}, error="Generation failed"),
    )
    provider = mock(REVIEW)
    ProviderManagerReviewer(provider).review(task, subtask, manager, (failed,))
    context = provider.requests[0].context
    execution = context["specialist_executions"][0]
    assert execution["status"] == "failed"
    assert execution["agent_result"]["success"] is False
    assert execution["agent_result"]["error"] == "Generation failed"
    assert context["subtask"]["metadata"]["revision"]["feedback"] == "Check evidence"
    assert context["task"]["metadata"]["final_revision"]["feedback"] == "Check completeness"
    execution["agent_result"]["output"]["partial"].append(2)
    assert failed.agent_result.output == {"partial": [1]}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_provider_errors_are_not_parsing_errors(strategy: type) -> None:
    failure = ProviderRuntimeError("offline failure")

    class BrokenProvider(MockProvider):
        def generate(self, request: ProviderRequest) -> ProviderResponse:
            raise failure

    with pytest.raises(ProviderRuntimeError) as caught:
        invoke(strategy, BrokenProvider())
    assert caught.value is failure
    assert not isinstance(caught.value, DecisionOutputError)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_malformed_output_for_every_strategy(strategy: type) -> None:
    with pytest.raises(DecisionParseError):
        invoke(strategy, MockProvider(response_content="broken"))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_injection_requires_base_provider(strategy: type) -> None:
    with pytest.raises(TypeError, match="BaseProvider"):
        strategy(provider=object())


def test_input_identity_mismatches_fail_before_generation() -> None:
    task, manager, root, subtask, reviewed = inputs()
    provider = mock(REVIEW)
    with pytest.raises(ValueError):
        ProviderTaskDecomposer(provider).decompose(task, manager, TriageResult("other", "x"))
    with pytest.raises(ValueError):
        ProviderManagerReviewer(provider).review(Task(objective="other"), subtask, manager, ())
    wrong = SpecialistExecution("other", "specialist", ExecutionStatus.FAILED)
    with pytest.raises(ValueError):
        ProviderManagerReviewer(provider).review(task, subtask, manager, (wrong,))
    with pytest.raises(ValueError):
        ProviderFinalReviewer(provider).review(Task(objective="other"), root, reviewed)
    assert provider.requests == ()


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("strategy, data", [
    (ProviderTaskTriage, TRIAGE), (ProviderTaskDecomposer, DECOMPOSITION),
    (ProviderManagerReviewer, REVIEW), (ProviderFinalReviewer, REVIEW),
])
def test_all_strategies_work_through_real_adapters_with_fake_clients(
    adapter: type, strategy: type, data: dict[str, Any],
) -> None:
    raw = sdk_result(adapter)
    raw.output_text = raw.text = raw.response = json.dumps(data)
    client = FakeClient(raw)
    provider = adapter(ProviderConfig(provider_name="adapter", model="fixture"), client=client)
    assert invoke(strategy, provider) is not None
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == "fixture"


def run_pipeline(providers: dict[str, MockProvider]) -> FinalResult:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    specialist = SpecialistAgent(name="Specialist", capabilities=("work",))
    manager.register_specialist(specialist)
    registry.register(manager)
    registry.register(specialist)
    engine = OrchestrationEngine(triage=ProviderTaskTriage(providers["triage"]), registry=registry)
    task = Task(objective="Complete work")
    delegation = engine.delegate(engine.orchestrate(task), task,
                                 decomposer=ProviderTaskDecomposer(providers["decomposition"]))
    provider_registry = ProviderRegistry()
    provider_registry.register(providers["specialist"])
    executor = ProviderSpecialistExecutor(
        provider_registry=provider_registry,
        provider_bindings={specialist.id: providers["specialist"].name},
    )
    execution = engine.execute(delegation, task, executor=executor)
    reviewer = ProviderManagerReviewer(providers["manager_review"])
    reviewed = engine.review(execution, task, reviewer=reviewer, executor=executor)
    result = engine.finalize(
        reviewed, task, root_agent=RootAgent(name="Root"),
        final_reviewer=ProviderFinalReviewer(providers["final_review"]),
        manager_reviewer=reviewer, executor=executor,
    )
    assert result.success and result.status is FinalStatus.COMPLETED
    assert result.final_review.decision is ReviewDecision.PASS
    assert result.manager_results[0].subtask_outcomes[0].executions[-1].agent_result.output == "finished"
    assert result.trace.task_id == task.id
    return result


def test_complete_mock_provider_pipeline() -> None:
    providers = {
        "triage": mock(TRIAGE), "decomposition": mock(DECOMPOSITION),
        "specialist": MockProvider(response_content="finished"),
        "manager_review": mock(REVIEW), "final_review": mock(REVIEW),
    }
    result = run_pipeline(providers)
    assert all(len(provider.requests) == 1 for provider in providers.values())
    assert result.revision_count == 0
    final_context = providers["final_review"].requests[0].context["manager_review_result"]
    outcome = final_context["manager_results"][0]["subtask_outcomes"][0]
    assert outcome["executions"][0]["agent_result"]["output"] == "finished"
    assert outcome["reviews"][0]["feedback"] == "Accepted"
    outcome["reviews"][0]["feedback"] = "mutated snapshot"
    assert result.manager_results[0].subtask_outcomes[0].reviews[0].feedback == "Accepted"


def test_engine_retains_revision_control_with_provider_decisions() -> None:
    class SequenceMock(MockProvider):
        """Return revise once, then pass, while using normal mock recording."""

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            super().generate(request)
            data = {"decision": "revise", "feedback": "Check evidence"} if len(self.requests) == 1 else REVIEW
            return ProviderResponse(content=json.dumps(data), provider=self.name)

    manager = SequenceMock()
    final = SequenceMock()
    specialist = MockProvider(response_content="finished")
    result = run_pipeline({
        "triage": mock(TRIAGE), "decomposition": mock(DECOMPOSITION),
        "specialist": specialist, "manager_review": manager, "final_review": final,
    })
    assert result.revision_count == 1
    assert len(specialist.requests) == 2
    assert len(manager.requests) == 3 and len(final.requests) == 2
    # The existing engine sends derived revision metadata to the executor;
    # manager re-review receives the original subtask and the new executions.
    assert specialist.requests[1].context["subtask"]["metadata"]["revision"]["feedback"] == "Check evidence"
    assert manager.requests[2].context["task"]["metadata"]["final_revision"]["feedback"] == "Check evidence"
    assert [review.decision for review in result.final_reviews] == [ReviewDecision.REVISE, ReviewDecision.PASS]


def test_mixed_providers_can_be_shared_across_decision_layers() -> None:
    # One static JSON object can satisfy both contracts without a global provider.
    provider_a = mock({**TRIAGE, **REVIEW}, "provider-a")
    provider_b = mock({**DECOMPOSITION, **REVIEW}, "provider-b")
    provider_c = MockProvider(ProviderConfig("provider-c"), response_content="finished")
    run_pipeline({
        "triage": provider_a, "decomposition": provider_b, "specialist": provider_c,
        "manager_review": provider_a, "final_review": provider_b,
    })
    assert [request.metadata["strategy"] for request in provider_a.requests] == ["triage", "manager_review"]
    assert [request.metadata["strategy"] for request in provider_b.requests] == ["decomposition", "final_review"]
    assert len(provider_c.requests) == 1


def test_public_api_imports_without_optional_provider_sdks() -> None:
    script = '''
import importlib.abc
import sys
class BlockSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'openai', 'google', 'ollama'}:
            raise AssertionError('Optional SDK import: ' + fullname)
sys.meta_path.insert(0, BlockSDKs())
from agenttree.core import (
    ProviderTaskTriage, ProviderTaskDecomposer,
    ProviderManagerReviewer, ProviderFinalReviewer,
    RuleBasedTaskTriage, StaticTaskDecomposer,
    StaticManagerReviewer, StaticFinalReviewer,
    DecisionOutputError, DecisionParseError,
)
from agenttree.exceptions import DecisionOutputError, DecisionParseError
from agenttree.providers import OpenAIProvider, GeminiProvider, OllamaProvider
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
