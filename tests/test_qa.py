"""Focused final-QA cases and deterministic thesis-demo checks."""

from copy import deepcopy
from importlib.util import find_spec
from types import SimpleNamespace

import pytest

from agenttree import AgentTree, AgentTreeConfig, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    ProviderTaskTriage,
    RuleBasedTaskTriage,
    StaticFinalReviewer,
    StaticManagerReviewer,
    StaticTaskDecomposer,
)
from agenttree.models import ReviewDecision, SubtaskTemplate
from agenttree.orchestration import FinalResult, FinalStatus, ManagerReviewStatus
from agenttree.orchestration.backends import (
    BaseOrchestrationBackend,
    LangGraphOrchestrationBackend,
    SequentialOrchestrationBackend,
)
from agenttree.providers import (
    BaseProvider,
    MockProvider,
    OpenAIProvider,
    ProviderConfig,
    ProviderRequest,
    ProviderResponse,
    ProviderRuntimeError,
)
from agenttree.tracing import ExecutionEventType
from examples.demo_helpers import format_trace
from examples.thesis_demo import build_demo, main as demo_main, render_demo


def _edge_framework(
    *,
    fallback: tuple[str, ...] = ("manage",),
    templates: tuple[SubtaskTemplate, ...] = (
        SubtaskTemplate("Work", ("work",)),
    ),
) -> AgentTree:
    return AgentTree(
        root_agent=RootAgent(name="Root", id="edge-root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=fallback),
        decomposer=StaticTaskDecomposer(templates),
        manager_reviewer=StaticManagerReviewer(),
        final_reviewer=StaticFinalReviewer(),
    )


def test_empty_objective_and_no_capabilities_return_structured_no_manager() -> None:
    framework = _edge_framework(fallback=())
    framework.register_manager(ManagerAgent(
        name="Manager", id="edge-manager", capabilities=("manage",),
    ))
    task = Task(id="empty-objective", objective="")
    original = deepcopy(task)

    result = framework.run(task)

    assert result.status is FinalStatus.FAILED
    assert not result.success
    assert result.final_review.feedback == "No matching manager found"
    assert task == original


def test_manager_with_zero_specialists_returns_structured_no_assignment() -> None:
    framework = _edge_framework()
    framework.register_manager(ManagerAgent(
        name="Manager", id="edge-manager", capabilities=("manage",),
    ))

    result = framework.run(Task(id="no-specialists", objective="Work"))

    assert result.status is FinalStatus.FAILED
    assert result.final_review.feedback == "No specialist assignments produced"


def test_subtask_with_zero_capabilities_does_not_execute_specialist() -> None:
    framework = _edge_framework(templates=(SubtaskTemplate("Work", ()),))
    manager = ManagerAgent(
        name="Manager", id="edge-manager", capabilities=("manage",),
    )
    specialist = SpecialistAgent(
        name="Worker", id="edge-specialist", capabilities=("work",),
    )
    provider = MockProvider()
    framework.register_manager(manager)
    framework.register_specialist(manager, specialist)
    framework.register_provider(provider)
    framework.bind_provider(specialist, provider)

    result = framework.run(Task(id="empty-subtask-capabilities", objective="Work"))

    assert result.status is FinalStatus.FAILED
    assert provider.requests == ()


def test_real_adapter_rejects_whitespace_only_normalized_content() -> None:
    response = SimpleNamespace(
        output_text="   ",
        model="configured-model",
        usage=None,
        error=None,
        status="completed",
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: response),
    )
    provider = OpenAIProvider(
        ProviderConfig(provider_name="offline-openai", model="configured-model"),
        client=client,
    )

    with pytest.raises(ProviderRuntimeError, match="no normalized text content"):
        provider.generate(ProviderRequest(prompt="Work"))


def test_decision_strategy_rejects_invalid_provider_response_contract() -> None:
    class InvalidProvider(BaseProvider):
        def generate(self, request: ProviderRequest) -> ProviderResponse:
            return object()  # type: ignore[return-value]

    strategy = ProviderTaskTriage(InvalidProvider(
        ProviderConfig(provider_name="invalid-response"),
    ))

    with pytest.raises(TypeError, match="triage.*ProviderResponse"):
        strategy.triage(Task(objective="Work"))


class _RevisionFailureProvider(MockProvider):
    """Succeed initially and fail only during requested re-execution."""

    def __init__(self) -> None:
        super().__init__(response_content="Initial result")
        self.calls = 0

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.calls += 1
        if self.calls == 2:
            raise ProviderRuntimeError("Offline revision failure")
        return super().generate(request)


def _revision_failure_run(
    backend: BaseOrchestrationBackend,
) -> tuple[FinalResult, tuple[str, ...]]:
    framework = AgentTree(
        root_agent=RootAgent(name="Root", id="failure-root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        decomposer=StaticTaskDecomposer((SubtaskTemplate("Work", ("work",)),)),
        manager_reviewer=StaticManagerReviewer(outcomes=(
            (ReviewDecision.REVISE, "Retry with the requested correction"),
            (ReviewDecision.FAIL, "Revision execution failed"),
        )),
        final_reviewer=StaticFinalReviewer(ReviewDecision.FAIL, "Work was not accepted"),
        config=AgentTreeConfig(max_manager_revisions=1),
        orchestration_backend=backend,
    )
    manager = ManagerAgent(name="Manager", id="failure-manager", capabilities=("manage",))
    specialist = SpecialistAgent(name="Worker", id="failure-worker", capabilities=("work",))
    provider = _RevisionFailureProvider()
    framework.register_manager(manager)
    framework.register_specialist(manager, specialist)
    framework.register_provider(provider)
    framework.bind_provider(specialist, provider)
    result = framework.run(Task(id="revision-failure-task", objective="Work"))
    assert provider.calls == 2
    outcome = result.manager_results[0].subtask_outcomes[0]
    assert outcome.status is ManagerReviewStatus.FAILED
    assert outcome.executions[-1].agent_result is not None
    assert "Offline revision failure" in outcome.executions[-1].agent_result.error
    return result, tuple(event.event_type for event in result.trace.events)


def test_provider_failure_during_revision_is_structured() -> None:
    result, labels = _revision_failure_run(SequentialOrchestrationBackend())

    assert result.status is FinalStatus.FAILED
    assert ExecutionEventType.SPECIALIST_EXECUTION_FAILED.value in labels
    assert ExecutionEventType.MANAGER_REVIEW_FAILED.value in labels


@pytest.mark.skipif(find_spec("langgraph") is None, reason="Optional LangGraph is absent")
def test_backend_equivalence_after_revision_provider_failure() -> None:
    sequential, sequential_labels = _revision_failure_run(SequentialOrchestrationBackend())
    graph, graph_labels = _revision_failure_run(LangGraphOrchestrationBackend())

    assert graph.status == sequential.status
    assert graph.success == sequential.success
    assert graph_labels == sequential_labels


def test_primary_thesis_demo_uses_expected_public_results(capsys) -> None:
    assert demo_main([]) == 0
    output = capsys.readouterr().out

    assert "AgentTree Thesis Demo" in output
    assert "Selected Manager:\n- IT Incident Manager" in output
    assert "- Network Specialist: offline-network" in output
    assert "- Security and Log Specialist: offline-security-log" in output
    assert "Final Status:\nCOMPLETED" in output
    assert "8. result created" in output
    assert "FinalResult(" not in output


def test_revision_demo_shows_feedback_reexecution_and_latest_outputs_only() -> None:
    framework, task, manager, specialists = build_demo(revision=True)
    result = framework.run(task)
    output = render_demo(
        framework,
        task,
        manager,
        specialists,
        result,
        backend="sequential",
        revision=True,
    )

    assert all(len(provider.requests) == 2 for provider in framework.providers)
    assert "REVISE -> PASS" in output
    assert "Manager requested revision" in output
    assert "Feedback returned to the Specialist context" in output
    assert "Specialist executed again" in output
    assert output.count("- Network Specialist: Latency") == 1
    assert output.count("- Security and Log Specialist: The synthetic") == 1


def test_full_trace_formatter_preserves_event_order() -> None:
    framework, task, _, _ = build_demo()
    result = framework.run(task)
    rendered = format_trace(result.trace).splitlines()

    assert len(rendered) == result.trace.event_count
    assert rendered[0] == "[01] orchestration.started"
    assert rendered[-1].endswith("orchestration.final_result_created")
