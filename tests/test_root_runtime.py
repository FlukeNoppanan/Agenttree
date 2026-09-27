"""End-to-end Root answer, direct path, failure, and usage contracts."""

import pytest

from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    ProviderRootPlanner, ProviderRootSynthesizer, RuleBasedTaskTriage,
    StaticFinalReviewer, StaticManagerReviewer, StaticTaskDecomposer,
)
from agenttree.models import ReviewDecision, SubtaskTemplate, WorkflowPhase
from agenttree.exceptions import DecisionParseError
from agenttree.orchestration import FinalStatus
from agenttree.providers import MockProvider, ProviderRequest, ProviderResponse, ProviderUsage
from agenttree.providers.exceptions import (
    ProviderAuthenticationError, ProviderRateLimitError, ProviderTimeoutError,
    normalize_provider_error,
)


def make_tree(*, planner=None, synthesizer=None, manager_reviewer=None,
              final_reviewer=None, managers=1, specialists=1):
    tree = AgentTree(
        root_agent=RootAgent(name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        decomposer=StaticTaskDecomposer((SubtaskTemplate("Explain the subject", ("work",)),)),
        manager_reviewer=manager_reviewer or StaticManagerReviewer(),
        final_reviewer=final_reviewer or StaticFinalReviewer(),
        root_planner=planner, root_synthesizer=synthesizer,
    )
    workers = []
    for index in range(managers):
        manager = ManagerAgent(name=f"Manager {index}", capabilities=("manage",))
        tree.register_manager(manager)
        for worker_index in range(specialists):
            worker = SpecialistAgent(name=f"Worker {index}/{worker_index}", capabilities=("work",))
            tree.register_specialist(manager, worker)
            workers.append(worker)
    return tree, workers


def bind_workers(tree, workers, provider):
    tree.register_provider(provider)
    for worker in workers:
        tree.bind_provider(worker, provider)


def test_api_gateway_root_answer_and_diagnostic_separation():
    synthesis = MockProvider(response_content="An API gateway routes client requests to services.")
    tree, workers = make_tree(synthesizer=ProviderRootSynthesizer(synthesis))
    worker = MockProvider(response_content="Gateways route, authenticate, and rate limit.")
    bind_workers(tree, workers, worker)
    result = tree.run(Task(objective="Explain what an API gateway does."))
    assert result.success and result.status is FinalStatus.COMPLETED
    assert result.final_output == "An API gateway routes client requests to services."
    assert result.orchestration is result.content
    assert result.orchestration["managers"][0]["subtasks"][0]["specialists"]
    context = synthesis.requests[0].context
    assert context["original_request"] == "Explain what an API gateway does."
    assert context["managers"][0]["subtasks"][0]["accepted_outputs"] == [
        "Gateways route, authenticate, and rate limit."]
    assert "provider_metadata" not in str(context)
    assert [event.event_type for event in result.trace.events[-2:]] == [
        "root.synthesis.completed", "execution.completed"]


def test_direct_root_response_skips_manager_and_specialist():
    planner = MockProvider(response_content='{"delegate":false,"direct_output":"สวัสดี!"}')
    tree, workers = make_tree(planner=ProviderRootPlanner(planner))
    worker = MockProvider()
    bind_workers(tree, workers, worker)
    result = tree.run(Task(objective="สวัสดี"))
    assert result.final_output == "สวัสดี!" and result.success
    assert result.manager_results == () and result.orchestration == {"managers": []}
    assert worker.requests == ()
    assert tree.last_state.current_phase is WorkflowPhase.COMPLETED
    assert [event.event_type for event in result.trace.events] == [
        "execution.started", "root.planning.started", "root.planning.completed",
        "root.direct_response", "root.synthesis.completed", "execution.completed"]


def test_root_planner_delegates_and_usage_includes_planning():
    usage = ProviderUsage(1, 1, 2)
    planner = MockProvider(response_content='{"delegate":true}', usage=usage)
    tree, workers = make_tree(planner=ProviderRootPlanner(planner))
    worker = MockProvider(response_content="Delegated answer", usage=usage)
    bind_workers(tree, workers, worker)
    result = tree.run(Task(objective="Build a complete API"))
    assert result.final_output == "Delegated answer"
    assert len(worker.requests) == 1
    assert result.usage["total_tokens"] == 4
    assert [call["stage"] for call in result.usage["calls"]] == [
        "root_planning", "specialist"]
    labels = [event.event_type for event in result.trace.events]
    assert labels.index("root.planning.completed") < labels.index("orchestration.started")


def test_malformed_root_plan_fails_without_delegation():
    planner = MockProvider(response_content="not JSON")
    tree, workers = make_tree(planner=ProviderRootPlanner(planner))
    worker = MockProvider()
    bind_workers(tree, workers, worker)
    with pytest.raises(DecisionParseError):
        tree.run(Task(objective="Any request"))
    assert worker.requests == ()
    assert tree.last_state.current_phase is WorkflowPhase.FAILED
    assert tree.last_state.trace.last_event.event_type == "execution.failed"


def test_multiple_workers_usage_includes_synthesis_once():
    usage = ProviderUsage(2, 3, 5)
    synthesis = MockProvider(response_content="Combined answer", usage=usage)
    tree, workers = make_tree(synthesizer=ProviderRootSynthesizer(synthesis),
                              managers=2, specialists=2)
    worker = MockProvider(response_content="Work", usage=usage)
    bind_workers(tree, workers, worker)
    result = tree.run(Task(objective="Explain gateways"))
    assert result.final_output == "Combined answer"
    assert len(result.manager_results) == 2 and len(worker.requests) == 4
    assert result.usage["input_tokens"] == 10
    assert result.usage["output_tokens"] == 15
    assert result.usage["total_tokens"] == 25
    assert len(result.usage["calls"]) == 5


def test_synthesis_failure_preserves_work_without_leaking_secret():
    class Failing(MockProvider):
        def generate(self, request: ProviderRequest) -> ProviderResponse:
            raise RuntimeError("Authorization: Bearer secret-123")

    tree, workers = make_tree(synthesizer=ProviderRootSynthesizer(Failing()))
    bind_workers(tree, workers, MockProvider(response_content="Valid work"))
    result = tree.run(Task(objective="Explain gateways"))
    assert not result.success and result.final_output is None
    assert result.error == {"type": "RuntimeError", "message": "Root synthesis failed"}
    assert result.orchestration["managers"] and result.manager_results
    assert result.trace.events[-2].event_type == "root.synthesis.failed"
    assert result.trace.events[-1].event_type == "execution.failed"
    assert tree.last_state.current_phase is WorkflowPhase.FAILED
    assert "secret-123" not in str(tree.last_state.to_dict())


def test_failed_specialist_cannot_be_marked_success_by_pass_review():
    class Failing(MockProvider):
        def generate(self, request: ProviderRequest) -> ProviderResponse:
            raise RuntimeError("Bearer secret-123")

    tree, workers = make_tree()
    bind_workers(tree, workers, Failing())
    result = tree.run(Task(objective="Work"))
    assert not result.success and result.status is FinalStatus.PARTIAL
    assert result.final_output is None
    assert result.trace.last_event.event_type == "execution.failed"
    assert "secret-123" not in str(result)


def test_revision_is_bounded_and_uses_revised_accepted_output():
    reviewer = StaticManagerReviewer(outcomes=(
        (ReviewDecision.REVISE, "Improve"), (ReviewDecision.PASS, "Accepted")))
    tree, workers = make_tree(manager_reviewer=reviewer)
    provider = MockProvider(response_content="Improved explanation")
    bind_workers(tree, workers, provider)
    result = tree.run(Task(objective="Explain gateways"))
    assert result.final_output == "Improved explanation"
    assert len(provider.requests) == 2
    assert result.manager_results[0].subtask_outcomes[0].revision_count == 1


def test_provider_error_categories_do_not_copy_secret_messages():
    class StatusError(Exception):
        status_code = 401

    assert isinstance(normalize_provider_error(StatusError("Bearer secret-123")),
                      ProviderAuthenticationError)
    assert isinstance(normalize_provider_error(TimeoutError("Bearer secret-123")),
                      ProviderTimeoutError)
    class LimitError(Exception):
        status_code = 429
    assert isinstance(normalize_provider_error(LimitError("Bearer secret-123")),
                      ProviderRateLimitError)
    for error in (StatusError("Bearer secret-123"), LimitError("Bearer secret-123")):
        assert "secret-123" not in str(normalize_provider_error(error))
