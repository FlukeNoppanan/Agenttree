"""Manager review and bounded revision-loop tests."""

from dataclasses import fields

import pytest

from agenttree.agents import ManagerAgent, SpecialistAgent
from agenttree.core import (
    BaseManagerReviewer,
    ProviderSpecialistExecutor,
    RuleBasedTaskTriage,
    StaticManagerReviewer,
    StaticTaskDecomposer,
)
from agenttree.models import (
    ReviewDecision,
    ReviewResult,
    Subtask,
    SubtaskTemplate,
    Task,
    TaskContext,
)
from agenttree.orchestration import (
    ManagerReviewStatus,
    OrchestrationEngine,
    OrchestrationEventType,
    TaskExecutionResult,
)
from agenttree.providers import (
    BaseProvider,
    MockProvider,
    ProviderConfig,
    ProviderRegistry,
    ProviderRequest,
    ProviderResponse,
)
from agenttree.registry import CapabilityRegistry


class FailingProvider(BaseProvider):
    """Fail every generation call for review-input tests."""

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        raise RuntimeError("generation failed")


def _review_pipeline(
    *,
    templates: tuple[SubtaskTemplate, ...] | None = None,
    specialists: tuple[SpecialistAgent, ...] | None = None,
    provider: BaseProvider | None = None,
) -> tuple[
    OrchestrationEngine,
    Task,
    TaskExecutionResult,
    ProviderSpecialistExecutor,
    ManagerAgent,
    tuple[SpecialistAgent, ...],
]:
    templates = templates or (
        SubtaskTemplate(objective="Work", required_capabilities=("work",)),
    )
    specialists = specialists or (
        SpecialistAgent(name="Worker", capabilities=("work",)),
    )
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    registry.register(manager)
    for specialist in specialists:
        manager.register_specialist(specialist)
        registry.register(specialist)
    engine = OrchestrationEngine(
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        registry=registry,
    )
    task = Task(
        objective="Parent",
        context=TaskContext(data={"input": {"value": 1}}),
        metadata={"source": {"id": 1}},
    )
    delegation = engine.delegate(
        engine.orchestrate(task),
        task,
        decomposer=StaticTaskDecomposer(templates),
    )
    provider = provider or MockProvider()
    providers = ProviderRegistry()
    providers.register(provider)
    executor = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={item.id: provider.name for item in specialists},
    )
    execution = engine.execute(delegation, task, executor=executor)
    return engine, task, execution, executor, manager, specialists


def test_base_manager_reviewer_is_an_abstract_contract() -> None:
    with pytest.raises(TypeError):
        BaseManagerReviewer()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "decision",
    [ReviewDecision.PASS, ReviewDecision.REVISE, ReviewDecision.FAIL],
)
def test_static_reviewer_returns_configured_decision_and_feedback(
    decision: ReviewDecision,
) -> None:
    engine, task, execution, _, manager, _ = _review_pipeline()
    subtask = execution.delegation.manager_delegations[0].subtasks[0]  # type: ignore[union-attr]
    executions = execution.manager_executions[0].specialist_executions
    result = StaticManagerReviewer(decision, "Configured feedback").review(
        task, subtask, manager, executions,
    )
    assert result.decision is decision
    assert result.feedback == "Configured feedback"
    assert result.reviewer_id == manager.id


def test_static_reviewer_sequence_is_independent_per_subtask() -> None:
    reviewer = StaticManagerReviewer(outcomes=(
        (ReviewDecision.REVISE, "Revise"),
        (ReviewDecision.PASS, "Accepted"),
    ))
    manager = ManagerAgent(name="Manager")
    task = Task(objective="Parent")
    first = Subtask(parent_task_id=task.id, manager_id=manager.id, objective="First")
    second = Subtask(parent_task_id=task.id, manager_id=manager.id, objective="Second")
    assert reviewer.review(task, first, manager, ()).decision is ReviewDecision.REVISE
    assert reviewer.review(task, second, manager, ()).decision is ReviewDecision.REVISE
    assert reviewer.review(task, first, manager, ()).decision is ReviewDecision.PASS


def test_pass_accepts_result_without_reexecution() -> None:
    provider = MockProvider()
    engine, task, execution, executor, _, _ = _review_pipeline(provider=provider)
    original_calls = len(provider.requests)
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(ReviewDecision.PASS, "Accepted"),
        executor=executor,
    )
    outcome = result.manager_results[0].subtask_outcomes[0]
    assert result.status is ManagerReviewStatus.PASSED
    assert outcome.status is ManagerReviewStatus.PASSED
    assert outcome.decision is ReviewDecision.PASS
    assert outcome.feedback == "Accepted"
    assert outcome.revision_count == 0
    assert outcome.revisions == ()
    assert len(provider.requests) == original_calls


def test_fail_is_terminal_without_reexecution() -> None:
    provider = MockProvider()
    engine, task, execution, executor, _, _ = _review_pipeline(provider=provider)
    original_calls = len(provider.requests)
    before_count = len(execution.trace.events)
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(ReviewDecision.FAIL, "Rejected"),
        executor=executor,
    )
    outcome = result.manager_results[0].subtask_outcomes[0]
    assert result.status is ManagerReviewStatus.FAILED
    assert outcome.decision is ReviewDecision.FAIL
    assert outcome.feedback == "Rejected"
    assert outcome.revision_count == 0
    assert len(provider.requests) == original_calls
    assert tuple(event.event_type for event in result.trace.events[before_count:]) == (
        OrchestrationEventType.MANAGER_REVIEW_STARTED.value,
        OrchestrationEventType.MANAGER_REVIEW_FAILED.value,
        OrchestrationEventType.MANAGER_REVIEW_COMPLETED.value,
    )


def test_revise_reexecutes_then_reviews_new_results() -> None:
    provider = MockProvider()
    engine, task, execution, executor, _, specialists = _review_pipeline(
        provider=provider,
    )
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(outcomes=(
            (ReviewDecision.REVISE, "Add detail"),
            (ReviewDecision.PASS, "Accepted"),
        )),
        executor=executor,
        max_revisions=2,
    )
    outcome = result.manager_results[0].subtask_outcomes[0]
    assert outcome.status is ManagerReviewStatus.PASSED
    assert outcome.revision_count == 1
    assert tuple(item.decision for item in outcome.reviews) == (
        ReviewDecision.REVISE, ReviewDecision.PASS,
    )
    assert len(outcome.executions) == 2
    assert outcome.revisions[0].revision_number == 1
    assert outcome.revisions[0].specialist_ids == (specialists[0].id,)
    assert len(provider.requests) == 2


def test_revision_count_and_limit_allow_exact_number_of_reexecutions() -> None:
    provider = MockProvider()
    engine, task, execution, executor, _, _ = _review_pipeline(provider=provider)
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(ReviewDecision.REVISE, "Try again"),
        executor=executor,
        max_revisions=2,
    )
    outcome = result.manager_results[0].subtask_outcomes[0]
    assert outcome.status is ManagerReviewStatus.REVISION_LIMIT_REACHED
    assert result.status is ManagerReviewStatus.REVISION_LIMIT_REACHED
    assert outcome.revision_count == 2
    assert tuple(item.revision_number for item in outcome.revisions) == (1, 2)
    assert len(outcome.reviews) == 3
    assert len(provider.requests) == 3


def test_zero_revision_limit_stops_before_reexecution() -> None:
    provider = MockProvider()
    engine, task, execution, executor, _, _ = _review_pipeline(provider=provider)
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(ReviewDecision.REVISE, "Revise"),
        executor=executor,
        max_revisions=0,
    )
    outcome = result.manager_results[0].subtask_outcomes[0]
    assert outcome.revision_count == 0
    assert outcome.revisions == ()
    assert outcome.status is ManagerReviewStatus.REVISION_LIMIT_REACHED
    assert len(provider.requests) == 1


def test_feedback_reaches_revision_request_without_mutating_subtask() -> None:
    provider = MockProvider()
    engine, task, execution, executor, manager, _ = _review_pipeline(
        provider=provider,
    )
    original_subtask = execution.delegation.manager_delegations[0].subtasks[0]  # type: ignore[union-attr]
    original_metadata = original_subtask.metadata.copy()
    engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(outcomes=(
            (ReviewDecision.REVISE, "Use the reviewer feedback"),
            (ReviewDecision.PASS, "Accepted"),
        )),
        executor=executor,
    )
    revision_context = provider.requests[1].context["subtask"]["metadata"]["revision"]
    assert revision_context == {
        "manager_id": manager.id,
        "revision_number": 1,
        "feedback": "Use the reviewer feedback",
    }
    assert original_subtask.metadata == original_metadata
    assert provider.requests[1].context["subtask"]["id"] == original_subtask.id


def test_revision_reruns_all_originally_assigned_specialists_in_order() -> None:
    specialists = (
        SpecialistAgent(name="First", capabilities=("work",)),
        SpecialistAgent(name="Second", capabilities=("work",)),
    )
    provider = MockProvider()
    engine, task, execution, executor, _, _ = _review_pipeline(
        specialists=specialists, provider=provider,
    )
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(outcomes=(
            (ReviewDecision.REVISE, "Revise all"),
            (ReviewDecision.PASS, "Accepted"),
        )),
        executor=executor,
    )
    revision = result.manager_results[0].subtask_outcomes[0].revisions[0]
    assert revision.specialist_ids == tuple(item.id for item in specialists)
    assert tuple(item.specialist_id for item in revision.executions) == (
        specialists[0].id, specialists[1].id,
    )
    assert len(provider.requests) == 4


def test_failed_initial_execution_is_preserved_and_reviewable() -> None:
    provider = FailingProvider(ProviderConfig(provider_name="failing"))
    engine, task, execution, executor, _, _ = _review_pipeline(provider=provider)
    assert execution.manager_executions[0].specialist_executions[0].agent_result.success is False  # type: ignore[union-attr]
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(ReviewDecision.PASS),
        executor=executor,
    )
    outcome = result.manager_results[0].subtask_outcomes[0]
    assert outcome.status is ManagerReviewStatus.PASSED
    assert outcome.executions[0] is (
        execution.manager_executions[0].specialist_executions[0]
    )


def test_mixed_subtask_outcomes_aggregate_to_partial_in_order() -> None:
    templates = (
        SubtaskTemplate(objective="First", required_capabilities=("work",)),
        SubtaskTemplate(objective="Second", required_capabilities=("work",)),
    )
    engine, task, execution, executor, _, _ = _review_pipeline(
        templates=templates,
    )

    class KeyedReviewer(BaseManagerReviewer):
        def review(
            self, task: Task, subtask: Subtask, manager: ManagerAgent,
            specialist_executions: tuple[object, ...],
        ) -> ReviewResult:
            decision = (
                ReviewDecision.PASS
                if subtask.objective == "First"
                else ReviewDecision.FAIL
            )
            return ReviewResult(decision=decision, reviewer_id=manager.id)

    result = engine.review(
        execution, task, reviewer=KeyedReviewer(), executor=executor,
    )
    manager_result = result.manager_results[0]
    assert tuple(item.subtask_id for item in manager_result.subtask_outcomes) == tuple(
        item.id for item in execution.delegation.manager_delegations[0].subtasks  # type: ignore[union-attr]
    )
    assert tuple(item.status for item in manager_result.subtask_outcomes) == (
        ManagerReviewStatus.PASSED, ManagerReviewStatus.FAILED,
    )
    assert manager_result.status is ManagerReviewStatus.PARTIAL
    assert result.status is ManagerReviewStatus.PARTIAL


def test_multiple_managers_are_reviewed_in_execution_order() -> None:
    registry = CapabilityRegistry()
    first_manager = ManagerAgent(name="First manager", capabilities=("manage",))
    second_manager = ManagerAgent(name="Second manager", capabilities=("manage",))
    first_worker = SpecialistAgent(name="First worker", capabilities=("work",))
    second_worker = SpecialistAgent(name="Second worker", capabilities=("work",))
    for manager, specialist in (
        (first_manager, first_worker),
        (second_manager, second_worker),
    ):
        registry.register(manager)
        manager.register_specialist(specialist)
        registry.register(specialist)
    engine = OrchestrationEngine(
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        registry=registry,
    )
    task = Task(objective="Parent")
    delegation = engine.delegate(
        engine.orchestrate(task),
        task,
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate(objective="Work", required_capabilities=("work",)),
        )),
    )
    provider = MockProvider()
    providers = ProviderRegistry()
    providers.register(provider)
    executor = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={
            first_worker.id: provider.name,
            second_worker.id: provider.name,
        },
    )
    execution = engine.execute(delegation, task, executor=executor)
    calls: list[tuple[str, str]] = []

    class TrackingReviewer(BaseManagerReviewer):
        def review(
            self, task: Task, subtask: Subtask, manager: ManagerAgent,
            specialist_executions: tuple[object, ...],
        ) -> ReviewResult:
            calls.append((manager.id, subtask.id))
            return ReviewResult(
                decision=ReviewDecision.PASS, reviewer_id=manager.id,
            )

    result = engine.review(
        execution, task, reviewer=TrackingReviewer(), executor=executor,
    )
    assert tuple(item.manager_id for item in result.manager_results) == (
        first_manager.id, second_manager.id,
    )
    assert tuple(manager_id for manager_id, _ in calls) == (
        first_manager.id, second_manager.id,
    )


def test_review_does_not_mutate_inputs_or_existing_trace() -> None:
    provider = MockProvider()
    engine, task, execution, executor, manager, specialists = _review_pipeline(
        provider=provider,
    )
    delegation = execution.delegation
    subtask = delegation.manager_delegations[0].subtasks[0]  # type: ignore[union-attr]
    task_before = (
        task.id, task.objective, task.context.data.copy(), task.metadata.copy(),
        task.status, task.created_at,
    )
    subtask_before = (subtask.id, subtask.objective, subtask.metadata.copy())
    execution_before = (
        execution.manager_executions, execution.status,
        execution.trace.events, execution.metadata.copy(), execution.delegation,
    )
    registry_before = (
        engine.registry.agents, engine.registry.capabilities,
        engine.registry.capability_index, manager.specialists,
    )
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(outcomes=(
            (ReviewDecision.REVISE, "Revise"),
            (ReviewDecision.PASS, "Pass"),
        )),
        executor=executor,
    )
    assert (
        task.id, task.objective, task.context.data, task.metadata,
        task.status, task.created_at,
    ) == task_before
    assert (subtask.id, subtask.objective, subtask.metadata) == subtask_before
    assert (
        execution.manager_executions, execution.status,
        execution.trace.events, execution.metadata, execution.delegation,
    ) == execution_before
    assert (
        engine.registry.agents, engine.registry.capabilities,
        engine.registry.capability_index, manager.specialists,
    ) == registry_before
    assert tuple(item.id for item in specialists) == tuple(
        item.id for item in manager.specialists
    )
    assert result.trace is not execution.trace
    assert result.trace.events[:len(execution.trace.events)] == execution.trace.events


def test_pass_review_trace_order() -> None:
    engine, task, execution, executor, _, _ = _review_pipeline()
    before_count = len(execution.trace.events)
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(ReviewDecision.PASS),
        executor=executor,
    )
    assert tuple(event.event_type for event in result.trace.events[before_count:]) == (
        OrchestrationEventType.MANAGER_REVIEW_STARTED.value,
        OrchestrationEventType.MANAGER_REVIEW_PASSED.value,
        OrchestrationEventType.MANAGER_REVIEW_COMPLETED.value,
    )


def test_revision_trace_order() -> None:
    engine, task, execution, executor, _, _ = _review_pipeline()
    before_count = len(execution.trace.events)
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(outcomes=(
            (ReviewDecision.REVISE, "Revise"),
            (ReviewDecision.PASS, "Pass"),
        )),
        executor=executor,
    )
    assert tuple(event.event_type for event in result.trace.events[before_count:]) == (
        OrchestrationEventType.MANAGER_REVIEW_STARTED.value,
        OrchestrationEventType.MANAGER_REVIEW_REVISION_REQUESTED.value,
        OrchestrationEventType.REVISION_STARTED.value,
        OrchestrationEventType.SPECIALIST_EXECUTION_STARTED.value,
        OrchestrationEventType.SPECIALIST_EXECUTION_COMPLETED.value,
        OrchestrationEventType.REVISION_EXECUTION_COMPLETED.value,
        OrchestrationEventType.MANAGER_REVIEW_STARTED.value,
        OrchestrationEventType.MANAGER_REVIEW_PASSED.value,
        OrchestrationEventType.MANAGER_REVIEW_COMPLETED.value,
    )


def test_revision_limit_trace_event() -> None:
    engine, task, execution, executor, _, _ = _review_pipeline()
    before_count = len(execution.trace.events)
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(ReviewDecision.REVISE),
        executor=executor,
        max_revisions=0,
    )
    assert tuple(event.event_type for event in result.trace.events[before_count:]) == (
        OrchestrationEventType.MANAGER_REVIEW_STARTED.value,
        OrchestrationEventType.MANAGER_REVIEW_REVISION_REQUESTED.value,
        OrchestrationEventType.REVISION_LIMIT_REACHED.value,
        OrchestrationEventType.MANAGER_REVIEW_COMPLETED.value,
    )


def test_review_models_have_isolated_metadata_defaults() -> None:
    engine, task, execution, executor, _, _ = _review_pipeline()
    first = engine.review(
        execution, task, reviewer=StaticManagerReviewer(), executor=executor,
    )
    second = engine.review(
        execution, task, reviewer=StaticManagerReviewer(), executor=executor,
    )
    first.metadata["changed"] = True
    first.manager_results[0].metadata["changed"] = True
    first.manager_results[0].subtask_outcomes[0].metadata["changed"] = True
    assert second.metadata == {}
    assert second.manager_results[0].metadata == {}
    assert second.manager_results[0].subtask_outcomes[0].metadata == {}


def test_review_input_and_result_contract_validation() -> None:
    engine, task, execution, executor, manager, _ = _review_pipeline()

    class WrongReviewer(BaseManagerReviewer):
        def review(
            self, task: Task, subtask: Subtask, manager: ManagerAgent,
            specialist_executions: tuple[object, ...],
        ) -> ReviewResult:
            return ReviewResult(
                decision=ReviewDecision.PASS, reviewer_id="other",
            )

    with pytest.raises(ValueError, match="reviewer_id"):
        engine.review(
            execution, task, reviewer=WrongReviewer(), executor=executor,
        )
    with pytest.raises(ValueError, match="max_revisions"):
        engine.review(
            execution,
            task,
            reviewer=StaticManagerReviewer(),
            executor=executor,
            max_revisions=-1,
        )
    assert manager.id == execution.manager_executions[0].manager_id


def test_step_9_execution_result_reviews_without_conversion() -> None:
    engine, task, execution, executor, _, _ = _review_pipeline()
    assert execution.delegation is not None
    result = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(ReviewDecision.PASS),
        executor=executor,
    )
    assert result.task_id == execution.task_id
    assert result.status is ManagerReviewStatus.PASSED
    assert {item.name for item in fields(type(result))} >= {
        "task_id", "manager_results", "status", "trace", "metadata",
    }
