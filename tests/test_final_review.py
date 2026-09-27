"""Root final review, revision, and normalized result tests."""

from copy import deepcopy
from dataclasses import fields

import pytest

from agenttree.agents import ManagerAgent, RootAgent, SpecialistAgent
from agenttree.core import (
    BaseFinalReviewer,
    BaseManagerReviewer,
    ProviderSpecialistExecutor,
    RuleBasedTaskTriage,
    StaticFinalReviewer,
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
    FinalResult,
    FinalStatus,
    ManagerReviewStatus,
    OrchestrationEngine,
    OrchestrationEventType,
    TaskExecutionResult,
    TaskManagerReviewResult,
)
from agenttree.providers import MockProvider, ProviderRegistry
from agenttree.registry import CapabilityRegistry


def _pipeline(
    *,
    templates: tuple[SubtaskTemplate, ...] | None = None,
    specialists: tuple[SpecialistAgent, ...] | None = None,
    initial_reviewer: BaseManagerReviewer | None = None,
    manager_count: int = 1,
) -> tuple[
    OrchestrationEngine,
    Task,
    TaskManagerReviewResult,
    ProviderSpecialistExecutor,
    RootAgent,
    MockProvider,
]:
    templates = templates or (
        SubtaskTemplate(objective="Work", required_capabilities=("work",)),
    )
    if manager_count < 1:
        raise ValueError("manager_count must be positive")
    if manager_count > 1 and specialists is not None:
        raise ValueError("custom specialists are supported for one manager")
    registry = CapabilityRegistry()
    managers = tuple(
        ManagerAgent(name=f"Manager {index + 1}", capabilities=("manage",))
        for index in range(manager_count)
    )
    all_specialists: list[SpecialistAgent] = []
    for index, manager in enumerate(managers):
        registry.register(manager)
        owned = specialists or (
            SpecialistAgent(
                name=f"Worker {index + 1}", capabilities=("work",),
            ),
        )
        for specialist in owned:
            manager.register_specialist(specialist)
            registry.register(specialist)
            all_specialists.append(specialist)
    engine = OrchestrationEngine(
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        registry=registry,
    )
    task = Task(
        objective="Parent",
        context=TaskContext(data={"nested": {"value": 1}}),
        metadata={"source": {"id": 1}},
    )
    delegation = engine.delegate(
        engine.orchestrate(task),
        task,
        decomposer=StaticTaskDecomposer(templates),
    )
    provider = MockProvider(response_content="accepted output")
    providers = ProviderRegistry()
    providers.register(provider)
    executor = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={item.id: provider.name for item in all_specialists},
    )
    execution = engine.execute(delegation, task, executor=executor)
    manager_result = engine.review(
        execution,
        task,
        reviewer=initial_reviewer or StaticManagerReviewer(),
        executor=executor,
    )
    return engine, task, manager_result, executor, RootAgent(name="Root"), provider


def _finalize(
    *,
    final_reviewer: BaseFinalReviewer | None = None,
    manager_reviewer: BaseManagerReviewer | None = None,
    max_manager_revisions: int = 2,
    max_final_revisions: int = 1,
    pipeline: tuple[
        OrchestrationEngine,
        Task,
        TaskManagerReviewResult,
        ProviderSpecialistExecutor,
        RootAgent,
        MockProvider,
    ] | None = None,
) -> tuple[FinalResult, tuple[object, ...]]:
    values = pipeline or _pipeline()
    engine, task, manager_result, executor, root, _ = values
    result = engine.finalize(
        manager_result,
        task,
        root_agent=root,
        final_reviewer=final_reviewer or StaticFinalReviewer(),
        manager_reviewer=manager_reviewer or StaticManagerReviewer(),
        executor=executor,
        max_manager_revisions=max_manager_revisions,
        max_final_revisions=max_final_revisions,
    )
    return result, values


def test_base_final_reviewer_is_an_abstract_contract() -> None:
    with pytest.raises(TypeError):
        BaseFinalReviewer()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "decision",
    (ReviewDecision.PASS, ReviewDecision.REVISE, ReviewDecision.FAIL),
)
def test_static_final_reviewer_returns_configured_outcome(
    decision: ReviewDecision,
) -> None:
    _, task, manager_result, _, root, _ = _pipeline()
    review = StaticFinalReviewer(decision, "Root feedback").review(
        task, root, manager_result,
    )
    assert review.decision is decision
    assert review.feedback == "Root feedback"
    assert review.reviewer_id == root.id


def test_static_final_reviewer_sequence_is_independent_per_task() -> None:
    first_pipeline = _pipeline()
    second_pipeline = _pipeline()
    reviewer = StaticFinalReviewer(outcomes=(
        (ReviewDecision.REVISE, "Revise"),
        (ReviewDecision.PASS, "Pass"),
    ))
    assert reviewer.review(
        first_pipeline[1], first_pipeline[4], first_pipeline[2],
    ).decision is ReviewDecision.REVISE
    assert reviewer.review(
        second_pipeline[1], second_pipeline[4], second_pipeline[2],
    ).decision is ReviewDecision.REVISE
    assert reviewer.review(
        first_pipeline[1], first_pipeline[4], first_pipeline[2],
    ).decision is ReviewDecision.PASS


def test_pass_creates_successful_structured_final_result() -> None:
    result, values = _finalize()
    _, task, manager_result, _, root, _ = values
    assert result.task_id == task.id
    assert result.root_agent_id == root.id
    assert result.success is True
    assert result.status is FinalStatus.COMPLETED
    assert result.final_review.decision is ReviewDecision.PASS
    assert result.manager_results == manager_result.manager_results
    assert result.content["managers"][0]["subtasks"][0]["specialists"][0][
        "output"
    ] == "accepted output"


def test_fail_creates_unsuccessful_terminal_result_and_preserves_feedback() -> None:
    result, _ = _finalize(
        final_reviewer=StaticFinalReviewer(ReviewDecision.FAIL, "Reject task"),
    )
    assert result.success is False
    assert result.status is FinalStatus.FAILED
    assert result.final_review.feedback == "Reject task"
    assert result.revision_count == 0


def test_revise_reuses_manager_review_for_every_manager_and_carries_feedback() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class TrackingManagerReviewer(BaseManagerReviewer):
        def review(
            self,
            task: Task,
            subtask: Subtask,
            manager: ManagerAgent,
            specialist_executions: tuple[object, ...],
        ) -> ReviewResult:
            calls.append((manager.id, deepcopy(task.metadata["final_revision"])))
            return ReviewResult(
                decision=ReviewDecision.PASS,
                reviewer_id=manager.id,
            )

    pipeline = _pipeline(manager_count=2)
    result, values = _finalize(
        final_reviewer=StaticFinalReviewer(outcomes=(
            (ReviewDecision.REVISE, "Reconsider all work"),
            (ReviewDecision.PASS, "Accepted"),
        )),
        manager_reviewer=TrackingManagerReviewer(),
        pipeline=pipeline,
    )
    _, _, original, _, root, _ = values
    assert len(calls) == len(original.manager_results)
    assert calls[0][1] == {
        "root_agent_id": root.id,
        "revision_number": 1,
        "feedback": "Reconsider all work",
    }
    assert result.revision_count == 1
    assert result.revisions[0].feedback == "Reconsider all work"
    assert tuple(item[0] for item in calls) == tuple(
        item.manager_id for item in original.manager_results
    )


def test_final_content_preserves_manager_order() -> None:
    pipeline = _pipeline(manager_count=2)
    result, _ = _finalize(pipeline=pipeline)
    assert tuple(item["manager_id"] for item in result.content["managers"]) == tuple(
        item.manager_id for item in pipeline[2].manager_results
    )


def test_final_revision_count_allows_exact_configured_rounds() -> None:
    result, _ = _finalize(
        final_reviewer=StaticFinalReviewer(outcomes=(
            (ReviewDecision.REVISE, "First"),
            (ReviewDecision.REVISE, "Second"),
            (ReviewDecision.PASS, "Done"),
        )),
        max_final_revisions=2,
    )
    assert result.success is True
    assert result.revision_count == 2
    assert tuple(item.revision_number for item in result.revisions) == (1, 2)
    assert tuple(item.decision for item in result.final_reviews) == (
        ReviewDecision.REVISE,
        ReviewDecision.REVISE,
        ReviewDecision.PASS,
    )


def test_final_revision_limit_is_explicit_and_bounded() -> None:
    result, _ = _finalize(
        final_reviewer=StaticFinalReviewer(
            ReviewDecision.REVISE, "Still incomplete",
        ),
        max_final_revisions=1,
    )
    assert result.success is False
    assert result.status is FinalStatus.FINAL_REVISION_LIMIT_REACHED
    assert result.revision_count == 1
    assert len(result.final_reviews) == 2


def test_zero_final_revision_limit_performs_no_manager_reconsideration() -> None:
    result, values = _finalize(
        final_reviewer=StaticFinalReviewer(ReviewDecision.REVISE),
        max_final_revisions=0,
    )
    assert result.revision_count == 0
    assert result.revisions == ()
    assert result.manager_results == values[2].manager_results


def test_manager_and_final_revision_counts_are_separate() -> None:
    result, values = _finalize(
        final_reviewer=StaticFinalReviewer(outcomes=(
            (ReviewDecision.REVISE, "Root revision"),
            (ReviewDecision.PASS, "Root pass"),
        )),
        manager_reviewer=StaticManagerReviewer(outcomes=(
            (ReviewDecision.REVISE, "Manager revision"),
            (ReviewDecision.PASS, "Manager pass"),
        )),
    )
    assert result.revision_count == 1
    assert result.manager_results[0].subtask_outcomes[0].revision_count == 1
    assert len(values[5].requests) == 2


def test_final_content_preserves_manager_subtask_and_specialist_order() -> None:
    specialists = (
        SpecialistAgent(name="First", capabilities=("work",)),
        SpecialistAgent(name="Second", capabilities=("work",)),
    )
    templates = (
        SubtaskTemplate(objective="One", required_capabilities=("work",)),
        SubtaskTemplate(objective="Two", required_capabilities=("work",)),
    )
    pipeline = _pipeline(templates=templates, specialists=specialists)
    result, _ = _finalize(pipeline=pipeline)
    manager_content = result.content["managers"][0]
    expected_subtasks = pipeline[2].manager_results[0].subtask_outcomes
    assert tuple(item["subtask_id"] for item in manager_content["subtasks"]) == tuple(
        item.subtask_id for item in expected_subtasks
    )
    for subtask in manager_content["subtasks"]:
        assert tuple(item["specialist_id"] for item in subtask["specialists"]) == (
            specialists[0].id,
            specialists[1].id,
        )


def test_partial_and_failed_manager_outcomes_remain_transparent() -> None:
    templates = (
        SubtaskTemplate(objective="Pass", required_capabilities=("work",)),
        SubtaskTemplate(objective="Fail", required_capabilities=("work",)),
    )

    class MixedReviewer(BaseManagerReviewer):
        def review(
            self,
            task: Task,
            subtask: Subtask,
            manager: ManagerAgent,
            specialist_executions: tuple[object, ...],
        ) -> ReviewResult:
            decision = (
                ReviewDecision.PASS
                if subtask.objective == "Pass"
                else ReviewDecision.FAIL
            )
            return ReviewResult(decision=decision, reviewer_id=manager.id)

    pipeline = _pipeline(
        templates=templates,
        initial_reviewer=MixedReviewer(),
    )
    result, _ = _finalize(pipeline=pipeline)
    assert pipeline[2].status is ManagerReviewStatus.PARTIAL
    assert result.status is FinalStatus.PARTIAL
    assert result.success is True
    assert result.manager_results == pipeline[2].manager_results
    subtasks = result.content["managers"][0]["subtasks"]
    assert subtasks[0]["specialists"]
    assert subtasks[1]["status"] == ManagerReviewStatus.FAILED.value
    assert subtasks[1]["specialists"] == []


def test_failed_manager_result_is_preserved_after_final_pass() -> None:
    pipeline = _pipeline(
        initial_reviewer=StaticManagerReviewer(
            ReviewDecision.FAIL, "Manager rejected work",
        ),
    )
    result, _ = _finalize(pipeline=pipeline)
    assert pipeline[2].status is ManagerReviewStatus.FAILED
    assert result.status is FinalStatus.PARTIAL
    assert result.success is False
    assert result.manager_results[0].status is ManagerReviewStatus.FAILED
    assert result.manager_results[0].subtask_outcomes[0].feedback == (
        "Manager rejected work"
    )


def test_finalization_does_not_mutate_task_manager_result_or_trace() -> None:
    pipeline = _pipeline()
    _, task, manager_result, _, root, _ = pipeline
    task_before = deepcopy(task)
    manager_results_before = deepcopy(manager_result.manager_results)
    trace_before = manager_result.trace.events
    root_before = deepcopy(root)
    result, _ = _finalize(
        pipeline=pipeline,
        final_reviewer=StaticFinalReviewer(outcomes=(
            (ReviewDecision.REVISE, "Revise"),
            (ReviewDecision.PASS, "Pass"),
        )),
    )
    assert task == task_before
    assert root == root_before
    assert manager_result.manager_results == manager_results_before
    assert manager_result.trace.events == trace_before
    assert result.trace is not manager_result.trace
    assert result.trace.events[:len(trace_before)] == trace_before


def test_pass_and_fail_trace_event_order() -> None:
    for decision, expected in (
        (ReviewDecision.PASS, OrchestrationEventType.FINAL_REVIEW_PASSED),
        (ReviewDecision.FAIL, OrchestrationEventType.FINAL_REVIEW_FAILED),
    ):
        pipeline = _pipeline()
        before = len(pipeline[2].trace.events)
        result, _ = _finalize(
            pipeline=pipeline,
            final_reviewer=StaticFinalReviewer(decision),
        )
        assert tuple(event.event_type for event in result.trace.events[before:]) == (
            OrchestrationEventType.FINAL_REVIEW_STARTED.value,
            expected.value,
            OrchestrationEventType.FINAL_RESULT_CREATED.value,
        )


def test_final_revision_trace_events_wrap_reused_manager_review() -> None:
    pipeline = _pipeline()
    before = len(pipeline[2].trace.events)
    result, _ = _finalize(
        pipeline=pipeline,
        final_reviewer=StaticFinalReviewer(outcomes=(
            (ReviewDecision.REVISE, "Revise"),
            (ReviewDecision.PASS, "Pass"),
        )),
    )
    events = tuple(event.event_type for event in result.trace.events[before:])
    assert events[:3] == (
        OrchestrationEventType.FINAL_REVIEW_STARTED.value,
        OrchestrationEventType.FINAL_REVIEW_REVISION_REQUESTED.value,
        OrchestrationEventType.FINAL_REVISION_STARTED.value,
    )
    assert OrchestrationEventType.MANAGER_REVIEW_STARTED.value in events
    assert events[-4:] == (
        OrchestrationEventType.FINAL_REVISION_COMPLETED.value,
        OrchestrationEventType.FINAL_REVIEW_STARTED.value,
        OrchestrationEventType.FINAL_REVIEW_PASSED.value,
        OrchestrationEventType.FINAL_RESULT_CREATED.value,
    )


def test_final_revision_limit_trace_event_is_terminal() -> None:
    pipeline = _pipeline()
    before = len(pipeline[2].trace.events)
    result, _ = _finalize(
        pipeline=pipeline,
        final_reviewer=StaticFinalReviewer(ReviewDecision.REVISE),
        max_final_revisions=0,
    )
    assert tuple(event.event_type for event in result.trace.events[before:]) == (
        OrchestrationEventType.FINAL_REVIEW_STARTED.value,
        OrchestrationEventType.FINAL_REVIEW_REVISION_REQUESTED.value,
        OrchestrationEventType.FINAL_REVISION_LIMIT_REACHED.value,
        OrchestrationEventType.FINAL_RESULT_CREATED.value,
    )


def test_final_reviewer_contract_and_limits_are_validated() -> None:
    pipeline = _pipeline()
    engine, task, manager_result, executor, root, _ = pipeline

    class WrongReviewer(BaseFinalReviewer):
        def review(
            self,
            task: Task,
            root_agent: RootAgent,
            manager_review_result: TaskManagerReviewResult,
        ) -> ReviewResult:
            return ReviewResult(
                decision=ReviewDecision.PASS,
                reviewer_id="another-root",
            )

    with pytest.raises(ValueError, match="RootAgent id"):
        engine.finalize(
            manager_result,
            task,
            root_agent=root,
            final_reviewer=WrongReviewer(),
            manager_reviewer=StaticManagerReviewer(),
            executor=executor,
        )
    with pytest.raises(ValueError, match="max_final_revisions"):
        engine.finalize(
            manager_result,
            task,
            root_agent=root,
            final_reviewer=StaticFinalReviewer(),
            manager_reviewer=StaticManagerReviewer(),
            executor=executor,
            max_final_revisions=-1,
        )


def test_step_10_result_finalizes_without_conversion() -> None:
    pipeline = _pipeline()
    result, _ = _finalize(pipeline=pipeline)
    assert result.manager_results == pipeline[2].manager_results
    assert {item.name for item in fields(FinalResult)} >= {
        "task_id",
        "success",
        "content",
        "status",
        "manager_results",
        "final_review",
        "revision_count",
        "trace",
        "metadata",
    }
