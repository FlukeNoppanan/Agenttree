"""Workflow state snapshots and hardened execution trace tests."""

from datetime import datetime

import pytest

from agenttree.agents import ManagerAgent, RootAgent, SpecialistAgent
from agenttree.core import (
    ProviderSpecialistExecutor,
    RuleBasedTaskTriage,
    StaticFinalReviewer,
    StaticManagerReviewer,
    StaticTaskDecomposer,
    WorkflowStateManager,
)
from agenttree.models import (
    ExecutionEvent,
    ExecutionTrace,
    SubtaskTemplate,
    Task,
    TaskStatus,
    TriageResult,
    WorkflowPhase,
    WorkflowState,
)
from agenttree.orchestration import OrchestrationEngine, OrchestrationEventType
from agenttree.providers import MockProvider, ProviderRegistry
from agenttree.registry import CapabilityRegistry
from agenttree.tracing import ExecutionEventType


def _advance(
    manager: WorkflowStateManager,
    state: WorkflowState,
    target: WorkflowPhase,
) -> WorkflowState:
    phases = (
        WorkflowPhase.TRIAGE,
        WorkflowPhase.PLANNING,
        WorkflowPhase.DELEGATION,
        WorkflowPhase.EXECUTION,
        WorkflowPhase.MANAGER_REVIEW,
        WorkflowPhase.FINAL_REVIEW,
        WorkflowPhase.COMPLETED,
    )
    for phase in phases:
        state = manager.transition(state, phase)
        if phase is target:
            return state
    raise AssertionError("target phase not reached")


def _full_pipeline() -> tuple[Task, tuple[object, ...]]:
    registry = CapabilityRegistry()
    specialist = SpecialistAgent(name="Worker", capabilities=("work",))
    manager_agent = ManagerAgent(name="Manager", capabilities=("manage",))
    manager_agent.register_specialist(specialist)
    registry.register(manager_agent)
    registry.register(specialist)
    engine = OrchestrationEngine(
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        registry=registry,
    )
    task = Task(objective="Process structured input")
    triage = engine.triage.triage(task)
    plan = engine.orchestrate(task)
    delegation = engine.delegate(
        plan,
        task,
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate(objective="Work", required_capabilities=("work",)),
        )),
    )
    provider = MockProvider(response_content="done")
    providers = ProviderRegistry()
    providers.register(provider)
    executor = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={specialist.id: provider.name},
    )
    execution = engine.execute(delegation, task, executor=executor)
    manager_review = engine.review(
        execution,
        task,
        reviewer=StaticManagerReviewer(),
        executor=executor,
    )
    final = engine.finalize(
        manager_review,
        task,
        root_agent=RootAgent(name="Root"),
        final_reviewer=StaticFinalReviewer(),
        manager_reviewer=StaticManagerReviewer(),
        executor=executor,
    )
    return task, (triage, plan, delegation, execution, manager_review, final)


def test_initial_workflow_state_creation_is_received_and_isolated() -> None:
    task = Task(objective="Work")
    metadata = {"input": {"items": [1, 2]}}
    trace = ExecutionTrace(task_id=task.id)
    state = WorkflowStateManager().create_initial(
        task, trace=trace, metadata=metadata,
    )
    metadata["input"]["items"].append(3)
    trace.append(ExecutionEvent(event_type="custom.after", task_id=task.id))
    assert state.task_id == task.id
    assert state.current_phase is WorkflowPhase.RECEIVED
    assert state.status is TaskStatus.PENDING
    assert state.metadata["input"]["items"] == (1, 2)
    assert state.trace.event_count == 0


def test_workflow_phase_values_are_distinct_from_stage_statuses() -> None:
    assert tuple(phase.value for phase in WorkflowPhase) == (
        "received",
        "triage",
        "planning",
        "delegation",
        "execution",
        "manager_review",
        "final_review",
        "completed",
        "failed",
    )


def test_valid_phase_transitions_follow_current_engine_order() -> None:
    manager = WorkflowStateManager()
    state = manager.create_initial(Task(objective="Work"))
    completed = _advance(manager, state, WorkflowPhase.COMPLETED)
    assert completed.current_phase is WorkflowPhase.COMPLETED


def test_invalid_phase_transition_is_rejected() -> None:
    manager = WorkflowStateManager()
    state = manager.create_initial(Task(objective="Work"))
    with pytest.raises(ValueError, match="received -> execution"):
        manager.transition(state, WorkflowPhase.EXECUTION)


def test_completed_and_failed_states_are_terminal() -> None:
    manager = WorkflowStateManager()
    initial = manager.create_initial(Task(objective="Work"))
    completed = _advance(manager, initial, WorkflowPhase.COMPLETED)
    failed = manager.fail(initial, reason="Invalid input")
    with pytest.raises(ValueError, match="Terminal"):
        manager.transition(completed, WorkflowPhase.FAILED)
    with pytest.raises(ValueError, match="Terminal"):
        manager.transition(failed, WorkflowPhase.TRIAGE)


def test_failure_preserves_reason_phase_results_and_trace() -> None:
    task = Task(objective="Work")
    state_manager = WorkflowStateManager()
    state = state_manager.create_initial(task)
    triage = TriageResult(task_id=task.id, objective=task.objective)
    state = state_manager.transition(
        state, WorkflowPhase.TRIAGE, result=triage,
    )
    trace = state.trace.clone()
    trace.append(ExecutionEvent(event_type="custom.failure", task_id=task.id))
    failed = state_manager.fail(
        state,
        reason="Unable to plan",
        trace=trace,
        metadata={"error_type": "PlanningError"},
    )
    assert failed.current_phase is WorkflowPhase.FAILED
    assert failed.failed_phase is WorkflowPhase.TRIAGE
    assert failed.failure_reason == "Unable to plan"
    assert failed.status is TaskStatus.FAILED
    assert failed.triage_result == triage
    assert failed.trace.last_event.event_type == "custom.failure"  # type: ignore[union-attr]
    assert failed.metadata["failure"] == {
        "phase": "triage", "reason": "Unable to plan",
    }


def test_transition_preserves_previous_state_and_metadata() -> None:
    task = Task(objective="Work")
    manager = WorkflowStateManager()
    first = manager.create_initial(task, metadata={"nested": {"value": 1}})
    second = manager.transition(
        first, WorkflowPhase.TRIAGE, metadata={"attempt": 1},
    )
    assert first.current_phase is WorkflowPhase.RECEIVED
    assert "attempt" not in first.metadata
    assert second.metadata["nested"]["value"] == 1
    assert second.metadata["attempt"] == 1
    with pytest.raises(TypeError):
        second.metadata["attempt"] = 2  # type: ignore[index]


def test_state_attaches_a_copied_phase_result() -> None:
    task = Task(objective="Original")
    triage = TriageResult(
        task_id=task.id,
        objective="Original",
        metadata={"nested": {"value": 1}},
    )
    manager = WorkflowStateManager()
    state = manager.transition(
        manager.create_initial(task),
        WorkflowPhase.TRIAGE,
        result=triage,
    )
    triage.metadata["nested"]["value"] = 2
    assert state.triage_result is not triage
    assert state.triage_result.metadata["nested"]["value"] == 1  # type: ignore[union-attr]


def test_state_and_trace_task_ids_are_validated() -> None:
    manager = WorkflowStateManager()
    task = Task(objective="Work")
    with pytest.raises(ValueError, match="trace task_id"):
        manager.create_initial(task, trace=ExecutionTrace(task_id="other"))
    with pytest.raises(ValueError, match="triage_result"):
        WorkflowState(
            task_id=task.id,
            current_phase=WorkflowPhase.TRIAGE,
            status=None,
            trace=ExecutionTrace(task_id=task.id),
            triage_result=TriageResult(task_id="other", objective="Work"),
        )


def test_state_trace_is_read_only_but_can_be_cloned_for_new_work() -> None:
    task = Task(objective="Work")
    state = WorkflowStateManager().create_initial(task)
    assert state.trace.is_read_only is True
    with pytest.raises(TypeError, match="read-only"):
        state.trace.append(ExecutionEvent(event_type="custom", task_id=task.id))
    working_trace = state.trace.clone()
    working_trace.append(ExecutionEvent(event_type="custom", task_id=task.id))
    assert state.trace.event_count == 0
    assert working_trace.event_count == 1


def test_trace_clone_preserves_order_and_isolates_mutation() -> None:
    source = ExecutionTrace(task_id="task")
    metadata = {"nested": {"value": 1}}
    first = ExecutionEvent(
        event_type="first", task_id="task", metadata=metadata,
    )
    source.append(first)
    metadata["nested"]["value"] = 2
    first.metadata["nested"]["value"] = 3
    clone = source.clone()
    clone.events[0].metadata["nested"]["value"] = 4
    clone.append(ExecutionEvent(event_type="second", task_id="task"))
    assert tuple(event.event_type for event in source.events) == ("first",)
    assert source.events[0].metadata["nested"]["value"] == 1
    assert tuple(event.event_type for event in clone.events) == ("first", "second")


def test_trace_query_helpers_return_isolated_ordered_snapshots() -> None:
    trace = ExecutionTrace(task_id="task")
    events = (
        ExecutionEvent(event_type="one", task_id="task", actor_id="a"),
        ExecutionEvent(event_type="two", task_id="task", actor_id="b"),
        ExecutionEvent(event_type="one", task_id="task", actor_id="a"),
    )
    for event in events:
        trace.append(event)
    assert trace.event_count == 3
    assert trace.last_event == events[-1]
    assert trace.events_by_type("one") == (events[0], events[2])
    assert trace.events_by_actor("a") == (events[0], events[2])
    queried = trace.events_by_type("one")
    queried[0].metadata["changed"] = True
    assert trace.events_by_type("one")[0].metadata == {}


def test_known_event_contract_preserves_old_name_and_custom_events() -> None:
    assert OrchestrationEventType is ExecutionEventType
    assert ExecutionEventType.STARTED.value == "orchestration.started"
    assert (
        ExecutionEventType.SPECIALIST_EXECUTION_STARTED.value
        == "orchestration.specialist_execution_started"
    )
    known = ExecutionEvent(
        event_type=ExecutionEventType.FINAL_RESULT_CREATED,
        task_id="task",
    )
    custom = ExecutionEvent(event_type="host.custom_event", task_id="task")
    trace = ExecutionTrace(task_id="task")
    trace.append(known)
    trace.append(custom)
    assert trace.events_by_type(ExecutionEventType.FINAL_RESULT_CREATED) == (known,)
    assert trace.events_by_type("host.custom_event") == (custom,)


def test_event_requires_aware_timestamp_and_nonempty_identity() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ExecutionEvent(
            event_type="custom", task_id="task", timestamp=datetime.now(),
        )
    with pytest.raises(ValueError, match="event_type"):
        ExecutionEvent(event_type=" ", task_id="task")
    with pytest.raises(ValueError, match="task_id"):
        ExecutionTrace(task_id=" ")


def test_event_trace_and_state_serialization_are_deterministic() -> None:
    event = ExecutionEvent(
        event_type="custom",
        task_id="task",
        metadata={"values": [1, 2]},
    )
    trace = ExecutionTrace(task_id="task")
    trace.append(event)
    state = WorkflowState(
        task_id="task",
        current_phase=WorkflowPhase.RECEIVED,
        status=TaskStatus.PENDING,
        trace=trace,
        metadata={"nested": {"active": True}},
    )
    assert event.to_dict() == event.to_dict()
    assert trace.to_dict() == trace.to_dict()
    assert state.to_dict() == state.to_dict()
    assert state.to_dict()["status"] == "pending"
    assert state.to_dict()["trace"]["events"][0]["metadata"] == {
        "values": [1, 2],
    }


def test_existing_engine_outputs_attach_without_api_changes() -> None:
    task, results = _full_pipeline()
    manager = WorkflowStateManager()
    state = manager.create_initial(task)
    phases = (
        WorkflowPhase.TRIAGE,
        WorkflowPhase.PLANNING,
        WorkflowPhase.DELEGATION,
        WorkflowPhase.EXECUTION,
        WorkflowPhase.MANAGER_REVIEW,
        WorkflowPhase.FINAL_REVIEW,
    )
    snapshots = [state]
    for phase, result in zip(phases, results):
        state = manager.transition(state, phase, result=result)
        snapshots.append(state)
    state = manager.transition(state, WorkflowPhase.COMPLETED)
    assert state.current_phase is WorkflowPhase.COMPLETED
    assert state.final_result.task_id == task.id  # type: ignore[union-attr]
    assert state.trace.event_count == results[-1].trace.event_count
    assert snapshots[2].orchestration_plan is not None
    assert snapshots[2].trace.event_count < state.trace.event_count
