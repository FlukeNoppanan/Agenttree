"""Subtask decomposition and specialist delegation tests."""

from dataclasses import FrozenInstanceError, replace
from uuid import UUID

import pytest

from agenttree.agents import ManagerAgent, SpecialistAgent
from agenttree.core import (
    BaseTaskDecomposer,
    RuleBasedTaskTriage,
    StaticTaskDecomposer,
)
from agenttree.models import Subtask, SubtaskTemplate, Task, TaskStatus, TriageResult
from agenttree.orchestration import (
    DelegationStatus,
    OrchestrationEngine,
    OrchestrationEventType,
    OrchestrationStatus,
)
from agenttree.registry import CapabilityRegistry


def _build_engine(
    registry: CapabilityRegistry,
    capabilities: tuple[str, ...] = ("manage",),
) -> OrchestrationEngine:
    return OrchestrationEngine(
        triage=RuleBasedTaskTriage({}, fallback_capabilities=capabilities),
        registry=registry,
    )


def _register_owned(
    registry: CapabilityRegistry,
    manager: ManagerAgent,
    *specialists: SpecialistAgent,
) -> None:
    for specialist in specialists:
        manager.register_specialist(specialist)
        registry.register(specialist)


def test_subtask_creation_normalizes_capabilities_and_has_stable_identity() -> None:
    subtask = Subtask(
        parent_task_id="task-1",
        manager_id="manager-1",
        objective="  Prepare result  ",
        required_capabilities=(" Analyze ", "analyze", "FORMAT"),
        metadata={"input": {"version": 1}},
    )
    assert UUID(subtask.id).version == 4
    assert subtask.parent_task_id == "task-1"
    assert subtask.manager_id == "manager-1"
    assert subtask.objective == "Prepare result"
    assert subtask.required_capabilities == ("Analyze", "FORMAT")
    with pytest.raises(FrozenInstanceError):
        setattr(subtask, "id", "replacement")


def test_subtask_default_ids_are_unique() -> None:
    first = Subtask(parent_task_id="task", manager_id="manager", objective="First")
    second = Subtask(parent_task_id="task", manager_id="manager", objective="Second")
    assert first.id != second.id


def test_subtask_mutable_defaults_are_independent() -> None:
    first = Subtask(parent_task_id="task", manager_id="manager", objective="First")
    second = Subtask(parent_task_id="task", manager_id="manager", objective="Second")
    first.metadata["changed"] = True
    assert second.metadata == {}


def test_base_decomposer_is_an_abstract_contract() -> None:
    with pytest.raises(TypeError):
        BaseTaskDecomposer()  # type: ignore[abstract]


def test_static_decomposer_preserves_template_order_and_links_subtasks() -> None:
    templates = (
        SubtaskTemplate(objective="First", required_capabilities=("alpha",)),
        SubtaskTemplate(
            objective="Second", required_capabilities=("beta",),
            metadata={"sequence": 2},
        ),
    )
    decomposer = StaticTaskDecomposer(templates)
    task = Task(objective="Parent")
    manager = ManagerAgent(name="Manager")
    triage = TriageResult(task_id=task.id, objective=task.objective)
    subtasks = decomposer.decompose(task, manager, triage)
    assert decomposer.templates == templates
    assert tuple(item.objective for item in subtasks) == ("First", "Second")
    assert all(item.parent_task_id == task.id for item in subtasks)
    assert all(item.manager_id == manager.id for item in subtasks)
    assert subtasks[1].metadata == {"sequence": 2}


def test_static_decomposer_creates_fresh_subtasks_and_metadata() -> None:
    template = SubtaskTemplate(objective="Work", metadata={"sequence": 1})
    decomposer = StaticTaskDecomposer((template,))
    task = Task(objective="Parent")
    manager = ManagerAgent(name="Manager")
    triage = TriageResult(task_id=task.id, objective=task.objective)
    first = decomposer.decompose(task, manager, triage)[0]
    second = decomposer.decompose(task, manager, triage)[0]
    first.metadata["changed"] = True
    assert first.id != second.id
    assert second.metadata == {"sequence": 1}
    assert template.metadata == {"sequence": 1}


@pytest.mark.parametrize("invalid", ["template", (object(),)])
def test_static_decomposer_rejects_invalid_templates(invalid: object) -> None:
    with pytest.raises(TypeError, match="SubtaskTemplate"):
        StaticTaskDecomposer(invalid)  # type: ignore[arg-type]


def test_delegate_selects_only_owned_specialists() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    owned = SpecialistAgent(name="Owned", capabilities=("work",))
    unrelated = SpecialistAgent(name="Unrelated", capabilities=("work",))
    registry.register(manager)
    _register_owned(registry, manager, owned)
    registry.register(unrelated)
    task = Task(objective="Parent")
    engine = _build_engine(registry)
    delegation = engine.delegate(
        engine.orchestrate(task), task,
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate(objective="Work", required_capabilities=("work",)),
        )),
    )
    assignment = delegation.manager_delegations[0].assignments[0]
    assert assignment.specialist_ids == (owned.id,)
    assert unrelated.id not in assignment.specialist_ids


def test_specialist_selection_uses_registry_order_after_ownership_filter() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    first = SpecialistAgent(name="First", capabilities=("work",))
    second = SpecialistAgent(name="Second", capabilities=("work",))
    registry.register(manager)
    registry.register(first)
    registry.register(second)
    manager.register_specialist(second)
    manager.register_specialist(first)
    task = Task(objective="Parent")
    engine = _build_engine(registry)
    result = engine.delegate(
        engine.orchestrate(task), task,
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate(objective="Work", required_capabilities=("work",)),
        )),
    )
    assert result.manager_delegations[0].assignments[0].specialist_ids == (
        first.id, second.id,
    )


def test_specialist_any_matching_is_default() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    alpha = SpecialistAgent(name="Alpha", capabilities=("alpha",))
    beta = SpecialistAgent(name="Beta", capabilities=("beta",))
    both = SpecialistAgent(name="Both", capabilities=("alpha", "beta"))
    registry.register(manager)
    _register_owned(registry, manager, alpha, beta, both)
    task = Task(objective="Parent")
    engine = _build_engine(registry)
    result = engine.delegate(
        engine.orchestrate(task), task,
        decomposer=StaticTaskDecomposer((SubtaskTemplate(
            objective="Work", required_capabilities=("alpha", "beta"),
        ),)),
    )
    assert result.manager_delegations[0].assignments[0].specialist_ids == (
        alpha.id, beta.id, both.id,
    )


def test_specialist_all_matching_can_be_configured() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    partial = SpecialistAgent(name="Partial", capabilities=("alpha",))
    complete = SpecialistAgent(name="Complete", capabilities=("alpha", "beta"))
    registry.register(manager)
    _register_owned(registry, manager, partial, complete)
    task = Task(objective="Parent")
    engine = _build_engine(registry)
    result = engine.delegate(
        engine.orchestrate(task), task,
        decomposer=StaticTaskDecomposer((SubtaskTemplate(
            objective="Work", required_capabilities=("alpha", "beta"),
        ),)),
        specialist_match_all=True,
    )
    assert result.manager_delegations[0].assignments[0].specialist_ids == (
        complete.id,
    )


def test_multiple_subtasks_report_ready_and_no_specialist_explicitly() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    specialist = SpecialistAgent(name="Worker", capabilities=("available",))
    registry.register(manager)
    _register_owned(registry, manager, specialist)
    task = Task(objective="Parent")
    engine = _build_engine(registry)
    result = engine.delegate(
        engine.orchestrate(task), task,
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate(objective="First", required_capabilities=("available",)),
            SubtaskTemplate(objective="Second", required_capabilities=("missing",)),
        )),
    )
    manager_result = result.manager_delegations[0]
    assert tuple(item.status for item in manager_result.assignments) == (
        DelegationStatus.READY, DelegationStatus.NO_SPECIALIST,
    )
    assert manager_result.status is DelegationStatus.PARTIAL
    assert result.status is DelegationStatus.PARTIAL


def test_multiple_managers_receive_separate_linked_subtasks() -> None:
    registry = CapabilityRegistry()
    first_manager = ManagerAgent(name="First", capabilities=("manage",))
    second_manager = ManagerAgent(name="Second", capabilities=("manage",))
    first_worker = SpecialistAgent(name="First worker", capabilities=("work",))
    second_worker = SpecialistAgent(name="Second worker", capabilities=("work",))
    registry.register(first_manager)
    registry.register(second_manager)
    _register_owned(registry, first_manager, first_worker)
    _register_owned(registry, second_manager, second_worker)
    task = Task(objective="Parent")
    engine = _build_engine(registry)
    result = engine.delegate(
        engine.orchestrate(task), task,
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate(objective="Work", required_capabilities=("work",)),
        )),
    )
    assert tuple(item.manager_id for item in result.manager_delegations) == (
        first_manager.id, second_manager.id,
    )
    assert result.manager_delegations[0].assignments[0].specialist_ids == (
        first_worker.id,
    )
    assert result.manager_delegations[1].assignments[0].specialist_ids == (
        second_worker.id,
    )
    assert all(
        delegation.subtasks[0].manager_id == delegation.manager_id
        for delegation in result.manager_delegations
    )


def test_no_subtasks_and_no_manager_have_structured_statuses() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    registry.register(manager)
    task = Task(objective="Parent")
    engine = _build_engine(registry)
    no_subtasks = engine.delegate(
        engine.orchestrate(task), task, decomposer=StaticTaskDecomposer(()),
    )
    assert no_subtasks.status is DelegationStatus.NO_SUBTASKS
    assert no_subtasks.manager_delegations[0].status is DelegationStatus.NO_SUBTASKS

    other_task = Task(objective="Other")
    empty_engine = _build_engine(CapabilityRegistry())
    planning = empty_engine.orchestrate(other_task)
    assert planning.status is OrchestrationStatus.NO_MANAGER
    no_manager = empty_engine.delegate(
        planning, other_task, decomposer=StaticTaskDecomposer(()),
    )
    assert no_manager.status is DelegationStatus.NO_MANAGER
    assert no_manager.manager_delegations == ()


def test_delegation_does_not_execute_specialists() -> None:
    class TrackingSpecialist(SpecialistAgent):
        def execute(self) -> None:
            raise AssertionError("Specialist execution is outside Step 8")

    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    specialist = TrackingSpecialist(name="Worker", capabilities=("work",))
    registry.register(manager)
    _register_owned(registry, manager, specialist)
    task = Task(objective="Parent")
    engine = _build_engine(registry)
    result = engine.delegate(
        engine.orchestrate(task), task,
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate(objective="Work", required_capabilities=("work",)),
        )),
    )
    assert result.status is DelegationStatus.READY


def test_delegation_does_not_mutate_inputs_or_membership() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    specialist = SpecialistAgent(name="Worker", capabilities=("work",))
    registry.register(manager)
    _register_owned(registry, manager, specialist)
    task = Task(
        objective="Parent", status=TaskStatus.PENDING,
        metadata={"nested": {"value": 1}},
    )
    engine = _build_engine(registry)
    planning = engine.orchestrate(task)
    task_before = (
        task.id, task.objective, task.status, task.metadata.copy(), task.created_at,
    )
    registry_before = (
        registry.agents, registry.capabilities, registry.capability_index,
    )
    membership_before = manager.specialists
    triage_before = planning.triage_result
    engine.delegate(
        planning, task,
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate(objective="Work", required_capabilities=("work",)),
        )),
    )
    assert (
        task.id, task.objective, task.status, task.metadata, task.created_at,
    ) == task_before
    assert (
        registry.agents, registry.capabilities, registry.capability_index,
    ) == registry_before
    assert manager.specialists == membership_before
    assert planning.triage_result is triage_before


def test_delegation_trace_order_and_no_specialist_event() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    specialist = SpecialistAgent(name="Worker", capabilities=("available",))
    registry.register(manager)
    _register_owned(registry, manager, specialist)
    task = Task(objective="Parent")
    engine = _build_engine(registry)
    result = engine.delegate(
        engine.orchestrate(task), task,
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate(objective="Ready", required_capabilities=("available",)),
            SubtaskTemplate(objective="Missing", required_capabilities=("missing",)),
        )),
    )
    delegation_events = tuple(
        event.event_type for event in result.trace.events[4:]
    )
    assert delegation_events == (
        OrchestrationEventType.MANAGER_DELEGATION_STARTED.value,
        OrchestrationEventType.SUBTASK_CREATED.value,
        OrchestrationEventType.SPECIALIST_DISCOVERY_COMPLETED.value,
        OrchestrationEventType.SUBTASK_CREATED.value,
        OrchestrationEventType.SPECIALIST_DISCOVERY_COMPLETED.value,
        OrchestrationEventType.NO_SPECIALIST.value,
        OrchestrationEventType.MANAGER_DELEGATION_COMPLETED.value,
    )
    assert result.trace.events[-2].metadata["subtask_id"] == (
        result.manager_delegations[0].subtasks[1].id
    )


def test_delegate_accepts_step_7_plan_and_preserves_planning_result() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    registry.register(manager)
    task = Task(objective="Parent")
    engine = _build_engine(registry)
    planning = engine.orchestrate(task)
    selected_ids = planning.selected_manager_ids
    result = engine.delegate(
        planning, task, decomposer=StaticTaskDecomposer(()),
    )
    assert planning.selected_manager_ids == selected_ids
    assert result.trace is planning.trace


def test_delegate_deduplicates_manager_ids_in_manual_plan() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    registry.register(manager)
    task = Task(objective="Parent")
    engine = _build_engine(registry)
    planning = engine.orchestrate(task)
    manual_plan = replace(
        planning, selected_manager_ids=(manager.id, manager.id),
    )
    result = engine.delegate(
        manual_plan, task, decomposer=StaticTaskDecomposer(()),
    )
    assert tuple(item.manager_id for item in result.manager_delegations) == (
        manager.id,
    )
