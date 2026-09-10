"""Integration tests for the planning-only orchestration engine."""

from dataclasses import fields

import pytest

from agenttree.agents import ManagerAgent, RootAgent, SpecialistAgent
from agenttree.core import BaseTaskTriage, RuleBasedTaskTriage
from agenttree.models import Task, TaskStatus, TriageResult
from agenttree.orchestration import (
    OrchestrationEngine,
    OrchestrationEventType,
    OrchestrationPlan,
    OrchestrationStatus,
)
from agenttree.registry import CapabilityRegistry


class StaticTriage(BaseTaskTriage):
    """Return fixed capabilities while retaining input task identity."""

    def __init__(self, capabilities: tuple[str, ...]) -> None:
        self.capabilities = capabilities
        self.tasks: list[Task] = []

    def triage(self, task: Task) -> TriageResult:
        self.tasks.append(task)
        return TriageResult(
            task_id=task.id,
            objective=task.objective.strip(),
            required_capabilities=self.capabilities,
            metadata={"source": "static"},
        )


def test_engine_construction_uses_injected_dependencies() -> None:
    triage = StaticTriage(("prepare",))
    registry = CapabilityRegistry()
    engine = OrchestrationEngine(triage=triage, registry=registry)
    assert engine.triage is triage
    assert engine.registry is registry
    assert engine.manager_match_all is False


@pytest.mark.parametrize(
    ("triage", "registry", "match_all", "message"),
    [
        (object(), CapabilityRegistry(), False, "BaseTaskTriage"),
        (StaticTriage(()), object(), False, "CapabilityRegistry"),
        (StaticTriage(()), CapabilityRegistry(), 1, "bool"),
    ],
)
def test_engine_rejects_invalid_dependencies(
    triage: object, registry: object, match_all: object, message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        OrchestrationEngine(  # type: ignore[arg-type]
            triage=triage, registry=registry, manager_match_all=match_all,
        )


def test_successful_triage_and_manager_discovery() -> None:
    triage = StaticTriage(("prepare",))
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("prepare",))
    registry.register(manager)
    task = Task(objective="  Prepare an output  ")
    plan = OrchestrationEngine(triage=triage, registry=registry).orchestrate(task)
    assert triage.tasks == [task]
    assert plan == OrchestrationPlan(
        task_id=task.id,
        objective="Prepare an output",
        required_capabilities=("prepare",),
        selected_manager_ids=(manager.id,),
        status=OrchestrationStatus.READY,
        trace=plan.trace,
    )


def test_discovery_filters_out_root_and_specialist_agents() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("shared",))
    for agent in (
        RootAgent(name="Root", capabilities=("shared",)),
        SpecialistAgent(name="Specialist", capabilities=("shared",)),
        manager,
    ):
        registry.register(agent)
    plan = OrchestrationEngine(
        triage=StaticTriage(("shared",)), registry=registry,
    ).orchestrate(Task(objective="Objective"))
    assert plan.selected_manager_ids == (manager.id,)


def test_any_matching_is_default_and_preserves_registration_order() -> None:
    registry = CapabilityRegistry()
    second_match = ManagerAgent(name="Registered first", capabilities=("beta",))
    first_match = ManagerAgent(name="Registered second", capabilities=("alpha",))
    both = ManagerAgent(name="Registered third", capabilities=("alpha", "beta"))
    for manager in (second_match, first_match, both):
        registry.register(manager)
    plan = OrchestrationEngine(
        triage=StaticTriage(("alpha", "beta", "ALPHA")), registry=registry,
    ).orchestrate(Task(objective="Objective"))
    assert plan.selected_manager_ids == (
        second_match.id, first_match.id, both.id,
    )


def test_all_matching_can_be_configured() -> None:
    registry = CapabilityRegistry()
    partial = ManagerAgent(name="Partial", capabilities=("alpha",))
    complete = ManagerAgent(name="Complete", capabilities=("alpha", "beta"))
    registry.register(partial)
    registry.register(complete)
    engine = OrchestrationEngine(
        triage=StaticTriage(("alpha", "beta")), registry=registry,
        manager_match_all=True,
    )
    assert engine.orchestrate(Task(objective="Objective")).selected_manager_ids == (
        complete.id,
    )


@pytest.mark.parametrize("capabilities", [("missing",), ()])
def test_no_manager_returns_structured_status(
    capabilities: tuple[str, ...],
) -> None:
    registry = CapabilityRegistry()
    registry.register(ManagerAgent(name="Other", capabilities=("other",)))
    plan = OrchestrationEngine(
        triage=StaticTriage(capabilities), registry=registry,
    ).orchestrate(Task(objective="Objective"))
    assert plan.status is OrchestrationStatus.NO_MANAGER
    assert plan.selected_manager_ids == ()
    assert plan.required_capabilities == capabilities


def test_plan_metadata_defaults_are_isolated() -> None:
    task = Task(objective="Objective")
    first = OrchestrationPlan(
        task_id=task.id, objective=task.objective, required_capabilities=(),
        selected_manager_ids=(), status=OrchestrationStatus.NO_MANAGER,
        trace=OrchestrationEngine(
            triage=StaticTriage(()), registry=CapabilityRegistry(),
        ).orchestrate(task).trace,
    )
    second = OrchestrationPlan(
        task_id="other", objective="Other", required_capabilities=(),
        selected_manager_ids=(), status=OrchestrationStatus.NO_MANAGER,
        trace=OrchestrationEngine(
            triage=StaticTriage(()), registry=CapabilityRegistry(),
        ).orchestrate(Task(objective="Other", id="other")).trace,
    )
    first.metadata["changed"] = True
    assert second.metadata == {}


def test_orchestration_does_not_execute_agents() -> None:
    class TrackingManager(ManagerAgent):
        def execute(self) -> None:
            raise AssertionError("Manager execution is outside Step 7")

    class TrackingSpecialist(SpecialistAgent):
        def execute(self) -> None:
            raise AssertionError("Specialist execution is outside Step 7")

    manager = TrackingManager(name="Manager", capabilities=("shared",))
    manager.register_specialist(TrackingSpecialist(name="Worker"))
    registry = CapabilityRegistry()
    registry.register(manager)
    plan = OrchestrationEngine(
        triage=StaticTriage(("shared",)), registry=registry,
    ).orchestrate(Task(objective="Objective"))
    assert plan.selected_manager_ids == (manager.id,)


def test_orchestration_does_not_mutate_registry_or_task() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("shared",))
    registry.register(manager)
    task = Task(
        objective="  Objective  ", status=TaskStatus.PENDING,
        metadata={"nested": {"value": 1}},
    )
    registry_before = (
        registry.agents, registry.capabilities, registry.capability_index,
    )
    task_before = (
        task.id, task.objective, task.context, task.metadata.copy(),
        task.status, task.created_at,
    )
    OrchestrationEngine(
        triage=StaticTriage(("shared",)), registry=registry,
    ).orchestrate(task)
    assert (
        registry.agents, registry.capabilities, registry.capability_index,
    ) == registry_before
    assert (
        task.id, task.objective, task.context, task.metadata,
        task.status, task.created_at,
    ) == task_before


def test_success_trace_event_order_and_content() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("shared",))
    registry.register(manager)
    plan = OrchestrationEngine(
        triage=StaticTriage(("shared",)), registry=registry,
    ).orchestrate(Task(objective="Objective"))
    assert tuple(event.event_type for event in plan.trace.events) == (
        OrchestrationEventType.STARTED.value,
        OrchestrationEventType.TRIAGE_COMPLETED.value,
        OrchestrationEventType.MANAGER_DISCOVERY_COMPLETED.value,
        OrchestrationEventType.PLANNING_COMPLETED.value,
    )
    assert all(event.task_id == plan.task_id for event in plan.trace.events)
    assert plan.trace.events[2].metadata["selected_manager_ids"] == (manager.id,)
    assert plan.trace.events[-1].metadata["status"] == "ready"


def test_no_manager_trace_records_explicit_event() -> None:
    plan = OrchestrationEngine(
        triage=StaticTriage(("missing",)), registry=CapabilityRegistry(),
    ).orchestrate(Task(objective="Objective"))
    assert tuple(event.event_type for event in plan.trace.events) == (
        OrchestrationEventType.STARTED.value,
        OrchestrationEventType.TRIAGE_COMPLETED.value,
        OrchestrationEventType.MANAGER_DISCOVERY_COMPLETED.value,
        OrchestrationEventType.NO_MANAGER.value,
        OrchestrationEventType.PLANNING_COMPLETED.value,
    )


def test_engine_validates_triage_result_identity_and_type() -> None:
    class WrongIdentityTriage(BaseTaskTriage):
        def triage(self, task: Task) -> TriageResult:
            return TriageResult(task_id="other", objective=task.objective)

    class WrongTypeTriage(BaseTaskTriage):
        def triage(self, task: Task) -> TriageResult:
            return object()  # type: ignore[return-value]

    task = Task(objective="Objective")
    with pytest.raises(ValueError, match="task_id"):
        OrchestrationEngine(
            triage=WrongIdentityTriage(), registry=CapabilityRegistry(),
        ).orchestrate(task)
    with pytest.raises(TypeError, match="TriageResult"):
        OrchestrationEngine(
            triage=WrongTypeTriage(), registry=CapabilityRegistry(),
        ).orchestrate(task)


def test_plan_contains_ids_without_registry_or_agent_objects() -> None:
    field_names = {item.name for item in fields(OrchestrationPlan)}
    assert "selected_managers" not in field_names
    assert "registry" not in field_names
    assert "selected_manager_ids" in field_names


def test_rule_based_triage_registry_integration() -> None:
    registry = CapabilityRegistry()
    first = ManagerAgent(name="First", capabilities=("summarize",))
    second = ManagerAgent(name="Second", capabilities=("format",))
    registry.register(first)
    registry.register(second)
    engine = OrchestrationEngine(
        triage=RuleBasedTaskTriage({
            "summary": (" SUMMARIZE ",),
            "format": ("format",),
        }),
        registry=registry,
    )
    plan = engine.orchestrate(Task(objective="Format a summary"))
    assert plan.required_capabilities == ("SUMMARIZE", "format")
    assert plan.selected_manager_ids == (first.id, second.id)
    assert plan.status is OrchestrationStatus.READY
