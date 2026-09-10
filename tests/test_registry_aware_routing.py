"""Regression tests for registry-constrained provider routing."""

import importlib.util
import json

import pytest

from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent
from agenttree.core import (
    ProviderTaskDecomposer,
    ProviderTaskTriage,
    RuleBasedTaskTriage,
    StaticFinalReviewer,
    StaticManagerReviewer,
    StaticTaskDecomposer,
)
from agenttree.exceptions import DecisionOutputError
from agenttree.models import SubtaskTemplate, Task
from agenttree.orchestration import (
    DelegationStatus,
    FinalStatus,
    OrchestrationEngine,
    OrchestrationEventType,
    OrchestrationStatus,
)
from agenttree.orchestration.backends import LangGraphOrchestrationBackend
from agenttree.providers import MockProvider, ProviderConfig
from agenttree.registry import CapabilityRegistry


def provider(data: dict[str, object], name: str) -> MockProvider:
    """Return one named offline JSON provider."""
    return MockProvider(
        ProviderConfig(provider_name=name),
        response_content=json.dumps(data),
    )


def register_pair(
    registry: CapabilityRegistry,
    manager: ManagerAgent,
    *specialists: SpecialistAgent,
) -> None:
    """Register one Manager and its owned Specialists for an engine test."""
    registry.register(manager)
    for specialist in specialists:
        manager.register_specialist(specialist)
        registry.register(specialist)


def test_provider_triage_uses_registered_manager_capabilities() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(
        name="Manager", capabilities=(" Incident_Analysis ",),
    )
    registry.register(manager)
    decision = provider({
        "objective": "Analyze incident",
        "required_capabilities": [" INCIDENT_ANALYSIS "],
    }, "triage")

    plan = OrchestrationEngine(
        triage=ProviderTaskTriage(decision), registry=registry,
    ).orchestrate(Task(objective="Analyze incident"))

    assert plan.status is OrchestrationStatus.READY
    assert plan.selected_manager_ids == (manager.id,)
    assert plan.required_capabilities == ("incident_analysis",)
    request = decision.requests[0]
    assert request.context["available_manager_capabilities"] == (
        "incident_analysis",
    )
    assert '["incident_analysis"]' in request.system_prompt
    triage_event = next(
        event for event in plan.trace.events
        if event.event_type == OrchestrationEventType.TRIAGE_COMPLETED.value
    )
    assert triage_event.metadata == {
        "available_manager_capabilities": ("incident_analysis",),
        "required_capabilities": ("incident_analysis",),
    }


def test_provider_triage_rejects_unknown_registered_capability() -> None:
    registry = CapabilityRegistry()
    registry.register(ManagerAgent(
        name="Manager", capabilities=("incident_analysis",),
    ))
    decision = provider({
        "objective": "Analyze incident",
        "required_capabilities": ["invented_capability"],
    }, "triage")

    with pytest.raises(
        DecisionOutputError,
        match=r"unknown capability.*registered Manager routing.*invented_capability",
    ):
        OrchestrationEngine(
            triage=ProviderTaskTriage(decision), registry=registry,
        ).orchestrate(Task(objective="Analyze incident"))


def test_multiple_manager_choices_are_canonical_and_deterministic() -> None:
    registry = CapabilityRegistry()
    first = ManagerAgent(
        name="First", capabilities=(" Quality_Review ", "Shared"),
    )
    second = ManagerAgent(
        name="Second", capabilities=("SUMMARIZATION", " shared "),
    )
    registry.register(first)
    registry.register(second)
    decision = provider({
        "objective": "Summarize",
        "required_capabilities": [" SUMMARIZATION ", "summarization"],
    }, "triage")

    plan = OrchestrationEngine(
        triage=ProviderTaskTriage(decision), registry=registry,
    ).orchestrate(Task(objective="Summarize"))

    assert decision.requests[0].context["available_manager_capabilities"] == (
        "quality_review", "shared", "summarization",
    )
    assert plan.required_capabilities == ("summarization",)
    assert plan.selected_manager_ids == (second.id,)


def test_external_consumer_provider_flow_routes_to_owned_specialists() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(
        name="Incident Manager", capabilities=("incident_analysis",),
    )
    network = SpecialistAgent(
        name="Network Specialist", capabilities=("network_analysis",),
    )
    logs = SpecialistAgent(
        name="Log Specialist", capabilities=("log_analysis",),
    )
    register_pair(registry, manager, network, logs)
    triage_provider = provider({
        "objective": "Analyze the incident",
        "required_capabilities": ["incident_analysis"],
    }, "triage")
    decomposition_provider = provider({"subtasks": [
        {
            "objective": "Analyze network latency",
            "required_capabilities": ["network_analysis"],
        },
        {
            "objective": "Analyze suspicious log events",
            "required_capabilities": ["log_analysis"],
        },
    ]}, "decomposition")
    task = Task(
        objective=(
            "Analyze this incident involving network latency and suspicious "
            "log events."
        ),
    )
    engine = OrchestrationEngine(
        triage=ProviderTaskTriage(triage_provider), registry=registry,
    )

    delegation = engine.delegate(
        engine.orchestrate(task),
        task,
        decomposer=ProviderTaskDecomposer(decomposition_provider),
    )

    assert delegation.status is DelegationStatus.READY
    assignments = delegation.manager_delegations[0].assignments
    assert tuple(item.specialist_ids for item in assignments) == (
        (network.id,), (logs.id,),
    )
    assert decomposition_provider.requests[0].context[
        "available_specialist_capabilities"
    ] == ("network_analysis", "log_analysis")
    created = tuple(
        event for event in delegation.trace.events
        if event.event_type == OrchestrationEventType.SUBTASK_CREATED.value
    )
    assert tuple(event.metadata["required_capabilities"] for event in created) == (
        ("network_analysis",), ("log_analysis",),
    )
    assert all(
        event.metadata["available_specialist_capabilities"]
        == ("network_analysis", "log_analysis")
        for event in created
    )


def test_provider_decomposer_rejects_unknown_specialist_capability() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    specialist = SpecialistAgent(name="Specialist", capabilities=("work",))
    register_pair(registry, manager, specialist)
    task = Task(objective="Complete work")
    engine = OrchestrationEngine(
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        registry=registry,
    )
    decision = provider({"subtasks": [{
        "objective": "Work",
        "required_capabilities": ["invented_capability"],
    }]}, "decomposition")

    with pytest.raises(
        DecisionOutputError,
        match=r"unknown capability.*Specialist routing owned by Manager",
    ):
        engine.delegate(
            engine.orchestrate(task),
            task,
            decomposer=ProviderTaskDecomposer(decision),
        )


def test_other_managers_specialist_capabilities_are_not_available() -> None:
    registry = CapabilityRegistry()
    selected = ManagerAgent(name="Selected", capabilities=("selected_manager",))
    selected_worker = SpecialistAgent(
        name="Selected worker", capabilities=("selected_work",),
    )
    other = ManagerAgent(name="Other", capabilities=("other_manager",))
    other_worker = SpecialistAgent(
        name="Other worker", capabilities=("other_work",),
    )
    register_pair(registry, selected, selected_worker)
    register_pair(registry, other, other_worker)
    task = Task(objective="Selected work")
    engine = OrchestrationEngine(
        triage=RuleBasedTaskTriage(
            {}, fallback_capabilities=("selected_manager",),
        ),
        registry=registry,
    )
    decision = provider({"subtasks": [{
        "objective": "Wrongly routed work",
        "required_capabilities": ["other_work"],
    }]}, "decomposition")

    with pytest.raises(DecisionOutputError, match="other_work"):
        engine.delegate(
            engine.orchestrate(task),
            task,
            decomposer=ProviderTaskDecomposer(decision),
        )

    request = decision.requests[0]
    assert request.context["available_specialist_capabilities"] == (
        "selected_work",
    )
    assert "other_work" not in request.system_prompt


def test_empty_available_capabilities_are_explicit_and_constrained() -> None:
    decision = provider({
        "objective": "Unroutable work", "required_capabilities": [],
    }, "triage")
    plan = OrchestrationEngine(
        triage=ProviderTaskTriage(decision), registry=CapabilityRegistry(),
    ).orchestrate(Task(objective="Unroutable work"))

    assert plan.status is OrchestrationStatus.NO_MANAGER
    assert decision.requests[0].context["available_manager_capabilities"] == ()
    assert "when the array is empty" in decision.requests[0].system_prompt


def test_rule_triage_and_static_decomposer_keep_existing_behavior() -> None:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("MANAGE",))
    specialist = SpecialistAgent(name="Specialist", capabilities=("WORK",))
    register_pair(registry, manager, specialist)
    task = Task(objective="Complete work")
    engine = OrchestrationEngine(
        triage=RuleBasedTaskTriage({}, fallback_capabilities=(" manage ",)),
        registry=registry,
    )

    plan = engine.orchestrate(task)
    delegation = engine.delegate(
        plan,
        task,
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate("Work", (" work ",)),
        )),
    )

    assert plan.required_capabilities == ("manage",)
    assert plan.selected_manager_ids == (manager.id,)
    assert delegation.manager_delegations[0].assignments[0].specialist_ids == (
        specialist.id,
    )


@pytest.mark.parametrize("backend_name", ["sequential", "langgraph"])
def test_public_framework_routes_provider_decisions_on_both_backends(
    backend_name: str,
) -> None:
    if backend_name == "langgraph" and importlib.util.find_spec("langgraph") is None:
        pytest.skip("Optional LangGraph is not installed")
    triage_provider = provider({
        "objective": "Analyze incident",
        "required_capabilities": ["INCIDENT_ANALYSIS"],
    }, f"triage-{backend_name}")
    decomposition_provider = provider({"subtasks": [
        {
            "objective": "Analyze network",
            "required_capabilities": ["NETWORK_ANALYSIS"],
        },
        {
            "objective": "Analyze logs",
            "required_capabilities": ["LOG_ANALYSIS"],
        },
    ]}, f"decomposition-{backend_name}")
    options = {}
    if backend_name == "langgraph":
        options["orchestration_backend"] = LangGraphOrchestrationBackend()
    framework = AgentTree(
        root_agent=RootAgent(name="Root", id=f"root-{backend_name}"),
        triage=ProviderTaskTriage(triage_provider),
        decomposer=ProviderTaskDecomposer(decomposition_provider),
        manager_reviewer=StaticManagerReviewer(),
        final_reviewer=StaticFinalReviewer(),
        **options,
    )
    manager = ManagerAgent(
        name="Manager", id=f"manager-{backend_name}",
        capabilities=("incident_analysis",),
    )
    network = SpecialistAgent(
        name="Network", id=f"network-{backend_name}",
        capabilities=("network_analysis",),
    )
    logs = SpecialistAgent(
        name="Logs", id=f"logs-{backend_name}",
        capabilities=("log_analysis",),
    )
    framework.register_manager(manager)
    framework.register_specialist(manager, network)
    framework.register_specialist(manager, logs)
    for specialist, label in ((network, "network"), (logs, "logs")):
        execution_provider = MockProvider(
            ProviderConfig(provider_name=f"{label}-{backend_name}"),
            response_content=f"{label} result",
        )
        framework.register_provider(execution_provider)
        framework.bind_provider(specialist, execution_provider)

    result = framework.run(Task(
        id=f"task-{backend_name}",
        objective=(
            "Analyze this incident involving network latency and suspicious "
            "log events."
        ),
    ))

    assert result.status is FinalStatus.COMPLETED
    assert result.success
    assert triage_provider.requests[0].context[
        "available_manager_capabilities"
    ] == ("incident_analysis",)
    assert decomposition_provider.requests[0].context[
        "available_specialist_capabilities"
    ] == ("network_analysis", "log_analysis")
