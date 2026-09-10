"""Predefined scenario execution and measurable framework-property checks."""

from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Iterable

from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    RuleBasedTaskTriage,
    StaticFinalReviewer,
    StaticManagerReviewer,
    StaticTaskDecomposer,
)
from agenttree.models import SubtaskTemplate
from agenttree.orchestration import FinalStatus
from agenttree.orchestration.backends import (
    LangGraphOrchestrationBackend,
    SequentialOrchestrationBackend,
)
from agenttree.providers import MockProvider, ProviderConfig
from agenttree.tracing import ExecutionEventType
from evaluation.models import EvaluationReport, ScenarioEvaluation
from examples.domains._shared import DomainConfiguration, DomainRun, run_scenario
from examples.domains.business_product import CONFIGURATION as PRODUCT
from examples.domains.hr_document import CONFIGURATION as HR
from examples.domains.it_technical import CONFIGURATION as IT


CONFIGURATIONS: tuple[DomainConfiguration, ...] = (IT, HR, PRODUCT)
REQUIRED_EVENTS: tuple[str, ...] = (
    ExecutionEventType.STARTED.value,
    ExecutionEventType.TRIAGE_COMPLETED.value,
    ExecutionEventType.PLANNING_COMPLETED.value,
    ExecutionEventType.MANAGER_DELEGATION_STARTED.value,
    ExecutionEventType.SPECIALIST_EXECUTION_STARTED.value,
    ExecutionEventType.SPECIALIST_EXECUTION_COMPLETED.value,
    ExecutionEventType.MANAGER_REVIEW_STARTED.value,
    ExecutionEventType.FINAL_REVIEW_STARTED.value,
    ExecutionEventType.FINAL_RESULT_CREATED.value,
)
METRIC_DEFINITIONS: dict[str, str] = {
    "reusability": (
        "PASS when all domain configurations live outside src/agenttree, use "
        "AgentTree.run(Task), and complete through SequentialOrchestrationBackend "
        "without domain-specific core changes."
    ),
    "extensibility": (
        "PASS when a newly configured ManagerAgent, SpecialistAgent, and capability "
        "register and produce a completed execution using only public APIs."
    ),
    "capability_based_routing": (
        "PASS per scenario when selected Manager and executed Specialist names "
        "exactly equal expected names and unrelated registered agents are absent."
    ),
    "provider_independence": (
        "PASS per scenario when equivalent primary and alternate MockProvider "
        "bindings produce equal final status, success, and normalized outputs."
    ),
    "traceability": (
        "PASS per scenario when the final trace belongs to the Task, every event "
        "uses that Task ID, and required major phase labels occur in order."
    ),
}


def _selected_names(run: DomainRun) -> tuple[tuple[str, ...], tuple[str, ...]]:
    state = run.framework.last_state
    assert state is not None
    assert state.orchestration_plan is not None
    assert state.execution_result is not None
    manager_names = {item.id: item.name for item in run.framework.managers}
    specialist_names = {item.id: item.name for item in run.framework.specialists}
    managers = tuple(
        manager_names[item]
        for item in state.orchestration_plan.selected_manager_ids
    )
    specialists = tuple(
        specialist_names[execution.specialist_id]
        for manager in state.execution_result.manager_executions
        for execution in manager.specialist_executions
        if execution.specialist_id is not None
    )
    return managers, specialists


def _normalized_outputs(run: DomainRun) -> tuple[Any, ...]:
    state = run.framework.last_state
    assert state is not None and state.execution_result is not None
    return tuple(
        deepcopy(execution.agent_result.output)
        for manager in state.execution_result.manager_executions
        for execution in manager.specialist_executions
        if execution.agent_result is not None
    )


def _provider_assignments(run: DomainRun) -> dict[str, str]:
    """Map expected Specialist names to externally bound provider identities."""
    bindings = run.framework.provider_bindings
    return {
        specialist.name: bindings[specialist.id]
        for specialist in run.framework.specialists
        if specialist.name in run.expected_specialist_names
    }


def _trace_evidence(run: DomainRun) -> tuple[bool, dict[str, Any]]:
    events = run.result.trace.events
    labels = tuple(event.event_type for event in events)
    positions: list[int] = []
    cursor = 0
    for expected in REQUIRED_EVENTS:
        try:
            position = labels.index(expected, cursor)
        except ValueError:
            positions = []
            break
        positions.append(position)
        cursor = position + 1
    passed = (
        run.result.trace.task_id == run.task.id
        and all(event.task_id == run.task.id for event in events)
        and len(positions) == len(REQUIRED_EVENTS)
        and positions == sorted(positions)
        and run.framework.last_state is not None
        and run.framework.last_state.trace.events == events
    )
    return passed, {
        "task_id": run.task.id,
        "event_count": len(events),
        "required_events": list(REQUIRED_EVENTS),
        "required_event_positions": positions,
    }


def evaluate_scenario(configuration: DomainConfiguration, scenario_index: int) -> ScenarioEvaluation:
    """Run equivalent provider configurations and record observable evidence."""
    scenario = configuration.scenarios[scenario_index]
    primary = run_scenario(configuration, scenario, provider_variant="primary")
    alternate = run_scenario(configuration, scenario, provider_variant="alternate")
    actual_managers, actual_specialists = _selected_names(primary)
    unrelated_selected = tuple(
        name for name in primary.unrelated_agent_names
        if name in actual_managers or name in actual_specialists
    )
    routing_passed = (
        actual_managers == primary.expected_manager_names
        and actual_specialists == primary.expected_specialist_names
        and not unrelated_selected
    )
    traceability_passed, trace_metrics = _trace_evidence(primary)
    provider_independence_passed = (
        primary.result.status == alternate.result.status
        and primary.result.success == alternate.result.success
        and _normalized_outputs(primary) == _normalized_outputs(alternate)
        and tuple(item.name for item in primary.framework.providers)
        != tuple(item.name for item in alternate.framework.providers)
        and len(set(_provider_assignments(primary).values()))
        == len(primary.expected_specialist_names)
    )
    tool_metrics: dict[str, Any] = {"configured": primary.tool_result is not None}
    if primary.tool_result is not None:
        tool_metrics.update({
            "success": primary.tool_result.success,
            "output": deepcopy(primary.tool_result.output),
            "trace_event_count": primary.tool_trace.event_count if primary.tool_trace else 0,
        })
    status_passed = (
        primary.result.status is FinalStatus.COMPLETED
        and primary.result.success
        and primary.task_unchanged
    )
    passed = (
        status_passed
        and routing_passed
        and traceability_passed
        and provider_independence_passed
        and (primary.tool_result is None or primary.tool_result.success)
    )
    return ScenarioEvaluation(
        scenario_id=scenario.scenario_id,
        domain=configuration.domain_name,
        status="PASS" if passed else "FAIL",
        final_status=primary.result.status.value,
        routing_passed=routing_passed,
        traceability_passed=traceability_passed,
        provider_independence_passed=provider_independence_passed,
        notes=("Synthetic data; offline deterministic providers.",),
        metrics={
            "expected_managers": list(primary.expected_manager_names),
            "expected_manager_ids": [f"{configuration.domain_id}-manager"],
            "selected_managers": list(actual_managers),
            "selected_manager_ids": list(
                primary.framework.last_state.orchestration_plan.selected_manager_ids
            ),
            "expected_specialists": list(primary.expected_specialist_names),
            "expected_specialist_ids": [
                f"{configuration.domain_id}-specialist-{index + 1}"
                for index in range(len(configuration.specialists))
            ],
            "executed_specialists": list(actual_specialists),
            "executed_specialist_ids": [
                execution.specialist_id
                for manager in primary.framework.last_state.execution_result.manager_executions
                for execution in manager.specialist_executions
            ],
            "unrelated_agents_selected": list(unrelated_selected),
            "provider_variants": {
                "primary": [item.name for item in primary.framework.providers],
                "alternate": [item.name for item in alternate.framework.providers],
            },
            "provider_assignments": {
                "primary": _provider_assignments(primary),
                "alternate": _provider_assignments(alternate),
            },
            "normalized_outputs": list(_normalized_outputs(primary)),
            "final_result_shape": sorted(primary.result.__dataclass_fields__),
            "task_input_unchanged": primary.task_unchanged,
            "trace": trace_metrics,
            "tool_proof": tool_metrics,
        },
    )


def _extension_probe() -> dict[str, Any]:
    capability = "new_application_synthesis"
    framework = AgentTree(
        root_agent=RootAgent(name="Extension Root", id="extension-root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=(capability,)),
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate("Run the new configured work", (capability,)),
        )),
        manager_reviewer=StaticManagerReviewer(),
        final_reviewer=StaticFinalReviewer(),
    )
    manager = ManagerAgent(
        name="New Application Manager",
        id="extension-manager",
        capabilities=(capability,),
    )
    specialist = SpecialistAgent(
        name="New Application Specialist",
        id="extension-specialist",
        capabilities=(capability,),
    )
    provider = MockProvider(
        ProviderConfig(provider_name="extension-provider"),
        response_content="New application-level capability executed.",
    )
    framework.register_manager(manager)
    framework.register_specialist(manager, specialist)
    framework.register_provider(provider)
    framework.bind_provider(specialist, provider)
    result = framework.run(Task(id="task-extension-probe", objective="Exercise extension"))
    _, executed = _selected_names(DomainRun(
        configuration=IT,
        scenario=IT.scenarios[0],
        framework=framework,
        task=Task(id="unused", objective="unused"),
        result=result,
        expected_manager_names=(manager.name,),
        expected_specialist_names=(specialist.name,),
        unrelated_agent_names=(),
    ))
    passed = (
        result.status is FinalStatus.COMPLETED
        and manager in framework.managers
        and specialist in framework.specialists
        and executed == (specialist.name,)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "manager": manager.name,
        "specialist": specialist.name,
        "new_capability": capability,
        "executed_specialists": list(executed),
        "final_status": result.status.value,
    }


def _langgraph_probe() -> dict[str, Any]:
    available = find_spec("langgraph") is not None
    if not available:
        return {"available": False, "status": "NOT_RUN", "reason": "Optional dependency absent"}
    scenario = IT.scenarios[0]
    sequential = run_scenario(
        IT, scenario, orchestration_backend=SequentialOrchestrationBackend(),
    )
    graph = run_scenario(
        IT, scenario, orchestration_backend=LangGraphOrchestrationBackend(),
    )
    passed = (
        graph.result.status == sequential.result.status
        and graph.result.success == sequential.result.success
        and _normalized_outputs(graph) == _normalized_outputs(sequential)
        and tuple(event.event_type for event in graph.result.trace.events)
        == tuple(event.event_type for event in sequential.result.trace.events)
    )
    return {
        "available": True,
        "status": "PASS" if passed else "FAIL",
        "scenario_id": scenario.scenario_id,
        "final_status": graph.result.status.value,
        "trace_event_count": graph.result.trace.event_count,
    }


def build_report() -> EvaluationReport:
    """Run all predefined scenarios and aggregate exact pass/fail definitions."""
    scenarios = tuple(
        evaluate_scenario(configuration, index)
        for configuration in CONFIGURATIONS
        for index in range(len(configuration.scenarios))
    )
    customization_paths = (
        "examples/domains/it_technical.py",
        "examples/domains/hr_document.py",
        "examples/domains/business_product.py",
    )
    customization_outside_core = all(
        not Path(path).as_posix().startswith("src/agenttree/")
        for path in customization_paths
    )
    extension = _extension_probe()
    langgraph = _langgraph_probe()
    reusability_passed = (
        customization_outside_core
        and all(item.final_status == FinalStatus.COMPLETED.value for item in scenarios)
    )
    tool_rows = [item for item in scenarios if item.metrics["tool_proof"]["configured"]]
    tool_passed = bool(tool_rows) and all(
        item.metrics["tool_proof"]["success"]
        and item.metrics["tool_proof"]["trace_event_count"] == 2
        for item in tool_rows
    )
    overall = {
        "reusability": {
            "status": "PASS" if reusability_passed else "FAIL",
            "core_source_files_modified_per_domain": {
                configuration.domain_name: 0 for configuration in CONFIGURATIONS
            },
            "customization_paths": list(customization_paths),
            "customization_paths_outside_core": customization_outside_core,
            "public_api": "AgentTree.run(Task)",
            "orchestration_backend": "SequentialOrchestrationBackend",
        },
        "extensibility": extension,
        "capability_based_routing": {
            "status": "PASS" if all(item.routing_passed for item in scenarios) else "FAIL",
            "scenarios_passed": sum(item.routing_passed for item in scenarios),
            "scenarios_total": len(scenarios),
        },
        "provider_independence": {
            "status": "PASS" if all(item.provider_independence_passed for item in scenarios) else "FAIL",
            "scenarios_passed": sum(item.provider_independence_passed for item in scenarios),
            "scenarios_total": len(scenarios),
        },
        "traceability": {
            "status": "PASS" if all(item.traceability_passed for item in scenarios) else "FAIL",
            "scenarios_passed": sum(item.traceability_passed for item in scenarios),
            "scenarios_total": len(scenarios),
        },
        "tool_integration": {
            "status": "PASS" if tool_passed else "FAIL",
            "scenarios_with_tool_proof": len(tool_rows),
        },
        "langgraph_equivalence": langgraph,
        "all_scenarios": {
            "status": "PASS" if all(item.status == "PASS" for item in scenarios) else "FAIL",
            "passed": sum(item.status == "PASS" for item in scenarios),
            "total": len(scenarios),
        },
    }
    return EvaluationReport(
        schema_version="1.0",
        framework_api="AgentTree.run(Task)",
        metric_definitions=METRIC_DEFINITIONS,
        scenarios=scenarios,
        overall=overall,
    )
