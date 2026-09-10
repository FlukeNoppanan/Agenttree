"""Small shared helpers for readable, comparable domain examples."""

from dataclasses import dataclass
from copy import deepcopy
import json
from typing import Any

from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    RuleBasedTaskTriage,
    StaticFinalReviewer,
    StaticManagerReviewer,
    StaticTaskDecomposer,
)
from agenttree.models import ExecutionTrace, SubtaskTemplate, TaskContext
from agenttree.orchestration import FinalResult
from agenttree.orchestration.backends import BaseOrchestrationBackend
from agenttree.providers import MockProvider, ProviderConfig
from agenttree.tools import FunctionTool, ToolBindingRegistry, ToolExecutor, ToolRegistry
from agenttree.tools.models import ToolResult


@dataclass(frozen=True)
class DomainScenario:
    """One synthetic task submitted to an application-configured hierarchy."""

    scenario_id: str
    objective: str
    context: dict[str, Any]


@dataclass(frozen=True)
class DomainConfiguration:
    """Application-level identities, capabilities, and static work templates."""

    domain_id: str
    domain_name: str
    manager_name: str
    manager_capability: str
    specialists: tuple[tuple[str, str, str], ...]
    unrelated_specialist: tuple[str, str]
    templates: tuple[tuple[str, str], ...]
    scenarios: tuple[DomainScenario, ...]
    tool: tuple[str, Any, dict[str, Any]] | None = None


@dataclass(frozen=True)
class DomainRun:
    """Public SDK result plus evidence needed by the offline evaluation."""

    configuration: DomainConfiguration
    scenario: DomainScenario
    framework: AgentTree
    task: Task
    result: FinalResult
    expected_manager_names: tuple[str, ...]
    expected_specialist_names: tuple[str, ...]
    unrelated_agent_names: tuple[str, ...]
    task_unchanged: bool = True
    tool_result: ToolResult | None = None
    tool_trace: ExecutionTrace | None = None


def build_framework(
    configuration: DomainConfiguration,
    *,
    provider_variant: str = "primary",
    orchestration_backend: BaseOrchestrationBackend | None = None,
) -> tuple[AgentTree, ToolExecutor | None, tuple[ManagerAgent, ...]]:
    """Configure the public SDK from one domain description.

    This helper only reduces repeated registration. It does not wrap or replace
    ``AgentTree.run`` and contains no domain policy.
    """
    if provider_variant not in ("primary", "alternate"):
        raise ValueError("provider_variant must be primary or alternate")
    tool_registry = ToolRegistry()
    tool_bindings = ToolBindingRegistry()
    framework = AgentTree(
        root_agent=RootAgent(
            name=f"{configuration.domain_name} Root",
            id=f"{configuration.domain_id}-root",
        ),
        triage=RuleBasedTaskTriage(
            {}, fallback_capabilities=(configuration.manager_capability,),
        ),
        decomposer=StaticTaskDecomposer(tuple(
            SubtaskTemplate(objective, (capability,))
            for objective, capability in configuration.templates
        )),
        manager_reviewer=StaticManagerReviewer(),
        final_reviewer=StaticFinalReviewer(),
        tool_registry=tool_registry,
        tool_bindings=tool_bindings,
        orchestration_backend=orchestration_backend,
    )
    manager = ManagerAgent(
        name=configuration.manager_name,
        id=f"{configuration.domain_id}-manager",
        capabilities=(configuration.manager_capability,),
    )
    unrelated_manager = ManagerAgent(
        name=f"Unrelated {configuration.domain_name} Manager",
        id=f"{configuration.domain_id}-unrelated-manager",
        capabilities=(f"unrelated_{configuration.domain_id}_management",),
    )
    framework.register_manager(manager)
    framework.register_manager(unrelated_manager)

    expected_specialists: list[SpecialistAgent] = []
    for index, (name, capability, output) in enumerate(configuration.specialists):
        specialist = SpecialistAgent(
            name=name,
            id=f"{configuration.domain_id}-specialist-{index + 1}",
            capabilities=(capability,),
        )
        framework.register_specialist(manager, specialist)
        provider = MockProvider(
            ProviderConfig(
                provider_name=(
                    f"{configuration.domain_id}-{provider_variant}-provider-{index + 1}"
                ),
                model="offline-fixture",
            ),
            response_content=output,
        )
        framework.register_provider(provider)
        framework.bind_provider(specialist, provider)
        expected_specialists.append(specialist)

    unrelated_name, unrelated_capability = configuration.unrelated_specialist
    unrelated_specialist = SpecialistAgent(
        name=unrelated_name,
        id=f"{configuration.domain_id}-unrelated-specialist",
        capabilities=(unrelated_capability,),
    )
    framework.register_specialist(manager, unrelated_specialist)
    unrelated_manager_specialist = SpecialistAgent(
        name=f"Unrelated {configuration.domain_name} Worker",
        id=f"{configuration.domain_id}-other-manager-specialist",
        capabilities=(configuration.templates[0][1],),
    )
    framework.register_specialist(unrelated_manager, unrelated_manager_specialist)

    tool_executor: ToolExecutor | None = None
    if configuration.tool is not None:
        tool_name, function, _ = configuration.tool
        tool = FunctionTool(
            name=tool_name,
            tool_id=f"{configuration.domain_id}-tool",
            function=function,
            description="Application-configured offline helper.",
        )
        framework.register_tool(tool)
        framework.bind_tool(expected_specialists[0], tool)
        tool_executor = ToolExecutor(registry=tool_registry, bindings=tool_bindings)
    return framework, tool_executor, (manager, unrelated_manager)


def run_scenario(
    configuration: DomainConfiguration,
    scenario: DomainScenario | None = None,
    *,
    provider_variant: str = "primary",
    orchestration_backend: BaseOrchestrationBackend | None = None,
) -> DomainRun:
    """Run one synthetic scenario through the public AgentTree SDK."""
    selected_scenario = scenario or configuration.scenarios[0]
    framework, tool_executor, managers = build_framework(
        configuration,
        provider_variant=provider_variant,
        orchestration_backend=orchestration_backend,
    )
    task = Task(
        id=f"task-{selected_scenario.scenario_id}",
        objective=selected_scenario.objective,
        context=TaskContext(data=selected_scenario.context),
        metadata={"scenario_id": selected_scenario.scenario_id},
    )
    original_task = deepcopy(task)
    tool_result = None
    tool_trace = None
    if configuration.tool is not None and tool_executor is not None:
        tool_name, _, arguments = configuration.tool
        tool_trace = ExecutionTrace(task_id=task.id)
        tool_result = tool_executor.execute(
            specialist=framework.specialists[0],
            tool_name=tool_name,
            arguments=arguments,
            trace=tool_trace,
        )
    result = framework.run(task)
    return DomainRun(
        configuration=configuration,
        scenario=selected_scenario,
        framework=framework,
        task=task,
        result=result,
        expected_manager_names=(configuration.manager_name,),
        expected_specialist_names=tuple(item[0] for item in configuration.specialists),
        unrelated_agent_names=(
            managers[1].name,
            configuration.unrelated_specialist[0],
            f"Unrelated {configuration.domain_name} Worker",
        ),
        task_unchanged=task == original_task,
        tool_result=tool_result,
        tool_trace=tool_trace,
    )


def execution_summary(run: DomainRun) -> dict[str, Any]:
    """Create a concise name-based view without dumping raw framework objects."""
    state = run.framework.last_state
    assert state is not None and state.orchestration_plan is not None
    assert state.execution_result is not None
    managers = {manager.id: manager.name for manager in run.framework.managers}
    specialists = {specialist.id: specialist.name for specialist in run.framework.specialists}
    selected_managers = [
        managers[manager_id]
        for manager_id in state.orchestration_plan.selected_manager_ids
    ]
    executions: list[dict[str, Any]] = []
    for manager in state.execution_result.manager_executions:
        for item in manager.specialist_executions:
            executions.append({
                "specialist": specialists.get(item.specialist_id, item.specialist_id),
                "status": item.status.value,
                "output": item.agent_result.output if item.agent_result else None,
            })
    return {
        "domain": run.configuration.domain_name,
        "task_id": run.task.id,
        "final_status": run.result.status.value,
        "success": run.result.success,
        "selected_managers": selected_managers,
        "specialist_executions": executions,
        "final_result": {
            "manager_count": len(run.result.manager_results),
            "accepted_outputs": [item["output"] for item in executions],
        },
        "trace_event_count": run.result.trace.event_count,
    }


def print_summary(run: DomainRun) -> None:
    """Print the required concise domain/result fields as deterministic JSON."""
    print(json.dumps(execution_summary(run), indent=2, sort_keys=True))
