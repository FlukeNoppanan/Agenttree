"""Polished deterministic AgentTree thesis demonstration."""

import argparse
from collections.abc import Sequence

from agenttree import (
    AgentTree,
    AgentTreeConfig,
    ManagerAgent,
    RootAgent,
    SpecialistAgent,
    Task,
)
from agenttree.core import (
    RuleBasedTaskTriage,
    StaticFinalReviewer,
    StaticManagerReviewer,
    StaticTaskDecomposer,
)
from agenttree.models import ReviewDecision, SubtaskTemplate
from agenttree.orchestration import FinalResult
from agenttree.orchestration.backends import LangGraphOrchestrationBackend
from agenttree.providers import MockProvider, ProviderConfig

try:
    from examples.demo_helpers import format_major_trace, format_result
except ModuleNotFoundError:  # Direct ``python examples/thesis_demo.py`` execution.
    from demo_helpers import format_major_trace, format_result


OBJECTIVE = (
    "Analyze a service incident involving network latency and suspicious log "
    "events, then recommend next actions."
)


def build_demo(
    *,
    backend: str = "sequential",
    revision: bool = False,
) -> tuple[AgentTree, Task, ManagerAgent, tuple[SpecialistAgent, ...]]:
    """Build the offline demo using only public framework configuration APIs."""
    if backend not in ("sequential", "langgraph"):
        raise ValueError("backend must be sequential or langgraph")
    manager_reviewer = (
        StaticManagerReviewer(outcomes=(
            (ReviewDecision.REVISE, "Add a clearer next action and evidence link."),
            (ReviewDecision.PASS, "Revision addresses the requested evidence."),
        ))
        if revision
        else StaticManagerReviewer(
            ReviewDecision.PASS,
            "Both analyses support concrete next actions.",
        )
    )
    framework = AgentTree(
        root_agent=RootAgent(name="Incident Review Root", id="demo-root"),
        triage=RuleBasedTaskTriage({
            "incident": ("incident_analysis",),
        }),
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate(
                "Assess the synthetic latency evidence",
                ("network_analysis",),
            ),
            SubtaskTemplate(
                "Assess the synthetic suspicious log events",
                ("security_log_analysis",),
            ),
        )),
        manager_reviewer=manager_reviewer,
        final_reviewer=StaticFinalReviewer(
            ReviewDecision.PASS,
            "The reviewed recommendation satisfies the incident objective.",
        ),
        config=AgentTreeConfig(max_manager_revisions=1),
        orchestration_backend=(
            LangGraphOrchestrationBackend() if backend == "langgraph" else None
        ),
    )
    manager = ManagerAgent(
        name="IT Incident Manager",
        id="demo-it-manager",
        capabilities=("incident_analysis",),
    )
    specialists = (
        SpecialistAgent(
            name="Network Specialist",
            id="demo-network-specialist",
            capabilities=("network_analysis",),
        ),
        SpecialistAgent(
            name="Security and Log Specialist",
            id="demo-log-specialist",
            capabilities=("security_log_analysis",),
        ),
    )
    providers = (
        MockProvider(
            ProviderConfig(provider_name="offline-network", model="demo-fixture"),
            response_content=(
                "Latency is isolated to a synthetic upstream path; verify the "
                "route and continue bounded monitoring."
            ),
        ),
        MockProvider(
            ProviderConfig(provider_name="offline-security-log", model="demo-fixture"),
            response_content=(
                "The synthetic log pattern is anomalous but contained; preserve "
                "evidence and review the affected service identity."
            ),
        ),
    )
    framework.register_manager(manager)
    for specialist, provider in zip(specialists, providers):
        framework.register_specialist(manager, specialist)
        framework.register_provider(provider)
        framework.bind_provider(specialist, provider)
    task = Task(
        id="thesis-incident-001",
        objective=OBJECTIVE,
        metadata={"data_classification": "synthetic"},
    )
    return framework, task, manager, specialists


def render_demo(
    framework: AgentTree,
    task: Task,
    manager: ManagerAgent,
    specialists: tuple[SpecialistAgent, ...],
    result: FinalResult,
    *,
    backend: str,
    revision: bool,
) -> str:
    """Create concise, presentation-friendly output from public result/state APIs."""
    state = framework.last_state
    assert state is not None and state.orchestration_plan is not None
    assert state.execution_result is not None
    selected_manager_ids = state.orchestration_plan.selected_manager_ids
    selected_managers = [
        item.name for item in framework.managers if item.id in selected_manager_ids
    ]
    executed_specialist_ids = {
        execution.specialist_id
        for manager_execution in state.execution_result.manager_executions
        for execution in manager_execution.specialist_executions
        if execution.specialist_id is not None
    }
    selected_specialists = tuple(
        specialist for specialist in specialists
        if specialist.id in executed_specialist_ids
    )
    provider_bindings = framework.provider_bindings
    manager_lines: list[str] = []
    revision_lines: list[str] = []
    subtask_number = 0
    for manager_result in result.manager_results:
        for outcome in manager_result.subtask_outcomes:
            subtask_number += 1
            decisions = " -> ".join(review.decision.value.upper() for review in outcome.reviews)
            manager_lines.append(f"- {decisions}: {outcome.feedback}")
            if outcome.revisions:
                revision_record = outcome.revisions[0]
                revision_lines.extend((
                    f"- Subtask {subtask_number}: Specialist execution completed",
                    f"- Manager requested revision: {revision_record.feedback}",
                    "- Feedback returned to the Specialist context",
                    "- Specialist executed again",
                    f"- Manager decision: {outcome.decision.value.upper()}",
                ))
    names = {specialist.id: specialist.name for specialist in specialists}
    lines = [
        "AgentTree Thesis Demo",
        f"Backend: {backend}",
        f"Mode: {'manager revision' if revision else 'standard'}",
        "",
        "Task:",
        task.objective,
        "",
        "Selected Manager:",
        *(f"- {name}" for name in selected_managers),
        "",
        "Selected Specialists:",
        *(f"- {specialist.name}" for specialist in selected_specialists),
        "",
        "Provider Assignments:",
        *(
            f"- {specialist.name}: {provider_bindings[specialist.id]}"
            for specialist in selected_specialists
        ),
        "",
        "Manager Review:",
        *manager_lines,
        "",
        "Final Review:",
        f"- {result.final_review.decision.value.upper()}: {result.final_review.feedback}",
        "",
        "Final Status:",
        result.status.value.upper(),
        "",
        "Final Result:",
        format_result(result, specialist_names=names),
    ]
    if revision_lines:
        lines.extend(("", "Revision Cycle:", *revision_lines))
    lines.extend(("", "Trace Summary:", format_major_trace(result.trace)))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standard or revision demo and return nonzero on workflow failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("sequential", "langgraph"),
        default="sequential",
    )
    parser.add_argument(
        "--revision",
        action="store_true",
        help="Show Manager REVISE followed by PASS for each subtask.",
    )
    args = parser.parse_args(argv)
    framework, task, manager, specialists = build_demo(
        backend=args.backend,
        revision=args.revision,
    )
    result = framework.run(task)
    print(render_demo(
        framework,
        task,
        manager,
        specialists,
        result,
        backend=args.backend,
        revision=args.revision,
    ))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
