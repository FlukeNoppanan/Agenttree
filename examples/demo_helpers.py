"""Presentation helpers kept outside the AgentTree production package."""

from collections.abc import Iterable, Mapping
from typing import Any

from agenttree.models import ExecutionTrace
from agenttree.orchestration import FinalResult


MAJOR_TRACE_STAGES: dict[str, str] = {
    "orchestration.started": "task received",
    "orchestration.triage_completed": "triage completed",
    "orchestration.manager_discovery_completed": "manager selected",
    "orchestration.subtask_created": "subtasks created",
    "orchestration.specialist_execution_completed": "specialists executed",
    "orchestration.manager_review_passed": "manager review passed",
    "orchestration.final_review_passed": "final review passed",
    "orchestration.final_result_created": "result created",
}


def format_trace(
    trace: ExecutionTrace,
    *,
    event_types: Iterable[str] | None = None,
) -> str:
    """Render ordered trace labels with stable one-based sequence numbers."""
    selected = None if event_types is None else set(event_types)
    events = (
        event for event in trace.events
        if selected is None or event.event_type in selected
    )
    return "\n".join(
        f"[{index:02d}] {event.event_type}"
        for index, event in enumerate(events, start=1)
    )


def format_major_trace(trace: ExecutionTrace) -> str:
    """Collapse repeated events into the eight thesis-demo workflow stages."""
    present = {event.event_type for event in trace.events}
    missing = [event_type for event_type in MAJOR_TRACE_STAGES if event_type not in present]
    if missing:
        raise ValueError(f"Trace is missing required demo events: {missing}")
    return "\n".join(
        f"{index}. {label}"
        for index, label in enumerate(MAJOR_TRACE_STAGES.values(), start=1)
    )


def format_result(
    result: FinalResult,
    *,
    specialist_names: Mapping[str, str],
) -> str:
    """Render latest accepted Specialist outputs without raw dataclass reprs."""
    lines: list[str] = []
    for manager_result in result.manager_results:
        for outcome in manager_result.subtask_outcomes:
            order: list[str] = []
            latest: dict[str, Any] = {}
            for execution in outcome.executions:
                if execution.agent_result is None:
                    continue
                specialist_id = execution.specialist_id or ""
                if specialist_id not in latest:
                    order.append(specialist_id)
                latest[specialist_id] = execution.agent_result
            for specialist_id in order:
                agent_result = latest[specialist_id]
                name = specialist_names.get(
                    specialist_id,
                    specialist_id or "Unassigned Specialist",
                )
                output: Any = agent_result.output
                lines.append(f"- {name}: {output}")
    return "\n".join(lines) if lines else "- No accepted Specialist output"
