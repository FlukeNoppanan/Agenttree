"""Immutable workflow-state snapshots for incremental orchestration progress."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from agenttree.models.execution import ExecutionEvent, ExecutionTrace
from agenttree.models.triage import TriageResult

if TYPE_CHECKING:
    from agenttree.orchestration.models import (
        DelegationPlan,
        FinalResult,
        OrchestrationPlan,
        TaskExecutionResult,
        TaskManagerReviewResult,
    )


class WorkflowPhase(str, Enum):
    """Ordered stages in the synchronous AgentTree workflow."""

    RECEIVED = "received"
    TRIAGE = "triage"
    PLANNING = "planning"
    DELEGATION = "delegation"
    EXECUTION = "execution"
    MANAGER_REVIEW = "manager_review"
    FINAL_REVIEW = "final_review"
    COMPLETED = "completed"
    FAILED = "failed"


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _freeze_value(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return deepcopy(value)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        timestamp = value
        if timestamp.tzinfo is not None and timestamp.utcoffset() is not None:
            timestamp = timestamp.astimezone(timezone.utc)
        return timestamp.isoformat()
    if isinstance(value, (ExecutionEvent, ExecutionTrace)):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        prepared = [_to_serializable(item) for item in value]
        return sorted(prepared, key=repr)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_serializable(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    return deepcopy(value)


@dataclass(frozen=True)
class WorkflowState:
    """An immutable snapshot of one task's partial or complete workflow.

    Phase results are copied when the snapshot is created. The trace is cloned,
    and metadata is recursively frozen, so later states cannot alter older
    state snapshots through shared mutable containers.
    """

    task_id: str
    current_phase: WorkflowPhase
    status: str | Enum | None
    trace: ExecutionTrace
    triage_result: TriageResult | None = None
    orchestration_plan: OrchestrationPlan | None = None
    delegation_result: DelegationPlan | None = None
    execution_result: TaskExecutionResult | None = None
    manager_review_result: TaskManagerReviewResult | None = None
    final_result: FinalResult | None = None
    failed_phase: WorkflowPhase | None = None
    failure_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.current_phase, WorkflowPhase):
            raise TypeError("current_phase must be a WorkflowPhase")
        if self.status is not None and not isinstance(self.status, (str, Enum)):
            raise TypeError("status must be a string, enum, or None")
        if not isinstance(self.trace, ExecutionTrace):
            raise TypeError("trace must be an ExecutionTrace")
        if self.trace.task_id != self.task_id:
            raise ValueError("trace task_id must match WorkflowState task_id")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if self.failed_phase is not None and not isinstance(
            self.failed_phase, WorkflowPhase,
        ):
            raise TypeError("failed_phase must be a WorkflowPhase or None")
        if self.failure_reason is not None and not isinstance(
            self.failure_reason, str,
        ):
            raise TypeError("failure_reason must be a string or None")
        if self.current_phase is not WorkflowPhase.FAILED and (
            self.failed_phase is not None or self.failure_reason is not None
        ):
            raise ValueError("failure details require the FAILED phase")

        for name in (
            "triage_result",
            "orchestration_plan",
            "delegation_result",
            "execution_result",
            "manager_review_result",
            "final_result",
        ):
            result = getattr(self, name)
            if result is not None:
                if getattr(result, "task_id", None) != self.task_id:
                    raise ValueError(f"{name} must match WorkflowState task_id")
                object.__setattr__(self, name, deepcopy(result))

        object.__setattr__(self, "trace", self.trace.clone(read_only=True))
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serialization-ready state snapshot."""
        return {
            "task_id": self.task_id,
            "current_phase": self.current_phase.value,
            "status": _to_serializable(self.status),
            "triage_result": _to_serializable(self.triage_result),
            "orchestration_plan": _to_serializable(self.orchestration_plan),
            "delegation_result": _to_serializable(self.delegation_result),
            "execution_result": _to_serializable(self.execution_result),
            "manager_review_result": _to_serializable(
                self.manager_review_result,
            ),
            "final_result": _to_serializable(self.final_result),
            "failed_phase": _to_serializable(self.failed_phase),
            "failure_reason": self.failure_reason,
            "trace": self.trace.to_dict(),
            "metadata": _to_serializable(self.metadata),
        }
