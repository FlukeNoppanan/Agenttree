"""Structured outputs and stable labels for orchestration planning."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenttree.models import (
    AgentResult,
    ExecutionTrace,
    ReviewDecision,
    ReviewResult,
    Subtask,
    TriageResult,
    ArtifactRef,
)
from agenttree.tracing import ExecutionEventType


class OrchestrationStatus(str, Enum):
    """Outcomes produced by the current planning-only engine."""

    READY = "ready"
    NO_MANAGER = "no_manager"


# Backward-compatible public name retained from Steps 7-11.
OrchestrationEventType = ExecutionEventType


class DelegationStatus(str, Enum):
    """Assignment, manager-delegation, and aggregate delegation outcomes."""

    READY = "ready"
    PARTIAL = "partial"
    NO_MANAGER = "no_manager"
    NO_SUBTASKS = "no_subtasks"
    NO_SPECIALIST = "no_specialist"


@dataclass(frozen=True)
class OrchestrationPlan:
    """A deterministic task triage and manager-discovery result.

    Manager IDs are snapshots and do not expose registry-owned collections or
    imply that any manager has been executed.
    """

    task_id: str
    objective: str
    required_capabilities: tuple[str, ...]
    selected_manager_ids: tuple[str, ...]
    status: OrchestrationStatus
    trace: ExecutionTrace
    metadata: dict[str, Any] = field(default_factory=dict)
    triage_result: TriageResult | None = field(
        default=None, compare=False, repr=False,
    )


@dataclass(frozen=True)
class SpecialistAssignment:
    """Eligible specialist IDs selected for one unexecuted subtask."""

    subtask_id: str
    specialist_ids: tuple[str, ...]
    status: DelegationStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagerDelegation:
    """A manager's ordered subtasks and corresponding assignments."""

    manager_id: str
    subtasks: tuple[Subtask, ...]
    assignments: tuple[SpecialistAssignment, ...]
    status: DelegationStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DelegationPlan:
    """Structured manager delegation and specialist-selection output."""

    task_id: str
    manager_delegations: tuple[ManagerDelegation, ...]
    status: DelegationStatus
    trace: ExecutionTrace
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionStatus(str, Enum):
    """Specialist and aggregate outcomes for sequential execution."""

    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class SpecialistExecution:
    """One specialist outcome or one explicitly skipped assignment."""

    subtask_id: str
    specialist_id: str | None
    status: ExecutionStatus
    agent_result: AgentResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagerExecution:
    """Sequential specialist outcomes coordinated for one manager."""

    manager_id: str
    specialist_executions: tuple[SpecialistExecution, ...]
    status: ExecutionStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskExecutionResult:
    """Aggregate execution output for one delegated task."""

    task_id: str
    manager_executions: tuple[ManagerExecution, ...]
    status: ExecutionStatus
    trace: ExecutionTrace
    metadata: dict[str, Any] = field(default_factory=dict)
    delegation: DelegationPlan | None = field(
        default=None, compare=False, repr=False,
    )


class ManagerReviewStatus(str, Enum):
    """Terminal subtask and aggregate outcomes of manager review."""

    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"
    REVISION_LIMIT_REACHED = "revision_limit_reached"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class RevisionRecord:
    """One feedback-driven re-execution round for a subtask."""

    subtask_id: str
    manager_id: str
    revision_number: int
    feedback: str
    specialist_ids: tuple[str, ...]
    executions: tuple[SpecialistExecution, ...]
    status: ExecutionStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubtaskReviewOutcome:
    """Final manager decision plus all execution and revision history."""

    subtask_id: str
    manager_id: str
    decision: ReviewDecision
    feedback: str
    revision_count: int
    executions: tuple[SpecialistExecution, ...]
    reviews: tuple[ReviewResult, ...]
    revisions: tuple[RevisionRecord, ...]
    status: ManagerReviewStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagerReviewResult:
    """Ordered subtask review outcomes for one manager."""

    manager_id: str
    subtask_outcomes: tuple[SubtaskReviewOutcome, ...]
    status: ManagerReviewStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskManagerReviewResult:
    """Aggregate manager review and revision output for one task."""

    task_id: str
    manager_results: tuple[ManagerReviewResult, ...]
    status: ManagerReviewStatus
    trace: ExecutionTrace
    metadata: dict[str, Any] = field(default_factory=dict)
    execution: TaskExecutionResult | None = field(
        default=None, compare=False, repr=False,
    )


class FinalStatus(str, Enum):
    """Terminal statuses produced by Root-level finalization."""

    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    FINAL_REVISION_LIMIT_REACHED = "final_revision_limit_reached"


@dataclass(frozen=True)
class FinalRevisionRecord:
    """One final-review request followed by manager reconsideration."""

    revision_number: int
    feedback: str
    manager_review_result: TaskManagerReviewResult
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalResult:
    """Normalized task output returned after Root-level final review."""

    task_id: str
    root_agent_id: str
    success: bool
    content: dict[str, Any]
    status: FinalStatus
    manager_results: tuple[ManagerReviewResult, ...]
    final_review: ReviewResult
    final_reviews: tuple[ReviewResult, ...]
    revision_count: int
    revisions: tuple[FinalRevisionRecord, ...]
    trace: ExecutionTrace
    metadata: dict[str, Any] = field(default_factory=dict)
    final_output: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: dict[str, str] | None = None
    artifacts: tuple[ArtifactRef, ...] = ()

    @property
    def orchestration(self) -> dict[str, Any]:
        """The legacy content field, kept for structured diagnostics."""
        return self.content
