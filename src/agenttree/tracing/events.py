"""Central stable labels for framework-emitted execution events."""

from enum import Enum


class ExecutionEventType(str, Enum):
    """Known AgentTree event types; custom string event types remain valid."""

    STARTED = "orchestration.started"
    TRIAGE_COMPLETED = "orchestration.triage_completed"
    MANAGER_DISCOVERY_COMPLETED = "orchestration.manager_discovery_completed"
    NO_MANAGER = "orchestration.no_manager"
    PLANNING_COMPLETED = "orchestration.planning_completed"
    MANAGER_DELEGATION_STARTED = "orchestration.manager_delegation_started"
    SUBTASK_CREATED = "orchestration.subtask_created"
    SPECIALIST_DISCOVERY_COMPLETED = "orchestration.specialist_discovery_completed"
    NO_SPECIALIST = "orchestration.no_specialist"
    MANAGER_DELEGATION_COMPLETED = "orchestration.manager_delegation_completed"
    EXECUTION_STARTED = "orchestration.execution_started"
    SPECIALIST_EXECUTION_STARTED = "orchestration.specialist_execution_started"
    SPECIALIST_EXECUTION_COMPLETED = "orchestration.specialist_execution_completed"
    SPECIALIST_EXECUTION_FAILED = "orchestration.specialist_execution_failed"
    ASSIGNMENT_SKIPPED = "orchestration.assignment_skipped"
    EXECUTION_COMPLETED = "orchestration.execution_completed"
    MANAGER_REVIEW_STARTED = "orchestration.manager_review_started"
    MANAGER_REVIEW_PASSED = "orchestration.manager_review_passed"
    MANAGER_REVIEW_REVISION_REQUESTED = (
        "orchestration.manager_review_revision_requested"
    )
    REVISION_STARTED = "orchestration.revision_started"
    REVISION_EXECUTION_COMPLETED = "orchestration.revision_execution_completed"
    REVISION_LIMIT_REACHED = "orchestration.revision_limit_reached"
    MANAGER_REVIEW_FAILED = "orchestration.manager_review_failed"
    MANAGER_REVIEW_COMPLETED = "orchestration.manager_review_completed"
    FINAL_REVIEW_STARTED = "orchestration.final_review_started"
    FINAL_REVIEW_PASSED = "orchestration.final_review_passed"
    FINAL_REVIEW_REVISION_REQUESTED = (
        "orchestration.final_review_revision_requested"
    )
    FINAL_REVISION_STARTED = "orchestration.final_revision_started"
    FINAL_REVISION_COMPLETED = "orchestration.final_revision_completed"
    FINAL_REVISION_LIMIT_REACHED = "orchestration.final_revision_limit_reached"
    FINAL_REVIEW_FAILED = "orchestration.final_review_failed"
    FINAL_RESULT_CREATED = "orchestration.final_result_created"
    TOOL_EXECUTION_STARTED = "orchestration.tool_execution_started"
    TOOL_EXECUTION_COMPLETED = "orchestration.tool_execution_completed"
    TOOL_EXECUTION_FAILED = "orchestration.tool_execution_failed"
