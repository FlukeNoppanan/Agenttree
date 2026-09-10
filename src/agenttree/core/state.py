"""Validated derivation of immutable workflow-state snapshots."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, Mapping

from agenttree.models.execution import ExecutionTrace
from agenttree.models.state import WorkflowPhase, WorkflowState
from agenttree.models.task import Task, TaskStatus
from agenttree.models.triage import TriageResult


class WorkflowStateManager:
    """Create workflow snapshots and enforce the current linear phase order.

    Normal progress follows RECEIVED through COMPLETED one phase at a time.
    Any nonterminal phase may transition to FAILED. COMPLETED and FAILED reject
    further transitions. Existing engine phase results can be attached through
    ``transition`` without changing the engine APIs that produced them.
    """

    _NEXT_PHASE: dict[WorkflowPhase, WorkflowPhase] = {
        WorkflowPhase.RECEIVED: WorkflowPhase.TRIAGE,
        WorkflowPhase.TRIAGE: WorkflowPhase.PLANNING,
        WorkflowPhase.PLANNING: WorkflowPhase.DELEGATION,
        WorkflowPhase.DELEGATION: WorkflowPhase.EXECUTION,
        WorkflowPhase.EXECUTION: WorkflowPhase.MANAGER_REVIEW,
        WorkflowPhase.MANAGER_REVIEW: WorkflowPhase.FINAL_REVIEW,
        WorkflowPhase.FINAL_REVIEW: WorkflowPhase.COMPLETED,
    }

    _RESULT_FIELD: dict[WorkflowPhase, str] = {
        WorkflowPhase.TRIAGE: "triage_result",
        WorkflowPhase.PLANNING: "orchestration_plan",
        WorkflowPhase.DELEGATION: "delegation_result",
        WorkflowPhase.EXECUTION: "execution_result",
        WorkflowPhase.MANAGER_REVIEW: "manager_review_result",
        WorkflowPhase.FINAL_REVIEW: "final_result",
        WorkflowPhase.COMPLETED: "final_result",
    }

    def create_initial(
        self,
        task: Task,
        *,
        trace: ExecutionTrace | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkflowState:
        """Create a RECEIVED snapshot for ``task`` without mutating it."""
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        selected_trace = trace or ExecutionTrace(task_id=task.id)
        return WorkflowState(
            task_id=task.id,
            current_phase=WorkflowPhase.RECEIVED,
            status=task.status,
            trace=selected_trace,
            metadata=dict(metadata or {}),
        )

    def transition(
        self,
        state: WorkflowState,
        phase: WorkflowPhase,
        *,
        status: str | Enum | None = None,
        result: object | None = None,
        trace: ExecutionTrace | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkflowState:
        """Derive the next valid phase snapshot and optionally attach its result."""
        self._validate_state_and_phase(state, phase)
        if phase is WorkflowPhase.FAILED:
            return self.fail(
                state,
                reason="Workflow failed",
                status=status or TaskStatus.FAILED,
                trace=trace,
                metadata=metadata,
            )

        expected = self._NEXT_PHASE.get(state.current_phase)
        if phase is not expected:
            raise ValueError(
                f"Invalid workflow transition: {state.current_phase.value} -> "
                f"{phase.value}",
            )
        values = self._result_values(state)
        if result is not None:
            result_field = self._RESULT_FIELD.get(phase)
            if result_field is None:
                raise ValueError(f"Phase {phase.value} does not accept a result")
            self._validate_phase_result(phase, result)
            values[result_field] = result

        selected_trace = trace or getattr(result, "trace", None) or state.trace
        selected_status = self._select_status(
            phase=phase,
            result=result,
            status=status,
            final_result=values["final_result"],
        )
        return WorkflowState(
            task_id=state.task_id,
            current_phase=phase,
            status=selected_status,
            trace=selected_trace,
            metadata=self._merged_metadata(state, metadata),
            **values,
        )

    def fail(
        self,
        state: WorkflowState,
        *,
        reason: str,
        status: str | Enum = TaskStatus.FAILED,
        trace: ExecutionTrace | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkflowState:
        """Derive a terminal FAILED snapshot preserving results and failure phase."""
        if not isinstance(state, WorkflowState):
            raise TypeError("state must be a WorkflowState")
        if state.current_phase in (WorkflowPhase.COMPLETED, WorkflowPhase.FAILED):
            raise ValueError("Terminal workflow states cannot transition")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        failure_metadata = self._merged_metadata(state, metadata)
        failure_metadata["failure"] = {
            "phase": state.current_phase.value,
            "reason": reason,
        }
        return WorkflowState(
            task_id=state.task_id,
            current_phase=WorkflowPhase.FAILED,
            status=status,
            trace=trace or state.trace,
            failed_phase=state.current_phase,
            failure_reason=reason,
            metadata=failure_metadata,
            **self._result_values(state),
        )

    @classmethod
    def _validate_state_and_phase(
        cls,
        state: object,
        phase: object,
    ) -> None:
        if not isinstance(state, WorkflowState):
            raise TypeError("state must be a WorkflowState")
        if not isinstance(phase, WorkflowPhase):
            raise TypeError("phase must be a WorkflowPhase")
        if state.current_phase in (WorkflowPhase.COMPLETED, WorkflowPhase.FAILED):
            raise ValueError("Terminal workflow states cannot transition")

    @staticmethod
    def _validate_phase_result(phase: WorkflowPhase, result: object) -> None:
        from agenttree.orchestration.models import (
            DelegationPlan,
            FinalResult,
            OrchestrationPlan,
            TaskExecutionResult,
            TaskManagerReviewResult,
        )

        expected_types: dict[WorkflowPhase, type[object]] = {
            WorkflowPhase.TRIAGE: TriageResult,
            WorkflowPhase.PLANNING: OrchestrationPlan,
            WorkflowPhase.DELEGATION: DelegationPlan,
            WorkflowPhase.EXECUTION: TaskExecutionResult,
            WorkflowPhase.MANAGER_REVIEW: TaskManagerReviewResult,
            WorkflowPhase.FINAL_REVIEW: FinalResult,
            WorkflowPhase.COMPLETED: FinalResult,
        }
        expected = expected_types[phase]
        if not isinstance(result, expected):
            raise TypeError(
                f"result for {phase.value} must be a {expected.__name__}",
            )

    @staticmethod
    def _select_status(
        *,
        phase: WorkflowPhase,
        result: object | None,
        status: str | Enum | None,
        final_result: object | None,
    ) -> str | Enum | None:
        if status is not None:
            return status
        result_status = getattr(result, "status", None)
        if result_status is not None:
            return result_status
        if phase is WorkflowPhase.COMPLETED and final_result is not None:
            return getattr(final_result, "status", TaskStatus.COMPLETED)
        return None

    @staticmethod
    def _result_values(state: WorkflowState) -> dict[str, object | None]:
        return {
            "triage_result": state.triage_result,
            "orchestration_plan": state.orchestration_plan,
            "delegation_result": state.delegation_result,
            "execution_result": state.execution_result,
            "manager_review_result": state.manager_review_result,
            "final_result": state.final_result,
        }

    @staticmethod
    def _merged_metadata(
        state: WorkflowState,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        current = WorkflowStateManager._thaw_mapping(state.metadata)
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata must be a mapping")
            current.update(deepcopy(dict(metadata)))
        return current

    @staticmethod
    def _thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
        def thaw(item: Any) -> Any:
            if isinstance(item, Mapping):
                return {key: thaw(nested) for key, nested in item.items()}
            if isinstance(item, tuple):
                return [thaw(nested) for nested in item]
            if isinstance(item, frozenset):
                return {thaw(nested) for nested in item}
            return deepcopy(item)

        return {key: thaw(item) for key, item in value.items()}
