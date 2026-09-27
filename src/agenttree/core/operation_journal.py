"""Durable logical work boundaries within an orchestration phase."""

from __future__ import annotations

from contextvars import ContextVar
from hashlib import sha256
from typing import Any, Callable, TypeVar

from agenttree.core.execution_codec import dumps, loads
from agenttree.core.execution_control import ExecutionCancelled, check_execution
from agenttree.core.execution_store import (
    ExecutionRecoveryBlocked, ExecutionStore, OperationRecord, OperationState, utc_now,
)
from agenttree.models import ExecutionEvent

T = TypeVar("T")
RECONCILED_NOT_COMPLETED = object()


class ReconciledFailure(RuntimeError):
    """An external reconciler established that the operation failed."""
_active_operation_journal: ContextVar[OperationJournal | None] = ContextVar(
    "agenttree_operation_journal", default=None)
_active_operation_key: ContextVar[str | None] = ContextVar(
    "agenttree_operation_key", default=None)


def current_journal() -> OperationJournal | None:
    return _active_operation_journal.get()


def current_operation_idempotency_key() -> str | None:
    """Stable key available to Tool implementations during durable invocation."""
    journal = current_journal()
    key = _active_operation_key.get()
    if journal is None or key is None:
        return None
    record = journal.store.get_operation(journal.execution_id, key)
    return record.idempotency_key if record is not None else None


class OperationJournal:
    """Commit results before downstream code sees them; reuse committed results."""

    def __init__(self, store: ExecutionStore, execution_id: str) -> None:
        self.store = store
        self.execution_id = execution_id
        self.phase = "queued"
        self._counts: dict[tuple[str, str], int] = {}

    def start_phase(self, phase: str) -> None:
        self.phase = phase
        self._counts.clear()

    def next_key(self, kind: str) -> str:
        parent = _active_operation_key.get() or self.phase
        index = self._counts.get((parent, kind), 0) + 1
        self._counts[parent, kind] = index
        return f"{parent}/{kind}:{index}"

    def child_key(self, part: str) -> str:
        return f"{_active_operation_key.get() or self.phase}/{part}"

    def _event(self, record: OperationRecord, status: str) -> None:
        self.store.append_event(self.execution_id, ExecutionEvent(
            task_id=self.execution_id, actor_id=record.agent_id,
            event_type=f"operation.{status}",
            metadata={"operation_key": record.operation_key,
                      "operation_type": record.operation_type, "phase": record.phase,
                      "attempt": record.attempt, "status": status},
        ))

    def run(self, key: str, operation_type: str, input_value: Any,
            action: Callable[[], T], *, policy: str = "retry",
            agent_id: str | None = None,
            reconcile: Callable[[OperationRecord], T | None] | None = None) -> T:
        check_execution()
        digest = sha256(dumps(input_value).encode()).hexdigest()
        record = self.store.get_operation(self.execution_id, key)
        if record is None:
            now = utc_now()
            record = OperationRecord(self.execution_id, key, operation_type, self.phase,
                                     OperationState.PREPARED, digest, policy, now, now,
                                     agent_id=agent_id, parent_key=_active_operation_key.get(),
                                     idempotency_key=sha256(f"{self.execution_id}:{key}".encode()).hexdigest())
            self.store.create_operation(record)
        elif (record.input_digest != digest or record.operation_type != operation_type or
              record.replay_policy != policy):
            raise ExecutionRecoveryBlocked(f"Operation identity changed: {key}",
                                           code="OPERATION_INPUT_MISMATCH", operation_key=key)
        if record.state is OperationState.COMMITTED:
            self._event(record, "reused")
            return loads(record.result_json)
        if record.state is OperationState.IN_FLIGHT:
            record = self.store.transition_operation(self.execution_id, key, record.version,
                                                     OperationState.UNCERTAIN)
        if record.state is OperationState.UNCERTAIN:
            if reconcile is not None:
                try:
                    resolved = reconcile(record)
                except ReconciledFailure:
                    record = self.store.transition_operation(self.execution_id, key,
                        record.version, OperationState.FAILED,
                        failure_type="ReconciledFailure")
                    raise ExecutionRecoveryBlocked(f"Reconciled operation failed: {key}") from None
                except ExecutionCancelled:
                    raise
                except Exception:
                    raise ExecutionRecoveryBlocked(f"Reconciliation could not establish outcome: {key}",
                                                   code="RECONCILIATION_ERROR",
                                                   operation_key=key) from None
                if resolved is RECONCILED_NOT_COMPLETED:
                    self._event(record, "not_completed")
                    resolved = None
                    policy = "retry"
                if resolved is not None:
                    raw = dumps(resolved)
                    record = self.store.transition_operation(self.execution_id, key, record.version,
                                                             OperationState.COMMITTED, result_json=raw)
                    self._event(record, "reconciled")
                    return resolved
            if policy not in ("retry", "pure", "idempotent"):
                raise ExecutionRecoveryBlocked(f"Uncertain operation requires reconciliation: {key}",
                                               code="UNCERTAIN_OPERATION", operation_key=key)
        if record.state in (OperationState.FAILED, OperationState.CANCELLED):
            raise ExecutionRecoveryBlocked(f"Terminal operation cannot replay: {key}")
        check_execution()
        record = self.store.transition_operation(self.execution_id, key, record.version,
                                                 OperationState.IN_FLIGHT)
        token = _active_operation_key.set(key)
        try:
            result = action()
            check_execution()
            raw = dumps(result)
            record = self.store.transition_operation(self.execution_id, key, record.version,
                                                     OperationState.COMMITTED, result_json=raw)
            return result
        except ExecutionCancelled:
            self.store.transition_operation(self.execution_id, key, record.version,
                                            OperationState.CANCELLED)
            raise
        except ExecutionRecoveryBlocked:
            # A child may be uncertain. The parent is retryable after the child is reconciled.
            raise
        except Exception as error:
            self.store.transition_operation(self.execution_id, key, record.version,
                                            OperationState.FAILED,
                                            failure_type=type(error).__name__)
            raise
        finally:
            _active_operation_key.reset(token)
