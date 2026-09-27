"""Versioned execution records and local transactional stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any
import sqlite3

from agenttree.core.execution_codec import ExecutionDataError, dumps, loads


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLATION_REQUESTED = "cancellation_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = frozenset((ExecutionState.COMPLETED, ExecutionState.FAILED, ExecutionState.CANCELLED))
_NEXT = {
    ExecutionState.QUEUED: frozenset((ExecutionState.RUNNING, ExecutionState.CANCELLED)),
    ExecutionState.RUNNING: frozenset((ExecutionState.CANCELLATION_REQUESTED,
                                    ExecutionState.COMPLETED, ExecutionState.FAILED)),
    ExecutionState.CANCELLATION_REQUESTED: frozenset((ExecutionState.CANCELLED,)),
}
_UPDATE_FIELDS = frozenset(("owner_pid", "owner_id", "in_flight"))


class ExecutionConflict(RuntimeError):
    """A stale writer or illegal state transition was rejected."""


class ExecutionRecoveryBlocked(RuntimeError):
    """Automatic continuation cannot be proved safe."""

    def __init__(self, message: str, *, code: str = "RECOVERY_BLOCKED",
                 operation_key: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.operation_key = operation_key


class OperationState(str, Enum):
    PREPARED = "prepared"
    IN_FLIGHT = "in_flight"
    COMMITTED = "committed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"


_OP_NEXT = {
    OperationState.PREPARED: frozenset((OperationState.IN_FLIGHT, OperationState.CANCELLED)),
    OperationState.IN_FLIGHT: frozenset((OperationState.COMMITTED, OperationState.FAILED,
                                        OperationState.CANCELLED, OperationState.UNCERTAIN)),
    OperationState.UNCERTAIN: frozenset((OperationState.IN_FLIGHT, OperationState.COMMITTED,
                                        OperationState.FAILED)),
}


@dataclass(frozen=True)
class OperationRecord:
    execution_id: str
    operation_key: str
    operation_type: str
    phase: str
    state: OperationState
    input_digest: str
    replay_policy: str
    created_at: datetime
    updated_at: datetime
    agent_id: str | None = None
    parent_key: str | None = None
    idempotency_key: str | None = None
    result_json: str | None = None
    failure_type: str | None = None
    attempt: int = 0
    version: int = 0

    def __post_init__(self) -> None:
        if (not self.execution_id or not self.operation_key or not self.operation_type or
                not self.phase or len(self.operation_key) > 512 or
                len(self.operation_type) > 64 or len(self.phase) > 64 or
                not isinstance(self.state, OperationState) or
                len(self.input_digest) != 64 or any(c not in "0123456789abcdef" for c in self.input_digest) or
                self.replay_policy not in ("retry", "pure", "idempotent", "reconcilable",
                                           "non_idempotent", "unknown") or
                self.attempt < 0 or self.version < 0):
            raise ExecutionDataError("Invalid operation record")
        if self.state is OperationState.COMMITTED and self.result_json is None:
            raise ExecutionDataError("Committed operation requires a result")
        if self.result_json is not None:
            loads(self.result_json)


@dataclass(frozen=True)
class OperationAttempt:
    execution_id: str
    operation_key: str
    number: int
    started_at: datetime
    outcome: str = "in_flight"
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.number < 1 or self.started_at.tzinfo is None:
            raise ExecutionDataError("Invalid operation attempt")


@dataclass(frozen=True)
class DurableEvent:
    execution_id: str
    sequence: int
    event: Any

    def __post_init__(self) -> None:
        from agenttree.models import ExecutionEvent
        if (self.sequence < 1 or not isinstance(self.event, ExecutionEvent) or
                self.event.task_id != self.execution_id):
            raise ExecutionDataError("Invalid durable event")


def _operation_change(record: OperationRecord, state: OperationState,
                      *, result_json: str | None = None,
                      failure_type: str | None = None) -> OperationRecord:
    if state not in _OP_NEXT.get(record.state, ()):
        raise ExecutionConflict("Invalid operation transition")
    if state is OperationState.COMMITTED:
        if result_json is None:
            raise ExecutionConflict("Committed operation requires a result")
        loads(result_json)
    elif result_json is not None:
        raise ExecutionConflict("Result requires commit transition")
    if failure_type is not None and state is not OperationState.FAILED:
        raise ExecutionConflict("Failure type requires failed transition")
    return replace(record, state=state, version=record.version + 1,
                   updated_at=utc_now(), result_json=result_json or record.result_json,
                   failure_type=failure_type,
                   attempt=record.attempt + (state is OperationState.IN_FLIGHT))


def _operation_event(record: OperationRecord) -> Any:
    from agenttree.models import ExecutionEvent
    return ExecutionEvent(task_id=record.execution_id, actor_id=record.agent_id,
                          event_type=f"operation.{record.state.value}",
                          metadata={"operation_key": record.operation_key,
                                    "operation_type": record.operation_type,
                                    "phase": record.phase, "attempt": record.attempt,
                                    "status": record.state.value})


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    state: ExecutionState
    fingerprint: str
    input_json: str
    created_at: datetime
    updated_at: datetime
    version: int = 0
    checkpoint_sequence: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancellation_requested_at: datetime | None = None
    deadline_at: datetime | None = None
    owner_pid: int | None = None
    owner_id: str | None = None
    current_phase: str | None = None
    result_json: str | None = None
    failure_type: str | None = None
    in_flight: str | None = None
    events_json: str = "[]"

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str) or not self.execution_id:
            raise ExecutionDataError("Invalid execution ID")
        if not isinstance(self.state, ExecutionState):
            raise ExecutionDataError("Invalid execution state")
        if (not isinstance(self.version, int) or self.version < 0 or
                not isinstance(self.checkpoint_sequence, int) or self.checkpoint_sequence < 0):
            raise ExecutionDataError("Invalid execution version")
        for name in ("created_at", "updated_at", "started_at", "finished_at",
                     "cancellation_requested_at", "deadline_at"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, datetime) or value.tzinfo is None):
                raise ExecutionDataError("Invalid execution timestamp")


@dataclass(frozen=True)
class ExecutionCheckpoint:
    schema_version: int
    execution_id: str
    sequence: int
    created_at: datetime
    phase: str
    state_json: str
    usage_json: str
    tool_json: str
    collaboration_json: str
    runtime_json: str = "[]"

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ExecutionDataError("Unsupported checkpoint version")
        if not isinstance(self.execution_id, str) or not self.execution_id:
            raise ExecutionDataError("Invalid checkpoint execution ID")
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None:
            raise ExecutionDataError("Invalid checkpoint timestamp")
        if not isinstance(self.phase, str) or not self.phase:
            raise ExecutionDataError("Invalid checkpoint phase")


def _validate_checkpoint(checkpoint: ExecutionCheckpoint) -> None:
    from agenttree.models import WorkflowState, ExecutionEvent
    state = loads(checkpoint.state_json)
    if (not isinstance(state, WorkflowState) or state.task_id != checkpoint.execution_id or
            state.current_phase.value != checkpoint.phase):
        raise ExecutionDataError("Checkpoint workflow state is invalid")
    usage = loads(checkpoint.usage_json)
    tool = loads(checkpoint.tool_json)
    collaboration = loads(checkpoint.collaboration_json)
    runtime_events = loads(checkpoint.runtime_json)
    if (not isinstance(usage, dict) or not isinstance(usage.get("calls"), (tuple, list)) or
            not isinstance(tool, dict) or not isinstance(tool.get("calls"), dict) or
            not isinstance(tool.get("metrics"), dict) or
            not isinstance(tool.get("events"), (tuple, list)) or
            not isinstance(collaboration, dict) or
            not isinstance(collaboration.get("events"), (tuple, list)) or
            not isinstance(collaboration.get("messages"), (tuple, list)) or
            not isinstance(collaboration.get("threads"), dict) or
            not isinstance(runtime_events, (tuple, list)) or
            any(not isinstance(item, ExecutionEvent) or item.task_id != checkpoint.execution_id
                for item in (*tool["events"], *collaboration["events"], *runtime_events))):
        raise ExecutionDataError("Checkpoint runtime data is invalid")


def _change(record: ExecutionRecord, state: ExecutionState, **updates: Any) -> ExecutionRecord:
    if not isinstance(state, ExecutionState) or state not in _NEXT.get(record.state, ()):
        raise ExecutionConflict(f"Invalid execution transition: {record.state.value} -> {state}")
    permitted = ({"owner_pid", "owner_id", "events_json"} if state is ExecutionState.RUNNING
                 else {"result_json", "failure_type", "events_json"} if state in
                 (ExecutionState.COMPLETED, ExecutionState.FAILED)
                 else {"events_json"})
    if set(updates) - permitted:
        raise ExecutionConflict("Invalid execution transition fields")
    now = utc_now()
    if state is ExecutionState.RUNNING:
        updates["started_at"] = record.started_at or now
    if state is ExecutionState.CANCELLATION_REQUESTED:
        updates["cancellation_requested_at"] = now
    if state in TERMINAL:
        updates["finished_at"] = now
        updates["owner_pid"] = None
        updates["owner_id"] = None
    return replace(record, state=state, updated_at=now, version=record.version + 1, **updates)


class ExecutionStore(ABC):
    @abstractmethod
    def create(self, record: ExecutionRecord) -> None: ...

    @abstractmethod
    def get(self, execution_id: str) -> ExecutionRecord: ...

    @abstractmethod
    def transition(self, execution_id: str, expected_version: int,
                   state: ExecutionState, *, durable_events: tuple[Any, ...] = (),
                   **updates: Any) -> ExecutionRecord: ...

    @abstractmethod
    def update(self, execution_id: str, expected_version: int,
               **updates: Any) -> ExecutionRecord: ...

    @abstractmethod
    def checkpoint(self, execution_id: str, expected_version: int,
                   checkpoint: ExecutionCheckpoint) -> ExecutionRecord: ...

    @abstractmethod
    def load_checkpoint(self, execution_id: str) -> ExecutionCheckpoint | None: ...

    @abstractmethod
    def list_recoverable(self) -> tuple[ExecutionRecord, ...]: ...

    @abstractmethod
    def create_operation(self, operation: OperationRecord) -> None: ...

    @abstractmethod
    def get_operation(self, execution_id: str, operation_key: str) -> OperationRecord | None: ...

    @abstractmethod
    def transition_operation(self, execution_id: str, operation_key: str,
                             expected_version: int, state: OperationState,
                             *, result_json: str | None = None,
                             failure_type: str | None = None) -> OperationRecord: ...

    @abstractmethod
    def list_operations(self, execution_id: str) -> tuple[OperationRecord, ...]: ...

    @abstractmethod
    def list_attempts(self, execution_id: str, operation_key: str) -> tuple[OperationAttempt, ...]: ...

    @abstractmethod
    def append_event(self, execution_id: str, event: Any) -> DurableEvent: ...

    @abstractmethod
    def read_events(self, execution_id: str, *, after: int = 0,
                    limit: int = 100) -> tuple[DurableEvent, ...]: ...


class InMemoryExecutionStore(ExecutionStore):
    def __init__(self) -> None:
        self._records: dict[str, ExecutionRecord] = {}
        self._checkpoints: dict[str, ExecutionCheckpoint] = {}
        self._lock = RLock()
        self._operations: dict[tuple[str, str], OperationRecord] = {}
        self._attempts: dict[tuple[str, str], list[OperationAttempt]] = {}
        self._events: dict[str, list[DurableEvent]] = {}

    def create(self, record: ExecutionRecord) -> None:
        with self._lock:
            if record.execution_id in self._records:
                raise ExecutionConflict("Execution ID already exists")
            self._records[record.execution_id] = deepcopy(record)

    def get(self, execution_id: str) -> ExecutionRecord:
        with self._lock:
            return deepcopy(self._records[execution_id])

    def update(self, execution_id: str, expected_version: int, **updates: Any) -> ExecutionRecord:
        if set(updates) - _UPDATE_FIELDS:
            raise ExecutionConflict("Execution fields require a validated transition")
        with self._lock:
            record = self.get(execution_id)
            if record.version != expected_version or record.state in TERMINAL:
                raise ExecutionConflict("Stale or terminal execution")
            changed = replace(record, updated_at=utc_now(), version=record.version + 1, **updates)
            self._records[execution_id] = changed
            return deepcopy(changed)

    def transition(self, execution_id: str, expected_version: int,
                   state: ExecutionState, *, durable_events: tuple[Any, ...] = (),
                   **updates: Any) -> ExecutionRecord:
        with self._lock:
            record = self.get(execution_id)
            if record.version != expected_version:
                raise ExecutionConflict("Stale execution version")
            changed = _change(record, state, **updates)
            from agenttree.models import ExecutionEvent
            safe_events = []
            for event in durable_events:
                if not isinstance(event, ExecutionEvent) or event.task_id != execution_id:
                    raise ExecutionDataError("Invalid durable event")
                safe_events.append(loads(dumps(event, max_bytes=65_536)))
            events = self._events.setdefault(execution_id, [])
            for event in safe_events:
                events.append(DurableEvent(execution_id, len(events) + 1, event))
            self._records[execution_id] = changed
            return deepcopy(changed)

    def checkpoint(self, execution_id: str, expected_version: int,
                   checkpoint: ExecutionCheckpoint) -> ExecutionRecord:
        with self._lock:
            record = self.get(execution_id)
            if (record.version != expected_version or record.state is not ExecutionState.RUNNING or
                    checkpoint.execution_id != execution_id or checkpoint.schema_version != 1 or
                    checkpoint.sequence != record.checkpoint_sequence + 1):
                raise ExecutionConflict("Invalid or stale checkpoint")
            _validate_checkpoint(checkpoint)
            changed = replace(record, version=record.version + 1,
                              checkpoint_sequence=checkpoint.sequence,
                              current_phase=checkpoint.phase, in_flight=None,
                              events_json=checkpoint.runtime_json,
                              updated_at=utc_now())
            self._checkpoints[execution_id] = deepcopy(checkpoint)
            self._records[execution_id] = changed
            return deepcopy(changed)

    def load_checkpoint(self, execution_id: str) -> ExecutionCheckpoint | None:
        with self._lock:
            return deepcopy(self._checkpoints.get(execution_id))

    def list_recoverable(self) -> tuple[ExecutionRecord, ...]:
        with self._lock:
            return tuple(deepcopy(item) for item in self._records.values() if item.state not in TERMINAL)

    def create_operation(self, operation: OperationRecord) -> None:
        with self._lock:
            key = operation.execution_id, operation.operation_key
            if operation.execution_id not in self._records or key in self._operations or operation.state is not OperationState.PREPARED:
                raise ExecutionConflict("Invalid or duplicate operation")
            self._operations[key] = deepcopy(operation)
            self.append_event(operation.execution_id, _operation_event(operation))

    def get_operation(self, execution_id: str, operation_key: str) -> OperationRecord | None:
        with self._lock:
            return deepcopy(self._operations.get((execution_id, operation_key)))

    def transition_operation(self, execution_id: str, operation_key: str,
                             expected_version: int, state: OperationState,
                             *, result_json: str | None = None,
                             failure_type: str | None = None) -> OperationRecord:
        with self._lock:
            key = execution_id, operation_key
            record = self._operations[key]
            if record.version != expected_version:
                raise ExecutionConflict("Stale operation version")
            changed = _operation_change(record, state, result_json=result_json,
                                        failure_type=failure_type)
            self._operations[key] = changed
            attempts = self._attempts.setdefault(key, [])
            if state is OperationState.IN_FLIGHT:
                attempts.append(OperationAttempt(execution_id, operation_key, changed.attempt, utc_now()))
            elif attempts and record.state in (OperationState.IN_FLIGHT, OperationState.UNCERTAIN):
                attempts[-1] = replace(attempts[-1], outcome=state.value, finished_at=utc_now())
            self.append_event(execution_id, _operation_event(changed))
            return deepcopy(changed)

    def list_operations(self, execution_id: str) -> tuple[OperationRecord, ...]:
        with self._lock:
            return tuple(deepcopy(op) for op in self._operations.values() if op.execution_id == execution_id)

    def list_attempts(self, execution_id: str, operation_key: str) -> tuple[OperationAttempt, ...]:
        with self._lock:
            return tuple(deepcopy(self._attempts.get((execution_id, operation_key), ())))

    def append_event(self, execution_id: str, event: Any) -> DurableEvent:
        from agenttree.models import ExecutionEvent
        if not isinstance(event, ExecutionEvent) or event.task_id != execution_id:
            raise ExecutionDataError("Invalid durable event")
        safe_event = loads(dumps(event, max_bytes=65_536))
        with self._lock:
            if execution_id not in self._records:
                raise KeyError(execution_id)
            events = self._events.setdefault(execution_id, [])
            item = DurableEvent(execution_id, len(events) + 1, safe_event)
            events.append(item)
            return deepcopy(item)

    def read_events(self, execution_id: str, *, after: int = 0,
                    limit: int = 100) -> tuple[DurableEvent, ...]:
        if after < 0 or not 1 <= limit <= 1000:
            raise ValueError("Invalid event cursor or limit")
        with self._lock:
            if execution_id not in self._records:
                raise KeyError(execution_id)
            return tuple(deepcopy(self._events.get(execution_id, ()))[after:after + limit])


class SQLiteExecutionStore(ExecutionStore):
    """SQLite CAS store; connections are short-lived and never shared by threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with self._connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS execution_schema (version INTEGER NOT NULL)")
            row = db.execute("SELECT version FROM execution_schema").fetchone()
            if row is None:
                db.execute("INSERT INTO execution_schema VALUES (1)")
            elif row[0] != 1:
                raise ExecutionDataError("Unsupported execution store schema")
            db.execute("CREATE TABLE IF NOT EXISTS executions (execution_id TEXT PRIMARY KEY, version INTEGER NOT NULL, state TEXT NOT NULL, data TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS checkpoints (execution_id TEXT PRIMARY KEY, sequence INTEGER NOT NULL, data TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS operations (execution_id TEXT NOT NULL, operation_key TEXT NOT NULL, version INTEGER NOT NULL, state TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(execution_id,operation_key))")
            db.execute("CREATE INDEX IF NOT EXISTS operations_by_state ON operations(execution_id,state)")
            db.execute("CREATE TABLE IF NOT EXISTS operation_attempts (execution_id TEXT NOT NULL, operation_key TEXT NOT NULL, number INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY(execution_id,operation_key,number))")
            db.execute("CREATE TABLE IF NOT EXISTS durable_events (execution_id TEXT NOT NULL, sequence INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY(execution_id,sequence))")

    @staticmethod
    def _append_event_transaction(db, execution_id: str, event: Any) -> DurableEvent:
        safe_event = loads(dumps(event, max_bytes=65_536))
        row = db.execute("SELECT COALESCE(MAX(sequence),0) FROM durable_events WHERE execution_id=?",
                         (execution_id,)).fetchone()
        item = DurableEvent(execution_id, row[0] + 1, safe_event)
        db.execute("INSERT INTO durable_events VALUES (?,?,?)",
                   (execution_id, item.sequence, dumps(item, max_bytes=65_536)))
        return item

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA busy_timeout=10000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, record: ExecutionRecord) -> None:
        raw = dumps(record)
        try:
            with self._connection() as db:
                db.execute("INSERT INTO executions VALUES (?,?,?,?)",
                           (record.execution_id, record.version, record.state.value, raw))
        except sqlite3.IntegrityError as error:
            raise ExecutionConflict("Execution ID already exists") from error

    def get(self, execution_id: str) -> ExecutionRecord:
        with self._connection() as db:
            row = db.execute("SELECT data FROM executions WHERE execution_id=?", (execution_id,)).fetchone()
        if row is None:
            raise KeyError(execution_id)
        record = loads(row[0])
        if not isinstance(record, ExecutionRecord):
            raise ExecutionDataError("Malformed execution record")
        return record

    def _write(self, execution_id: str, expected_version: int,
               state: ExecutionState | None, updates: dict[str, Any],
               checkpoint: ExecutionCheckpoint | None = None,
               durable_events: tuple[Any, ...] = ()) -> ExecutionRecord:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM executions WHERE execution_id=?", (execution_id,)).fetchone()
            if row is None:
                raise KeyError(execution_id)
            record = loads(row[0])
            if record.version != expected_version:
                raise ExecutionConflict("Stale execution version")
            if checkpoint is not None:
                if (record.state is not ExecutionState.RUNNING or
                        checkpoint.execution_id != execution_id or checkpoint.schema_version != 1 or
                        checkpoint.sequence != record.checkpoint_sequence + 1):
                    raise ExecutionConflict("Invalid checkpoint")
                _validate_checkpoint(checkpoint)
                changed = replace(record, version=record.version + 1,
                                  checkpoint_sequence=checkpoint.sequence,
                                  current_phase=checkpoint.phase, in_flight=None,
                                  events_json=checkpoint.runtime_json,
                                  updated_at=utc_now())
                db.execute("INSERT INTO checkpoints VALUES (?,?,?) ON CONFLICT(execution_id) DO UPDATE SET sequence=excluded.sequence,data=excluded.data",
                           (execution_id, checkpoint.sequence, dumps(checkpoint)))
            elif state is not None:
                changed = _change(record, state, **updates)
            else:
                if record.state in TERMINAL:
                    raise ExecutionConflict("Terminal execution cannot be updated")
                changed = replace(record, version=record.version + 1, updated_at=utc_now(), **updates)
            written = db.execute("UPDATE executions SET version=?,state=?,data=? WHERE execution_id=? AND version=?",
                                 (changed.version, changed.state.value, dumps(changed), execution_id, expected_version))
            if written.rowcount != 1:
                raise ExecutionConflict("Stale execution writer")
            from agenttree.models import ExecutionEvent
            for event in durable_events:
                if not isinstance(event, ExecutionEvent) or event.task_id != execution_id:
                    raise ExecutionDataError("Invalid durable event")
                self._append_event_transaction(db, execution_id, event)
        return changed

    def update(self, execution_id: str, expected_version: int, **updates: Any) -> ExecutionRecord:
        if set(updates) - _UPDATE_FIELDS:
            raise ExecutionConflict("Execution fields require a validated transition")
        return self._write(execution_id, expected_version, None, updates)

    def transition(self, execution_id: str, expected_version: int,
                   state: ExecutionState, *, durable_events: tuple[Any, ...] = (),
                   **updates: Any) -> ExecutionRecord:
        return self._write(execution_id, expected_version, state, updates,
                           durable_events=durable_events)

    def checkpoint(self, execution_id: str, expected_version: int,
                   checkpoint: ExecutionCheckpoint) -> ExecutionRecord:
        return self._write(execution_id, expected_version, None, {}, checkpoint)

    def load_checkpoint(self, execution_id: str) -> ExecutionCheckpoint | None:
        with self._connection() as db:
            row = db.execute("SELECT data FROM checkpoints WHERE execution_id=?", (execution_id,)).fetchone()
        if row is None:
            return None
        checkpoint = loads(row[0])
        if not isinstance(checkpoint, ExecutionCheckpoint) or checkpoint.schema_version != 1:
            raise ExecutionDataError("Unsupported or malformed checkpoint")
        return checkpoint

    def list_recoverable(self) -> tuple[ExecutionRecord, ...]:
        with self._connection() as db:
            rows = db.execute("SELECT data FROM executions WHERE state IN (?,?,?) ORDER BY rowid",
                              (ExecutionState.QUEUED.value, ExecutionState.RUNNING.value,
                               ExecutionState.CANCELLATION_REQUESTED.value)).fetchall()
        return tuple(loads(row[0]) for row in rows)

    def create_operation(self, operation: OperationRecord) -> None:
        if operation.state is not OperationState.PREPARED:
            raise ExecutionConflict("Operation must start prepared")
        try:
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT 1 FROM executions WHERE execution_id=?", (operation.execution_id,)).fetchone() is None:
                    raise KeyError(operation.execution_id)
                db.execute("INSERT INTO operations VALUES (?,?,?,?,?)",
                           (operation.execution_id, operation.operation_key, operation.version,
                            operation.state.value, dumps(operation)))
                self._append_event_transaction(db, operation.execution_id,
                                               _operation_event(operation))
        except sqlite3.IntegrityError as error:
            raise ExecutionConflict("Operation key already exists") from error

    def get_operation(self, execution_id: str, operation_key: str) -> OperationRecord | None:
        with self._connection() as db:
            row = db.execute("SELECT data FROM operations WHERE execution_id=? AND operation_key=?",
                             (execution_id, operation_key)).fetchone()
        if row is None:
            return None
        record = loads(row[0])
        if not isinstance(record, OperationRecord):
            raise ExecutionDataError("Malformed operation record")
        return record

    def transition_operation(self, execution_id: str, operation_key: str,
                             expected_version: int, state: OperationState,
                             *, result_json: str | None = None,
                             failure_type: str | None = None) -> OperationRecord:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM operations WHERE execution_id=? AND operation_key=?",
                             (execution_id, operation_key)).fetchone()
            if row is None:
                raise KeyError(operation_key)
            record = loads(row[0])
            if not isinstance(record, OperationRecord) or record.version != expected_version:
                raise ExecutionConflict("Stale operation version")
            changed = _operation_change(record, state, result_json=result_json,
                                        failure_type=failure_type)
            if state is OperationState.IN_FLIGHT:
                attempt = OperationAttempt(execution_id, operation_key, changed.attempt, utc_now())
                db.execute("INSERT INTO operation_attempts VALUES (?,?,?,?)",
                           (execution_id, operation_key, attempt.number, dumps(attempt)))
            elif record.attempt:
                attempt_row = db.execute("SELECT data FROM operation_attempts WHERE execution_id=? AND operation_key=? AND number=?",
                                         (execution_id, operation_key, record.attempt)).fetchone()
                if attempt_row is not None:
                    attempt = replace(loads(attempt_row[0]), outcome=state.value, finished_at=utc_now())
                    db.execute("UPDATE operation_attempts SET data=? WHERE execution_id=? AND operation_key=? AND number=?",
                               (dumps(attempt), execution_id, operation_key, record.attempt))
            written = db.execute("UPDATE operations SET version=?,state=?,data=? WHERE execution_id=? AND operation_key=? AND version=?",
                                 (changed.version, state.value, dumps(changed), execution_id,
                                  operation_key, expected_version))
            if written.rowcount != 1:
                raise ExecutionConflict("Stale operation writer")
            self._append_event_transaction(db, execution_id, _operation_event(changed))
        return changed

    def list_operations(self, execution_id: str) -> tuple[OperationRecord, ...]:
        with self._connection() as db:
            rows = db.execute("SELECT data FROM operations WHERE execution_id=? ORDER BY rowid",
                              (execution_id,)).fetchall()
        return tuple(loads(row[0]) for row in rows)

    def list_attempts(self, execution_id: str, operation_key: str) -> tuple[OperationAttempt, ...]:
        with self._connection() as db:
            rows = db.execute("SELECT data FROM operation_attempts WHERE execution_id=? AND operation_key=? ORDER BY number",
                              (execution_id, operation_key)).fetchall()
        return tuple(loads(row[0]) for row in rows)

    def append_event(self, execution_id: str, event: Any) -> DurableEvent:
        from agenttree.models import ExecutionEvent
        if not isinstance(event, ExecutionEvent) or event.task_id != execution_id:
            raise ExecutionDataError("Invalid durable event")
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM executions WHERE execution_id=?", (execution_id,)).fetchone() is None:
                raise KeyError(execution_id)
            item = self._append_event_transaction(db, execution_id, event)
        return item

    def read_events(self, execution_id: str, *, after: int = 0,
                    limit: int = 100) -> tuple[DurableEvent, ...]:
        if after < 0 or not 1 <= limit <= 1000:
            raise ValueError("Invalid event cursor or limit")
        with self._connection() as db:
            if db.execute("SELECT 1 FROM executions WHERE execution_id=?", (execution_id,)).fetchone() is None:
                raise KeyError(execution_id)
            rows = db.execute("SELECT data FROM durable_events WHERE execution_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                              (execution_id, after, limit)).fetchall()
        return tuple(loads(row[0]) for row in rows)
