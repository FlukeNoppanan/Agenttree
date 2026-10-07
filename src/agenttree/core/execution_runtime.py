"""Bounded background execution over the existing AgentTree orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from hashlib import sha256
import json
from math import isfinite
import os
from queue import Queue
from threading import Event, Lock, Semaphore, Thread
from time import monotonic
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agenttree.core.execution_codec import ExecutionDataError, dumps, loads
from agenttree.core.execution_control import ExecutionCancelled, ExecutionControl
from agenttree.core.execution_store import (
    ExecutionCheckpoint, ExecutionConflict, ExecutionRecord,
    ExecutionRecoveryBlocked, ExecutionState, ExecutionStore,
    InMemoryExecutionStore, OperationState, TERMINAL, utc_now,
)
from agenttree.core.operation_journal import OperationJournal, _active_operation_journal
from agenttree.core.artifact_store import (ArtifactStore, FileArtifactStore,
                                           InMemoryArtifactStore)
from agenttree.models import ExecutionEvent, ExecutionTrace, Task, WorkflowState
from agenttree.orchestration.models import FinalResult

if TYPE_CHECKING:
    from agenttree.framework import AgentTree


class ExecutionCapacityError(RuntimeError):
    """The bounded runtime has no worker or pending capacity."""


class ExecutionFailed(RuntimeError):
    """The execution ended without a successful FinalResult."""


@dataclass(frozen=True)
class ExecutionInfo:
    execution_id: str
    state: ExecutionState
    version: int
    checkpoint_sequence: int
    current_phase: str | None
    created_at: datetime
    started_at: datetime | None
    updated_at: datetime
    finished_at: datetime | None
    cancellation_requested_at: datetime | None
    deadline_at: datetime | None
    failure_type: str | None
    recovery_count: int = 0
    cancellation_count: int = 0
    operations_total: int = 0
    operations_reused: int = 0
    operations_retried: int = 0
    operations_uncertain: int = 0
    recovery_block_count: int = 0

    @property
    def metrics(self) -> dict[str, int | float | None]:
        queue_ms = ((self.started_at - self.created_at).total_seconds() * 1000
                    if self.started_at is not None else None)
        duration_ms = ((self.finished_at - self.started_at).total_seconds() * 1000
                       if self.started_at is not None and self.finished_at is not None else None)
        return {"queue_duration_ms": queue_ms, "execution_duration_ms": duration_ms,
                "checkpoint_count": self.checkpoint_sequence,
                "recovery_count": self.recovery_count,
                "cancellation_count": self.cancellation_count,
                "operations_total": self.operations_total,
                "operations_reused": self.operations_reused,
                "operations_retried": self.operations_retried,
                "operations_uncertain": self.operations_uncertain,
                "recovery_block_count": self.recovery_block_count}


def tree_fingerprint(tree: AgentTree) -> str:
    """Hash execution-relevant public identities, excluding credentials/clients."""
    agents = (tree.root_agent, *tree.managers, *tree.specialists)
    def digest(text: str) -> str:
        return sha256(text.encode()).hexdigest()
    def strategy_shape(strategy: Any) -> tuple[str, str | None]:
        if strategy is None:
            return "none", None
        safe = {}
        for key, value in vars(strategy).items():
            if key.startswith("_provider") or isinstance(value, BaseProvider):
                continue
            if isinstance(value, (str, int, float, bool, type(None), tuple, list, dict)):
                try:
                    safe[key] = dumps(value, max_bytes=32_768)
                except ExecutionDataError:
                    pass
        return f"{type(strategy).__module__}.{type(strategy).__qualname__}", digest(json.dumps(safe, sort_keys=True))
    from agenttree.providers import BaseProvider
    structure = {
        "agents": [(type(item).__name__, item.id, item.name, tuple(item.capabilities),
                    digest(item.description))
                   for item in agents],
        "ownership": [(item.id, tuple(peer.id for peer in item.specialists)) for item in tree.managers],
        "providers": tuple(tree.provider_bindings.items()),
        "models": tuple(tree.model_bindings.items()),
        "tools": tuple((item.id, item.name, digest(item.description),
                        digest(repr(item.input_spec)), item.enabled,
                        item.recovery_policy.value) for item in tree.tools),
        "tool_bindings": tuple(tree.tool_bindings.items()),
        "permissions": tuple(tree.manager_permissions.items()),
        "config": {key: value for key, value in vars(tree.config).items()},
        "strategies": tuple(strategy_shape(getattr(tree, key)) for key in (
            "_decomposer", "_manager_reviewer", "_final_reviewer", "_root_planner",
            "_root_synthesizer", "_executor", "_engine")),
    }
    return sha256(json.dumps(structure, sort_keys=True, default=str).encode()).hexdigest()


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lifecycle(execution_id: str, label: str, execution_mode: str | None = None) -> ExecutionEvent:
    return ExecutionEvent(task_id=execution_id, event_type=f"execution.{label}",
                          metadata={"execution_mode": execution_mode} if execution_mode else {})


def _with_runtime_trace(result: FinalResult, events: list[ExecutionEvent]) -> FinalResult:
    trace = ExecutionTrace(task_id=result.task_id)
    all_events = sorted((*result.trace.events, *events), key=lambda item: item.timestamp)
    completion = [item for item in all_events if item.event_type == "execution.completed"]
    for event in all_events:
        if event.event_type != "execution.completed":
            trace.append(event)
    for event in completion:
        trace.append(event)
    return replace(result, trace=trace)


class ExecutionHandle:
    def __init__(self, runtime: ExecutionRuntime, execution_id: str) -> None:
        self._runtime = runtime
        self.execution_id = execution_id

    def status(self) -> ExecutionInfo:
        return self._runtime.get_execution(self.execution_id)

    def cancel(self) -> ExecutionInfo:
        return self._runtime.cancel(self.execution_id)

    def result(self, timeout: float | None = None) -> FinalResult:
        return self._runtime.result(self.execution_id, timeout=timeout)

    def operations(self):
        return self._runtime.operations(self.execution_id)

    def events(self, *, after: int = 0, limit: int = 100):
        return self._runtime.events(self.execution_id, after=after, limit=limit)

    def artifacts(self):
        return self._runtime.artifacts(self.execution_id)

    def artifact(self, artifact_id: str) -> bytes:
        return self._runtime.artifact(self.execution_id, artifact_id)

    def stream_events(self, *, after_sequence: int = 0, timeout: float | None = None,
                      stop_on_terminal: bool = True, page_size: int = 100):
        return self._runtime.stream_events(self.execution_id,
            after_sequence=after_sequence, timeout=timeout,
            stop_on_terminal=stop_on_terminal, page_size=page_size)

    def stream_output(self, *, timeout: float | None = None):
        """Subscribe to future process-local Agent text; deltas are not replayable."""
        return self._runtime.stream_output(self.execution_id, timeout=timeout)


class ExecutionRuntime:
    """Local bounded worker queue with explicit recovery and host-supplied store."""

    def __init__(self, store: ExecutionStore | None = None, *, max_workers: int = 2,
                 max_pending: int = 8, max_record_bytes: int = 2_000_000,
                 artifact_store: ArtifactStore | None = None) -> None:
        if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
            raise ValueError("max_workers must be positive")
        if not isinstance(max_pending, int) or isinstance(max_pending, bool) or max_pending < 0:
            raise ValueError("max_pending must be nonnegative")
        if (not isinstance(max_record_bytes, int) or isinstance(max_record_bytes, bool) or
                not 1 <= max_record_bytes <= 2_000_000):
            raise ValueError("max_record_bytes must be between 1 and 2,000,000")
        if store is not None and not isinstance(store, ExecutionStore):
            raise TypeError("store must implement ExecutionStore")
        self.store = store if store is not None else InMemoryExecutionStore()
        if artifact_store is not None and not isinstance(artifact_store, ArtifactStore):
            raise TypeError("artifact_store must implement ArtifactStore")
        if artifact_store is None:
            from agenttree.core.execution_store import SQLiteExecutionStore
            artifact_store = (FileArtifactStore(self.store.path + ".artifacts")
                              if isinstance(self.store, SQLiteExecutionStore)
                              else InMemoryArtifactStore())
        self.artifact_store = artifact_store
        self.max_record_bytes = max_record_bytes
        self.owner_id = str(uuid4())
        self._queue: Queue[tuple[str, AgentTree, Event] | None] = Queue()
        self._capacity = Semaphore(max_workers + max_pending)
        self._controls: dict[str, Event] = {}
        from agenttree.core.live_output import LiveOutputHub
        self._output_hubs: dict[str, LiveOutputHub] = {}
        self._lock = Lock()
        self._closed = False
        self._workers = [Thread(target=self._worker, daemon=True,
                                name=f"agenttree-execution-{index}") for index in range(max_workers)]
        for worker in self._workers:
            worker.start()

    def submit(self, tree: AgentTree, task: Task, *, timeout: float | None = None) -> ExecutionHandle:
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or not isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be positive")
        with self._lock:
            if self._closed:
                raise RuntimeError("ExecutionRuntime is shut down")
            if not self._capacity.acquire(blocking=False):
                raise ExecutionCapacityError("Execution queue capacity exceeded")
            try:
                now = utc_now()
                record = ExecutionRecord(
                    execution_id=task.id, state=ExecutionState.QUEUED,
                    fingerprint=tree_fingerprint(tree),
                    input_json=dumps(task, max_bytes=self.max_record_bytes),
                    created_at=now, updated_at=now,
                    deadline_at=now + timedelta(seconds=timeout) if timeout else None,
                    events_json=dumps([_lifecycle(task.id, "queued", task.execution_mode.value)]),
                )
                self.store.create(record)
                self.store.append_event(task.id, _lifecycle(task.id, "queued", task.execution_mode.value))
                signal = Event()
                self._controls[task.id] = signal
                from agenttree.core.live_output import LiveOutputHub
                self._output_hubs[task.id] = LiveOutputHub(task.id)
                self._queue.put_nowait((task.id, tree, signal))
            except Exception:
                self._capacity.release()
                raise
        return ExecutionHandle(self, task.id)

    def get_execution(self, execution_id: str) -> ExecutionInfo:
        record = self.store.get(execution_id)
        events = loads(record.events_json)
        operations = self.store.list_operations(execution_id)
        reused = blocked_count = 0
        cursor = 0
        while True:
            page = self.store.read_events(execution_id, after=cursor, limit=1000)
            reused += sum(item.event.event_type == "operation.reused" for item in page)
            blocked_count += sum(item.event.event_type == "execution.recovery.blocked"
                                 for item in page)
            if not page or len(page) < 1000:
                break
            cursor = page[-1].sequence
        return ExecutionInfo(**{name: getattr(record, name) for name in
            ("execution_id", "state", "version", "checkpoint_sequence", "current_phase",
             "created_at", "started_at", "updated_at", "finished_at",
             "cancellation_requested_at", "deadline_at", "failure_type")},
            recovery_count=sum(item.event_type == "execution.recovery.started" for item in events),
            cancellation_count=sum(item.event_type == "execution.cancellation.requested" for item in events),
            operations_total=len(operations),
            operations_reused=reused,
            operations_retried=sum(item.attempt > 1 for item in operations),
            operations_uncertain=sum(item.state is OperationState.UNCERTAIN
                                     for item in operations),
            recovery_block_count=blocked_count)

    def handle(self, execution_id: str) -> ExecutionHandle:
        self.store.get(execution_id)
        return ExecutionHandle(self, execution_id)

    def stream_output(self, execution_id: str, *, timeout: float | None = None):
        if timeout is not None and (isinstance(timeout, bool) or
                                    not isinstance(timeout, (int, float)) or
                                    not isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be nonnegative and finite")
        self.store.get(execution_id)
        with self._lock:
            hub = self._output_hubs.get(execution_id)
            if hub is None:
                from agenttree.core.live_output import LiveOutputHub
                hub = LiveOutputHub(execution_id)
                hub.close()
        return hub.stream(hub.subscribe(), timeout)

    def operations(self, execution_id: str):
        self.store.get(execution_id)
        return self.store.list_operations(execution_id)

    def events(self, execution_id: str, *, after: int = 0, limit: int = 100):
        return self.store.read_events(execution_id, after=after, limit=limit)

    def artifacts(self, execution_id: str):
        self.store.get(execution_id)
        return tuple(ref for ref in self.artifact_store.list_for_execution(execution_id)
                     if ref.producer_operation_key is None or
                     (operation := self.store.get_operation(execution_id,
                         ref.producer_operation_key)) is not None and
                     operation.state is OperationState.COMMITTED)

    def artifact(self, execution_id: str, artifact_id: str) -> bytes:
        if not any(ref.artifact_id == artifact_id for ref in self.artifacts(execution_id)):
            raise KeyError(artifact_id)
        return self.artifact_store.get(execution_id, artifact_id)

    def stream_events(self, execution_id: str, *, after_sequence: int = 0,
                      timeout: float | None = None, stop_on_terminal: bool = True,
                      page_size: int = 100):
        """Yield durable event pages as they appear; a consumer owns its cursor."""
        if (not isinstance(after_sequence, int) or after_sequence < 0 or
                not isinstance(page_size, int) or not 1 <= page_size <= 1000 or
                not isinstance(stop_on_terminal, bool) or
                timeout is not None and (isinstance(timeout, bool) or
                not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout < 0)):
            raise ValueError("Invalid stream cursor, timeout, or page size")
        self.store.get(execution_id)
        def stream():
            cursor = after_sequence
            started = monotonic()
            while True:
                page = self.store.read_events(execution_id, after=cursor, limit=page_size)
                for item in page:
                    cursor = item.sequence
                    yield item
                if page:
                    continue
                if stop_on_terminal and self.store.get(execution_id).state in TERMINAL:
                    # Terminal event insertion follows the terminal record write.
                    Event().wait(0.05)
                    if not self.store.read_events(execution_id, after=cursor, limit=1):
                        return
                    continue
                if timeout is not None and monotonic() - started >= timeout:
                    return
                Event().wait(0.05)
        return stream()

    def cancel(self, execution_id: str) -> ExecutionRecord:
        while True:
            record = self.store.get(execution_id)
            if record.state in TERMINAL or record.state is ExecutionState.CANCELLATION_REQUESTED:
                return self.get_execution(execution_id)
            desired = (ExecutionState.CANCELLED if record.state is ExecutionState.QUEUED
                       else ExecutionState.CANCELLATION_REQUESTED)
            try:
                event = _lifecycle(execution_id, "cancelled" if desired is ExecutionState.CANCELLED
                                   else "cancellation.requested")
                events = loads(record.events_json)
                updated = self.store.transition(execution_id, record.version, desired,
                                                events_json=dumps([*events, event]))
                self.store.append_event(execution_id, event)
            except ExecutionConflict:
                continue
            signal = self._controls.get(execution_id)
            if signal is not None:
                signal.set()
            return self.get_execution(execution_id)

    def result(self, execution_id: str, *, timeout: float | None = None) -> FinalResult:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        started = monotonic()
        while True:
            record = self.store.get(execution_id)
            if record.state is ExecutionState.COMPLETED:
                result = loads(record.result_json, max_bytes=self.max_record_bytes)
                if not isinstance(result, FinalResult):
                    raise ExecutionDataError("Malformed durable FinalResult")
                return result
            if record.state is ExecutionState.FAILED:
                raise ExecutionFailed(record.failure_type or "Execution failed")
            if record.state is ExecutionState.CANCELLED:
                raise ExecutionCancelled("Execution cancelled")
            if timeout is not None and monotonic() - started >= timeout:
                raise TimeoutError("Execution has not finished")
            Event().wait(0.02)

    def recover(self, execution_id: str, tree: AgentTree) -> ExecutionHandle:
        record = self.store.get(execution_id)
        def blocked(message: str, code: str,
                    operation_key: str | None = None) -> None:
            self.store.append_event(execution_id, ExecutionEvent(
                task_id=execution_id, event_type="execution.recovery.blocked",
                metadata={"code": code, "operation_key": operation_key}))
            raise ExecutionRecoveryBlocked(message, code=code,
                                           operation_key=operation_key)
        if record.state in TERMINAL:
            return self.handle(execution_id)
        if tree_fingerprint(tree) != record.fingerprint:
            blocked("Tree definition fingerprint differs", "FINGERPRINT_MISMATCH")
        if record.state is ExecutionState.CANCELLATION_REQUESTED:
            while True:
                try:
                    self.store.transition(execution_id, record.version, ExecutionState.CANCELLED)
                    break
                except ExecutionConflict:
                    record = self.store.get(execution_id)
                    if record.state in TERMINAL:
                        break
            return self.handle(execution_id)
        if record.state is ExecutionState.RUNNING:
            if record.owner_id != self.owner_id and _pid_alive(record.owner_pid):
                blocked("Execution is owned by a live local process", "LIVE_OWNER")
            if record.in_flight is not None and not record.in_flight.startswith("phase:"):
                blocked("In-flight operation has no journal identity", "UNJOURNALED_WORK")
            for operation in self.store.list_operations(execution_id):
                if operation.state is OperationState.IN_FLIGHT:
                    operation = self.store.transition_operation(execution_id,
                        operation.operation_key, operation.version, OperationState.UNCERTAIN)
                if (operation.state is OperationState.UNCERTAIN and
                        operation.replay_policy not in ("retry", "pure", "idempotent", "reconcilable")):
                    blocked(f"Uncertain operation requires reconciliation: {operation.operation_key}",
                            "UNCERTAIN_TOOL_OUTCOME" if operation.operation_type == "tool.call"
                            else "UNCERTAIN_OPERATION", operation.operation_key)
            checkpoint = self.store.load_checkpoint(execution_id)
            if ((checkpoint is None and record.checkpoint_sequence != 0) or
                    (checkpoint is not None and checkpoint.sequence != record.checkpoint_sequence)):
                blocked("No valid completed checkpoint", "CHECKPOINT_MISSING")
            state = loads(checkpoint.state_json, max_bytes=self.max_record_bytes) if checkpoint else None
            if state is not None and (not isinstance(state, WorkflowState) or state.task_id != execution_id):
                blocked("Checkpoint state is invalid", "CHECKPOINT_INVALID")
            if state is not None and state.current_phase.value in ("completed", "failed"):
                result = state.final_result
                if not isinstance(result, FinalResult):
                    blocked("Terminal checkpoint has no FinalResult", "RESULT_MISSING")
                tool = loads(checkpoint.tool_json, max_bytes=self.max_record_bytes)
                collaboration = loads(checkpoint.collaboration_json, max_bytes=self.max_record_bytes)
                trace = ExecutionTrace(task_id=execution_id)
                for event in sorted((*result.trace.events, *tool["events"],
                                     *collaboration["events"]), key=lambda item: item.timestamp):
                    trace.append(event)
                metrics = dict(result.metadata)
                if tool["events"]:
                    metrics["tool_metrics"] = tool["metrics"]
                if collaboration["events"]:
                    metrics["collaboration_metrics"] = collaboration["metrics"]
                runtime_events = loads(checkpoint.runtime_json)
                runtime_events.extend((_lifecycle(execution_id, "recovery.started"),
                                       _lifecycle(execution_id, "recovery.completed")))
                result = _with_runtime_trace(replace(result, trace=trace, metadata=metrics),
                                             runtime_events)
                terminal = ExecutionState.COMPLETED if result.success else ExecutionState.FAILED
                durable_events = []
                if result.final_output is not None:
                    durable_events.append(ExecutionEvent(
                        task_id=execution_id, event_type="output.final.available",
                        metadata={"sha256": sha256(result.final_output.encode()).hexdigest(),
                                  "size_bytes": len(result.final_output.encode())}))
                durable_events.append(_lifecycle(execution_id,
                    "completed" if result.success else "failed"))
                self.store.transition(execution_id, record.version, terminal,
                                      result_json=dumps(result, max_bytes=self.max_record_bytes),
                                      failure_type=None if result.success else "FinalReviewFailed",
                                      events_json=dumps(runtime_events),
                                      durable_events=tuple(durable_events))
                return self.handle(execution_id)
        with self._lock:
            if self._closed or not self._capacity.acquire(blocking=False):
                raise ExecutionCapacityError("Execution queue capacity exceeded")
            try:
                if record.state is ExecutionState.RUNNING:
                    self.store.update(execution_id, record.version,
                                      owner_pid=os.getpid(), owner_id=self.owner_id)
                signal = Event()
                self._controls[execution_id] = signal
                from agenttree.core.live_output import LiveOutputHub
                self._output_hubs[execution_id] = LiveOutputHub(execution_id)
                self._queue.put_nowait((execution_id, tree, signal))
            except Exception:
                self._capacity.release()
                raise
        return self.handle(execution_id)

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for _ in self._workers:
                self._queue.put_nowait(None)
        if wait:
            for worker in self._workers:
                worker.join()

    def _worker(self) -> None:
        while True:
            work = self._queue.get()
            if work is None:
                self._queue.task_done()
                return
            execution_id, tree, signal = work
            try:
                self._execute(execution_id, tree, signal)
            finally:
                with self._lock:
                    self._controls.pop(execution_id, None)
                self._capacity.release()
                self._queue.task_done()

    def _execute(self, execution_id: str, tree: AgentTree, signal: Event) -> None:
        record = self.store.get(execution_id)
        if record.state is ExecutionState.CANCELLED:
            return
        watchdog_done = Event()
        try:
            task = loads(record.input_json, max_bytes=self.max_record_bytes)
            if not isinstance(task, Task) or task.id != execution_id:
                raise ExecutionDataError("Durable input is invalid")
            if record.state is ExecutionState.QUEUED:
                record = self.store.transition(execution_id, record.version,
                                               ExecutionState.RUNNING, owner_pid=os.getpid(),
                                               owner_id=self.owner_id)
                self.store.append_event(execution_id, _lifecycle(execution_id, "started", task.execution_mode.value))
            if record.state is not ExecutionState.RUNNING:
                return
            runtime_events = loads(record.events_json)
            if record.checkpoint_sequence:
                runtime_events.append(_lifecycle(execution_id, "recovery.started"))
                runtime_events.append(_lifecycle(execution_id, "recovery.completed"))
            else:
                runtime_events.append(_lifecycle(execution_id, "started", task.execution_mode.value))
            def watch() -> None:
                while not watchdog_done.wait(0.02):
                    current = self.store.get(execution_id)
                    if current.state in TERMINAL:
                        return
                    if current.state is ExecutionState.CANCELLATION_REQUESTED:
                        signal.set()
                    if current.deadline_at is not None and utc_now() >= current.deadline_at:
                        signal.set()
                    if signal.is_set():
                        self._finish_cancelled(execution_id)
                        return
            Thread(target=watch, daemon=True, name="agenttree-execution-watchdog").start()
            checkpoint = self.store.load_checkpoint(execution_id)
            resumed = checkpoint is not None
            control = ExecutionControl(signal, record.deadline_at)
            journal = OperationJournal(self.store, execution_id)
            if resumed:
                resume_state = loads(checkpoint.state_json, max_bytes=self.max_record_bytes)
                usage_state = loads(checkpoint.usage_json, max_bytes=self.max_record_bytes)
                tool_state = loads(checkpoint.tool_json, max_bytes=self.max_record_bytes)
                collaboration_state = loads(checkpoint.collaboration_json, max_bytes=self.max_record_bytes)
            else:
                resume_state = usage_state = tool_state = collaboration_state = None

            def phase_started(name: str) -> None:
                current = self.store.get(execution_id)
                control.check()
                journal.start_phase(name)
                self.store.update(execution_id, current.version, in_flight=f"phase:{name}")
                self.store.append_event(execution_id, ExecutionEvent(
                    task_id=execution_id, event_type="execution.phase.started",
                    metadata={"phase": name}))

            def checkpoint_phase(state, usage, tool, collaboration) -> None:
                control.check()
                current = self.store.get(execution_id)
                runtime_events.append(_lifecycle(execution_id, "checkpoint.saved"))
                saved = ExecutionCheckpoint(
                    schema_version=1, execution_id=execution_id,
                    sequence=current.checkpoint_sequence + 1, created_at=utc_now(),
                    phase=state.current_phase.value,
                    state_json=dumps(state, max_bytes=self.max_record_bytes),
                    usage_json=dumps({"calls": usage.calls}, max_bytes=self.max_record_bytes),
                    tool_json=dumps({"calls": dict(tool.calls), "metrics": dict(tool.metrics),
                                     "events": tuple(tool.events)}, max_bytes=self.max_record_bytes),
                    collaboration_json=dumps(collaboration.snapshot(), max_bytes=self.max_record_bytes),
                    runtime_json=dumps(runtime_events, max_bytes=self.max_record_bytes),
                )
                self.store.checkpoint(execution_id, current.version, saved)
                self.store.append_event(execution_id, ExecutionEvent(
                    task_id=execution_id, event_type="execution.checkpoint.saved",
                    metadata={"phase": state.current_phase.value,
                              "sequence": saved.sequence}))

            from agenttree.core.live_output import _active_output_sink
            with self._lock:
                hub = self._output_hubs.get(execution_id)
                if hub is None:
                    from agenttree.core.live_output import LiveOutputHub
                    hub = LiveOutputHub(execution_id)
                    self._output_hubs[execution_id] = hub
            output_token = _active_output_sink.set(hub.publish)
            journal_token = _active_operation_journal.set(journal)
            try:
                result = tree.run(task, _resume_state=resume_state,
                                  _checkpoint=checkpoint_phase, _phase_started=phase_started,
                                  _control=control, _usage_state=usage_state,
                                  _tool_state=tool_state, _collaboration_state=collaboration_state,
                                  _artifact_store=self.artifact_store,
                                  _execution_store=self.store)
            finally:
                _active_operation_journal.reset(journal_token)
                _active_output_sink.reset(output_token)
            control.check()
            result = _with_runtime_trace(result, runtime_events)
            current = self.store.get(execution_id)
            if current.state is ExecutionState.RUNNING:
                terminal = ExecutionState.COMPLETED if result.success else ExecutionState.FAILED
                durable_events = []
                if result.final_output is not None:
                    from hashlib import sha256
                    durable_events.append(ExecutionEvent(
                        task_id=execution_id, event_type="output.final.available",
                        metadata={"sha256": sha256(result.final_output.encode()).hexdigest(),
                                  "size_bytes": len(result.final_output.encode())}))
                durable_events.append(_lifecycle(execution_id,
                    "completed" if result.success else "failed"))
                self.store.transition(execution_id, current.version, terminal,
                                      result_json=dumps(result, max_bytes=self.max_record_bytes),
                                      failure_type=None if result.success else "FinalReviewFailed",
                                      durable_events=tuple(durable_events))
        except ExecutionCancelled:
            signal.set()
            self._finish_cancelled(execution_id)
        except ExecutionRecoveryBlocked as error:
            # Keep the execution recoverable; a host can reconcile and retry.
            self.store.append_event(execution_id, ExecutionEvent(
                task_id=execution_id, event_type="execution.recovery.blocked",
                metadata={"code": error.code, "operation_key": error.operation_key}))
        except Exception as error:
            current = self.store.get(execution_id)
            if current.state is ExecutionState.CANCELLATION_REQUESTED:
                self._finish_cancelled(execution_id)
            elif current.state is ExecutionState.RUNNING:
                try:
                    self.store.transition(execution_id, current.version, ExecutionState.FAILED,
                                          failure_type=type(error).__name__,
                                          events_json=dumps([*loads(current.events_json),
                                                             _lifecycle(execution_id, "failed")]))
                    self.store.append_event(execution_id, _lifecycle(execution_id, "failed"))
                except ExecutionConflict:
                    pass
        finally:
            watchdog_done.set()
            with self._lock:
                hub = self._output_hubs.pop(execution_id, None)
            if hub is not None:
                hub.close()

    def _finish_cancelled(self, execution_id: str) -> None:
        while True:
            record = self.store.get(execution_id)
            if record.state in TERMINAL:
                return
            try:
                if record.state is ExecutionState.RUNNING:
                    self.store.transition(execution_id, record.version,
                                          ExecutionState.CANCELLATION_REQUESTED,
                                          events_json=dumps([*loads(record.events_json),
                                                             _lifecycle(execution_id, "cancellation.requested")]))
                    self.store.append_event(execution_id,
                                            _lifecycle(execution_id, "cancellation.requested"))
                else:
                    self.store.transition(execution_id, record.version, ExecutionState.CANCELLED,
                                          events_json=dumps([*loads(record.events_json),
                                                             _lifecycle(execution_id, "cancelled")]))
                    self.store.append_event(execution_id, _lifecycle(execution_id, "cancelled"))
            except ExecutionConflict:
                continue
