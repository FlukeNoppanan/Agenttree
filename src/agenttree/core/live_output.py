"""Bounded, process-local delivery of human-readable agent text deltas."""

from __future__ import annotations

from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from threading import Condition
from time import monotonic
from typing import Callable

from agenttree.core.execution_store import utc_now


@dataclass(frozen=True)
class AgentOutputDelta:
    execution_id: str
    role: str
    agent_id: str
    operation_id: str | None
    attempt: int | None
    sequence: int
    delta: str
    timestamp: datetime
    dropped_before: int = 0


_active_output_sink: ContextVar[Callable[[str, str, str, str | None, int | None], None] | None] = ContextVar(
    "agenttree_output_sink", default=None)


def emit_output_delta(role: str, agent_id: str, delta: str) -> None:
    """Emit only semantic text; internal structured decisions never call here."""
    sink = _active_output_sink.get()
    if sink is None or not delta:
        return
    from agenttree.core.execution_control import check_execution
    check_execution()
    from agenttree.core.operation_journal import _active_operation_key, current_journal
    journal = current_journal()
    key = _active_operation_key.get() if journal is not None else None
    record = journal.store.get_operation(journal.execution_id, key) if key is not None else None
    sink(role, agent_id, delta, key, record.attempt if record is not None else None)


class _Subscriber:
    def __init__(self, capacity: int) -> None:
        self.items: deque[AgentOutputDelta] = deque(maxlen=capacity)
        self.dropped = 0


class LiveOutputHub:
    """Independent bounded subscriber queues; late subscribers see future deltas."""

    def __init__(self, execution_id: str, *, capacity: int = 64,
                 max_subscribers: int = 32) -> None:
        self.execution_id = execution_id
        self.capacity = capacity
        self.max_subscribers = max_subscribers
        self._condition = Condition()
        self._subscribers: set[_Subscriber] = set()
        self._sequence = 0
        self._closed = False

    def subscribe(self) -> _Subscriber:
        with self._condition:
            if len(self._subscribers) >= self.max_subscribers:
                raise RuntimeError("Live output subscriber limit exceeded")
            subscriber = _Subscriber(self.capacity)
            self._subscribers.add(subscriber)
            return subscriber

    def publish(self, role: str, agent_id: str, delta: str,
                operation_id: str | None, attempt: int | None) -> None:
        with self._condition:
            if self._closed:
                return
            self._sequence += 1
            item = AgentOutputDelta(self.execution_id, role, agent_id,
                                    operation_id, attempt, self._sequence,
                                    delta, utc_now())
            for subscriber in self._subscribers:
                if len(subscriber.items) == self.capacity:
                    subscriber.items.popleft()
                    subscriber.dropped += 1
                subscriber.items.append(item)
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def stream(self, subscriber: _Subscriber, timeout: float | None):
        deadline = None if timeout is None else monotonic() + timeout
        try:
            while True:
                with self._condition:
                    while not subscriber.items and not self._closed:
                        remaining = None if deadline is None else deadline - monotonic()
                        if remaining is not None and remaining <= 0:
                            return
                        self._condition.wait(remaining)
                    if not subscriber.items:
                        return
                    item = subscriber.items.popleft()
                    dropped = subscriber.dropped
                    subscriber.dropped = 0
                if dropped:
                    from dataclasses import replace
                    item = replace(item, dropped_before=dropped)
                yield item
        finally:
            with self._condition:
                self._subscribers.discard(subscriber)
