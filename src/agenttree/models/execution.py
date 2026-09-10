"""Execution event and per-task trace contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class ExecutionEvent:
    """A generic activity record with a UTC timestamp by default.

    Fields cannot be reassigned. Metadata is copied at construction, and trace
    snapshots copy it again so callers cannot mutate a trace indirectly.
    """

    event_type: str
    task_id: str
    actor_id: str | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        event_type = (
            self.event_type.value
            if isinstance(self.event_type, Enum)
            else self.event_type
        )
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if self.actor_id is not None and not isinstance(self.actor_id, str):
            raise TypeError("actor_id must be a string or None")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "metadata", deepcopy(self.metadata))
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, standard-library-friendly snapshot."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "task_id": self.task_id,
            "actor_id": self.actor_id,
            "message": self.message,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionTrace:
    """A single task's events, exposed as a tuple in insertion order.

    Use append() to add events; timestamps do not affect their ordering.
    """

    task_id: str
    _events: list[ExecutionEvent] = field(default_factory=list, init=False, repr=False)
    _read_only: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")

    @property
    def events(self) -> tuple[ExecutionEvent, ...]:
        """Return a snapshot that cannot modify the stored event sequence."""
        return tuple(deepcopy(event) for event in self._events)

    @property
    def last_event(self) -> ExecutionEvent | None:
        """Return an isolated copy of the most recent event, if present."""
        return deepcopy(self._events[-1]) if self._events else None

    @property
    def event_count(self) -> int:
        """Return the number of stored events."""
        return len(self._events)

    @property
    def is_read_only(self) -> bool:
        """Return whether this trace is an immutable state snapshot."""
        return self._read_only

    def append(self, event: ExecutionEvent) -> None:
        """Append an event, raising ValueError if it belongs to another task."""
        if self._read_only:
            raise TypeError("Cannot append to a read-only ExecutionTrace")
        if not isinstance(event, ExecutionEvent):
            raise TypeError("event must be an ExecutionEvent")
        if event.task_id != self.task_id:
            raise ValueError("Event task_id must match the trace task_id")
        self._events.append(deepcopy(event))

    def events_by_type(self, event_type: str | Enum) -> tuple[ExecutionEvent, ...]:
        """Return isolated events matching a known enum or custom string type."""
        value = event_type.value if isinstance(event_type, Enum) else event_type
        if not isinstance(value, str) or not value.strip():
            raise ValueError("event_type must be a non-empty string or string enum")
        return tuple(
            deepcopy(event) for event in self._events if event.event_type == value
        )

    def events_by_actor(self, actor_id: str | None) -> tuple[ExecutionEvent, ...]:
        """Return isolated events for an actor, including ``None`` if requested."""
        if actor_id is not None and not isinstance(actor_id, str):
            raise TypeError("actor_id must be a string or None")
        return tuple(
            deepcopy(event) for event in self._events if event.actor_id == actor_id
        )

    def clone(self, *, read_only: bool = False) -> ExecutionTrace:
        """Return an independent trace, optionally sealed against appends."""
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be a bool")
        cloned = ExecutionTrace(task_id=self.task_id)
        for event in self._events:
            cloned.append(event)
        object.__setattr__(cloned, "_read_only", read_only)
        return cloned

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serialization-ready trace snapshot."""
        return {
            "task_id": self.task_id,
            "events": [event.to_dict() for event in self._events],
        }
