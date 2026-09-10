"""Task identity, context, and lifecycle contracts."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class TaskStatus(str, Enum):
    """Lifecycle labels; transitions are not enforced by the data model."""

    PENDING = "pending"
    TRIAGE = "triage"
    PLANNING = "planning"
    RUNNING = "running"
    REVIEW = "review"
    REVISION = "revision"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskContext:
    """Caller-defined contextual values, including nested structured data."""

    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A normalized unit of work, independent of its source or domain."""

    objective: str
    id: str = field(default_factory=lambda: str(uuid4()))
    context: TaskContext = field(default_factory=TaskContext)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
