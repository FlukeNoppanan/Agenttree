"""Structured output produced by task triage."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TriageResult:
    """A domain-independent interpretation of a task's required work.

    Capability labels retain their configured spelling and order. Consumers
    may use registries that normalize labels for lookup.
    """

    task_id: str
    objective: str
    required_capabilities: tuple[str, ...] = ()
    category: str | None = None
    confidence: float | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
