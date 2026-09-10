"""Agent output and review contracts."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class AgentResult:
    """An agent outcome with arbitrary output and optional error details."""

    agent_id: str
    success: bool
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ReviewDecision(str, Enum):
    """Available outcomes of a review."""

    PASS = "pass"
    REVISE = "revise"
    FAIL = "fail"


@dataclass
class ReviewResult:
    """A reviewer's decision, feedback, and caller-defined metadata."""

    decision: ReviewDecision
    reviewer_id: str
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
