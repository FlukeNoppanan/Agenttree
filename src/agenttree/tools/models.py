"""Provider-independent contracts for tool inputs and results."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolParameter:
    """A normalized Python callable parameter description."""

    name: str
    required: bool
    annotation: str | None = None
    default: str | None = None
    kind: str = "positional_or_keyword"
    description: str = ""


@dataclass(frozen=True)
class ToolInputSpec:
    """An ordered parameter snapshot for explicit tool invocation."""

    parameters: tuple[ToolParameter, ...] = ()
    accepts_var_keyword: bool = False


@dataclass(frozen=True)
class ToolResult:
    """A normalized tool outcome independent of agents and providers."""

    tool_id: str
    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    duration_ms: float | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_id, str) or not self.tool_id.strip():
            raise ValueError("tool_id must be a non-empty string")
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        object.__setattr__(self, "output", deepcopy(self.output))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


@dataclass(frozen=True)
class ToolCall:
    """One model-requested call with correlation and execution identity."""

    call_id: str
    tool_name: str
    agent_id: str
    agent_role: str
    arguments: Any
    execution_id: str
