"""Normalized transport-independent MCP data contracts."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MCPToolDefinition:
    """A tool definition discovered from a configured MCP client."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        if not isinstance(self.input_schema, dict):
            raise TypeError("input_schema must be a dict")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "input_schema", deepcopy(self.input_schema))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


@dataclass(frozen=True)
class MCPCallResult:
    """A normalized MCP invocation outcome without external SDK types."""

    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_result: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        object.__setattr__(self, "output", deepcopy(self.output))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))
        object.__setattr__(self, "raw_result", deepcopy(self.raw_result))


@dataclass(frozen=True)
class MCPToolCall:
    """One recorded MCP tool call used by deterministic client implementations."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.arguments, dict):
            raise TypeError("arguments must be a dict")
        object.__setattr__(self, "arguments", deepcopy(self.arguments))
