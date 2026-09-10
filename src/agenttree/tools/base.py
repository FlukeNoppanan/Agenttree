"""Common interface for explicitly registered executable tools."""

from abc import ABC, abstractmethod
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from agenttree.tools.models import ToolInputSpec, ToolResult


class BaseTool(ABC):
    """Base interface for local and future externally backed tools."""

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        tool_id: str | None = None,
        input_spec: ToolInputSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(description, str):
            raise TypeError("description must be a string")
        prepared_id = str(uuid4()) if tool_id is None else tool_id
        if not isinstance(prepared_id, str) or not prepared_id.strip():
            raise ValueError("tool_id must be a non-empty string")
        if input_spec is not None and not isinstance(input_spec, ToolInputSpec):
            raise TypeError("input_spec must be a ToolInputSpec")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        self._id = prepared_id.strip()
        self._name = name.strip()
        self._description = description
        self._input_spec = input_spec or ToolInputSpec()
        self._metadata = deepcopy(dict(metadata or {}))

    @property
    def id(self) -> str:
        """Return the stable tool identifier."""
        return self._id

    @property
    def name(self) -> str:
        """Return the display name used for normalized registry lookup."""
        return self._name

    @property
    def description(self) -> str:
        """Return the developer-provided tool description."""
        return self._description

    @property
    def input_spec(self) -> ToolInputSpec:
        """Return the immutable structured input specification."""
        return self._input_spec

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return a read-only, isolated metadata snapshot."""
        return MappingProxyType(deepcopy(self._metadata))

    @abstractmethod
    def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Invoke this tool explicitly and return a normalized result."""
        raise NotImplementedError
