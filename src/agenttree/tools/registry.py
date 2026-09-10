"""Deterministic storage and lookup for registered tools."""

from types import MappingProxyType
from typing import Mapping

from agenttree.tools.base import BaseTool


def _tool_name_key(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Tool names must be strings")
    key = name.strip().casefold()
    if not key:
        raise ValueError("Tool names cannot be empty")
    return key


def _tool_id_key(tool_id: str) -> str:
    if not isinstance(tool_id, str):
        raise TypeError("Tool IDs must be strings")
    key = tool_id.strip()
    if not key:
        raise ValueError("Tool IDs cannot be empty")
    return key


class ToolRegistry:
    """Register tools by unique ID and normalized unique name in order."""

    def __init__(self) -> None:
        self._by_id: dict[str, BaseTool] = {}
        self._id_by_name: dict[str, str] = {}

    @property
    def tools(self) -> tuple[BaseTool, ...]:
        """Return an insertion-ordered tool snapshot."""
        return tuple(self._by_id.values())

    @property
    def ids(self) -> tuple[str, ...]:
        """Return tool IDs in registration order."""
        return tuple(self._by_id)

    @property
    def names(self) -> tuple[str, ...]:
        """Return normalized names in registration order."""
        return tuple(self._id_by_name)

    @property
    def tool_map(self) -> Mapping[str, BaseTool]:
        """Return a read-only snapshot keyed by tool ID."""
        return MappingProxyType(dict(self._by_id))

    def register(self, tool: BaseTool) -> None:
        """Register a tool, rejecting duplicate IDs and normalized names."""
        if not isinstance(tool, BaseTool):
            raise TypeError("Only BaseTool instances can be registered")
        tool_id = _tool_id_key(tool.id)
        name_key = _tool_name_key(tool.name)
        if tool_id in self._by_id:
            raise ValueError(f"Tool ID already registered: {tool.id}")
        if name_key in self._id_by_name:
            raise ValueError(f"Tool name already registered: {tool.name}")
        self._by_id[tool_id] = tool
        self._id_by_name[name_key] = tool_id

    def get_by_id(self, tool_id: str) -> BaseTool:
        """Return a tool by ID, raising ``KeyError`` if absent."""
        return self._by_id[_tool_id_key(tool_id)]

    def get(self, tool_id: str) -> BaseTool:
        """Return a tool by ID; alias retained for registry consistency."""
        return self.get_by_id(tool_id)

    def get_by_name(self, name: str) -> BaseTool:
        """Return a tool by normalized name, raising ``KeyError`` if absent."""
        return self._by_id[self._id_by_name[_tool_name_key(name)]]

    def unregister(self, tool_id: str) -> BaseTool:
        """Remove and return a tool by ID while cleaning its name index."""
        key = _tool_id_key(tool_id)
        tool = self._by_id.pop(key)
        del self._id_by_name[_tool_name_key(tool.name)]
        return tool
