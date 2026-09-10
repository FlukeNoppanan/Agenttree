"""Deterministic in-memory MCP client for tests and local development."""

from copy import deepcopy
from typing import Any, Iterable, Mapping

from agenttree.tools.mcp.base import BaseMCPClient
from agenttree.tools.mcp.models import MCPCallResult, MCPToolCall, MCPToolDefinition


class MockMCPClient(BaseMCPClient):
    """Discover configured tools and return configured results without a network."""

    def __init__(
        self,
        *,
        server_id: str = "mock",
        tools: Iterable[MCPToolDefinition] = (),
        responses: Mapping[str, MCPCallResult] | None = None,
    ) -> None:
        super().__init__(server_id=server_id)
        prepared_tools = tuple(tools)
        if any(not isinstance(item, MCPToolDefinition) for item in prepared_tools):
            raise TypeError("tools must contain MCPToolDefinition instances")
        names = tuple(item.name for item in prepared_tools)
        if len(set(names)) != len(names):
            raise ValueError("MCP tool definition names must be unique")
        prepared_responses = dict(responses or {})
        if any(
            not isinstance(name, str) or not isinstance(result, MCPCallResult)
            for name, result in prepared_responses.items()
        ):
            raise TypeError("responses must map tool names to MCPCallResult values")
        unknown_responses = set(prepared_responses).difference(names)
        if unknown_responses:
            raise ValueError("responses contain unknown MCP tool names")
        self._tools = deepcopy(prepared_tools)
        self._responses = deepcopy(prepared_responses)
        self._calls: list[MCPToolCall] = []
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Return the deterministic mock session state."""
        return self._connected

    @property
    def calls(self) -> tuple[MCPToolCall, ...]:
        """Return isolated recorded calls in invocation order."""
        return deepcopy(tuple(self._calls))

    def connect(self) -> None:
        """Mark the in-memory mock session as connected."""
        self._connected = True

    def list_tools(self) -> tuple[MCPToolDefinition, ...]:
        """Return isolated configured definitions in deterministic order."""
        return deepcopy(self._tools)

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> MCPCallResult:
        """Record a known tool call and return its configured normalized result."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be a mapping")
        definitions = {item.name: item for item in self._tools}
        if name not in definitions:
            raise KeyError(f"Unknown MCP tool: {name}")
        prepared_arguments = deepcopy(dict(arguments))
        self._calls.append(MCPToolCall(name=name, arguments=prepared_arguments))
        configured = self._responses.get(name)
        if configured is not None:
            return deepcopy(configured)
        return MCPCallResult(
            success=True,
            output={"tool": name, "arguments": prepared_arguments},
            metadata={"mock": True},
        )

    def close(self) -> None:
        """Mark the in-memory mock session as disconnected."""
        self._connected = False
