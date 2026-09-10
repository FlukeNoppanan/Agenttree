"""Transport-independent MCP client interface."""

from abc import ABC, abstractmethod
from typing import Any, Mapping

from agenttree.tools.mcp.models import MCPCallResult, MCPToolDefinition


class BaseMCPClient(ABC):
    """Discover and invoke tools through a developer-configured MCP endpoint."""

    def __init__(self, *, server_id: str) -> None:
        if not isinstance(server_id, str) or not server_id.strip():
            raise ValueError("server_id must be a non-empty string")
        self._server_id = server_id.strip()

    @property
    def server_id(self) -> str:
        """Return the stable logical identity of the configured MCP source."""
        return self._server_id

    @abstractmethod
    def connect(self) -> None:
        """Open transport resources when required by the implementation."""
        raise NotImplementedError

    @abstractmethod
    def list_tools(self) -> tuple[MCPToolDefinition, ...]:
        """Return normalized definitions in server discovery order."""
        raise NotImplementedError

    @abstractmethod
    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> MCPCallResult:
        """Invoke a known MCP tool and normalize its runtime outcome."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release transport resources when applicable."""
        raise NotImplementedError
