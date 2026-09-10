"""Explicit MCP tool discovery, adaptation, and optional registration."""

from agenttree.tools.mcp.base import BaseMCPClient
from agenttree.tools.mcp.models import MCPToolDefinition
from agenttree.tools.mcp.tool import MCPTool
from agenttree.tools.registry import ToolRegistry


class MCPToolLoader:
    """Import tools from one configured MCP client in discovery order."""

    def __init__(
        self,
        *,
        client: BaseMCPClient,
        namespace: str | None = None,
    ) -> None:
        if not isinstance(client, BaseMCPClient):
            raise TypeError("client must be a BaseMCPClient")
        if namespace is not None and (
            not isinstance(namespace, str) or not namespace.strip()
        ):
            raise ValueError("namespace must be a non-empty string or None")
        self._client = client
        self._namespace = namespace.strip() if namespace is not None else None

    @property
    def client(self) -> BaseMCPClient:
        """Return the developer-configured MCP client."""
        return self._client

    @property
    def namespace(self) -> str | None:
        """Return the optional display-name namespace."""
        return self._namespace

    def discover(self) -> tuple[MCPTool, ...]:
        """Discover and adapt tools without mutating a registry."""
        definitions = self._client.list_tools()
        if not isinstance(definitions, tuple) or any(
            not isinstance(item, MCPToolDefinition) for item in definitions
        ):
            raise TypeError(
                "MCP client list_tools must return MCPToolDefinition tuple",
            )
        return tuple(
            MCPTool(
                client=self._client,
                definition=definition,
                name=(
                    f"{self._namespace}.{definition.name}"
                    if self._namespace is not None
                    else definition.name
                ),
                tool_id=(
                    f"mcp:{self._client.server_id}:"
                    f"{self._namespace}.{definition.name}"
                    if self._namespace is not None
                    else None
                ),
            )
            for definition in definitions
        )

    def load(self, registry: ToolRegistry | None = None) -> tuple[MCPTool, ...]:
        """Discover tools and atomically preflight optional registration."""
        if registry is not None and not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry or None")
        tools = self.discover()
        if registry is None:
            return tools
        existing_ids = set(registry.ids)
        existing_names = set(registry.names)
        batch_ids: set[str] = set()
        batch_names: set[str] = set()
        for tool in tools:
            name_key = tool.name.strip().casefold()
            if tool.id in existing_ids or tool.id in batch_ids:
                raise ValueError(f"Tool ID collision: {tool.id}")
            if name_key in existing_names or name_key in batch_names:
                raise ValueError(f"Tool name collision: {tool.name}")
            batch_ids.add(tool.id)
            batch_names.add(name_key)
        for tool in tools:
            registry.register(tool)
        return tools
