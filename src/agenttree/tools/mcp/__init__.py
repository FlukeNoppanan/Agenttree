"""Public normalized MCP client, adapter, mock, and discovery APIs."""

from agenttree.tools.mcp.base import BaseMCPClient
from agenttree.tools.mcp.discovery import MCPToolLoader
from agenttree.tools.mcp.mock import MockMCPClient
from agenttree.tools.mcp.models import MCPCallResult, MCPToolCall, MCPToolDefinition
from agenttree.tools.mcp.tool import MCPTool, normalize_mcp_input_schema
from agenttree.tools.mcp.transports import StdioMCPClient, StreamableHttpMCPClient
from agenttree.tools.mcp.exceptions import (
    MCPError, MCPDependencyError, MCPConfigurationError, MCPConnectionError,
    MCPRuntimeError,
)

__all__ = [
    "StdioMCPClient",
    "StreamableHttpMCPClient",
    "MCPError",
    "MCPDependencyError",
    "MCPConfigurationError",
    "MCPConnectionError",
    "MCPRuntimeError",
    "BaseMCPClient",
    "MCPCallResult",
    "MCPTool",
    "MCPToolCall",
    "MCPToolDefinition",
    "MCPToolLoader",
    "MockMCPClient",
    "normalize_mcp_input_schema",
]
