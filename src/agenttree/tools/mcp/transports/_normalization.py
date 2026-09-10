"""Convert SDK models to deterministic, isolated framework values."""

from copy import deepcopy
from typing import Any

from agenttree.tools.mcp.models import MCPCallResult, MCPToolDefinition


def plain(value: Any) -> dict[str, Any]:
    """Dump SDK models using protocol aliases without retaining SDK objects."""
    result = value if isinstance(value, dict) else value.model_dump(
        mode="json", by_alias=True, exclude_none=True,
    )
    if not isinstance(result, dict):
        raise TypeError("Expected an MCP object")
    return deepcopy(result)


def definition(value: Any) -> MCPToolDefinition:
    data = plain(value)
    return MCPToolDefinition(
        name=data["name"], description=data.get("description") or "",
        input_schema=data["inputSchema"],
        metadata={key: item for key, item in data.items()
                  if key not in ("name", "description", "inputSchema")},
    )


def call_result(value: Any, *, server_id: str, transport: str) -> MCPCallResult:
    data = plain(value)
    content = data.get("content", [])
    failed = data.get("isError", False)
    if not isinstance(content, list) or not isinstance(failed, bool):
        raise TypeError("Invalid MCP call result")
    # Preserve both structured output and every content block, in server order.
    return MCPCallResult(
        success=not failed,
        output={"structured_content": data.get("structuredContent"), "content": content},
        error="MCP server reported a tool error" if failed else None,
        metadata={"server_id": server_id, "transport": transport},
        raw_result=data,
    )
