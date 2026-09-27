"""MCP definition normalization and BaseTool adapter."""

from copy import deepcopy
from typing import Any, Mapping

from agenttree.tools.base import BaseTool, ToolRecoveryPolicy
from agenttree.tools.mcp.base import BaseMCPClient
from agenttree.tools.mcp.models import MCPCallResult, MCPToolDefinition
from agenttree.tools.models import ToolInputSpec, ToolParameter, ToolResult


_JSON_PRIMITIVE_TYPES = {
    "array", "boolean", "integer", "null", "number", "object", "string",
}


def _schema_default(value: object) -> str:
    if isinstance(value, str):
        return repr(value)
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, (list, dict)):
        return repr(value)
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def normalize_mcp_input_schema(schema: Mapping[str, Any]) -> ToolInputSpec:
    """Conservatively map a common object schema into ``ToolInputSpec``."""
    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a mapping")
    if schema.get("type") not in (None, "object"):
        return ToolInputSpec()
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return ToolInputSpec()
    required_value = schema.get("required", ())
    required = (
        {item for item in required_value if isinstance(item, str)}
        if isinstance(required_value, (list, tuple, set, frozenset))
        else set()
    )
    parameters: list[ToolParameter] = []
    for name, definition in properties.items():
        if not isinstance(name, str) or not name.strip():
            continue
        property_schema = definition if isinstance(definition, Mapping) else {}
        schema_type = property_schema.get("type")
        annotation = (
            schema_type
            if isinstance(schema_type, str) and schema_type in _JSON_PRIMITIVE_TYPES
            else None
        )
        description = property_schema.get("description", "")
        if not isinstance(description, str):
            description = ""
        default = (
            _schema_default(property_schema["default"])
            if "default" in property_schema
            else None
        )
        parameters.append(ToolParameter(
            name=name.strip(),
            required=name in required,
            annotation=annotation,
            default=default,
            kind="keyword_only",
            description=description,
        ))
    return ToolInputSpec(
        parameters=tuple(parameters),
        accepts_var_keyword=schema.get("additionalProperties") is True,
    )


class MCPTool(BaseTool):
    """Adapt one normalized MCP tool definition to the common tool interface."""

    def __init__(
        self,
        *,
        client: BaseMCPClient,
        definition: MCPToolDefinition,
        name: str | None = None,
        tool_id: str | None = None,
        enabled: bool = True,
        recovery_policy: ToolRecoveryPolicy = ToolRecoveryPolicy.UNKNOWN,
    ) -> None:
        if not isinstance(client, BaseMCPClient):
            raise TypeError("client must be a BaseMCPClient")
        if not isinstance(definition, MCPToolDefinition):
            raise TypeError("definition must be an MCPToolDefinition")
        self._client = client
        self._definition = deepcopy(definition)
        adapted_name = definition.name if name is None else name
        stable_id = (
            f"mcp:{client.server_id}:{definition.name}"
            if tool_id is None
            else tool_id
        )
        super().__init__(
            name=adapted_name,
            description=definition.description,
            tool_id=stable_id,
            input_spec=normalize_mcp_input_schema(definition.input_schema),
            metadata={
                "mcp": {
                    "server_id": client.server_id,
                    "tool_name": definition.name,
                    "input_schema": deepcopy(definition.input_schema),
                    "definition_metadata": deepcopy(definition.metadata),
                },
            },
            enabled=enabled,
            recovery_policy=recovery_policy,
        )

    @property
    def client(self) -> BaseMCPClient:
        """Return the configured MCP client boundary."""
        return self._client

    @property
    def definition(self) -> MCPToolDefinition:
        """Return an isolated normalized definition snapshot."""
        return deepcopy(self._definition)

    def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Call the MCP client and convert its normalized result."""
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be a mapping")
        try:
            result = self._client.call_tool(
                self._definition.name, deepcopy(dict(arguments)),
            )
        except (KeyError, TypeError, ValueError):
            raise
        except Exception as error:
            return ToolResult(
                tool_id=self.id,
                success=False,
                error=f"{type(error).__name__}: {error}",
                metadata={
                    "mcp": {
                        "server_id": self._client.server_id,
                        "tool_name": self._definition.name,
                        "error_type": type(error).__name__,
                    },
                },
            )
        if not isinstance(result, MCPCallResult):
            raise TypeError("MCP client call_tool must return an MCPCallResult")
        return ToolResult(
            tool_id=self.id,
            success=result.success,
            output=result.output,
            error=result.error,
            metadata={
                "mcp": {
                    "server_id": self._client.server_id,
                    "tool_name": self._definition.name,
                    "call_metadata": deepcopy(result.metadata),
                    "raw_result": deepcopy(result.raw_result),
                },
            },
        )
