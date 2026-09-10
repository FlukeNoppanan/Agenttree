"""Normalized MCP client, adapter, discovery, and tool compatibility tests."""

from typing import Any, Mapping

import pytest

from agenttree.agents import SpecialistAgent
from agenttree.models import ExecutionTrace
from agenttree.tools import (
    BaseTool,
    FunctionTool,
    ToolBindingRegistry,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)
from agenttree.tools.mcp import (
    BaseMCPClient,
    MCPCallResult,
    MCPTool,
    MCPToolDefinition,
    MCPToolLoader,
    MockMCPClient,
    normalize_mcp_input_schema,
)
from agenttree.tracing import ExecutionEventType


def _definitions() -> tuple[MCPToolDefinition, ...]:
    return (
        MCPToolDefinition(
            name="lookup",
            description="Look up a value",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Lookup query",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        MCPToolDefinition(name="status", description="Return status"),
    )


def test_base_mcp_client_is_an_abstract_transport_contract() -> None:
    with pytest.raises(TypeError):
        BaseMCPClient(server_id="base")  # type: ignore[abstract]

    class IncompleteClient(BaseMCPClient):
        pass

    with pytest.raises(TypeError):
        IncompleteClient(server_id="incomplete")


def test_mcp_models_are_normalized_and_copy_input_data() -> None:
    schema = {"properties": {"value": {"type": "integer"}}}
    definition_metadata = {"source": {"version": 1}}
    definition = MCPToolDefinition(
        name="  calculate  ",
        description="Calculate",
        input_schema=schema,
        metadata=definition_metadata,
    )
    output = {"items": [1]}
    raw = {"transport": {"id": "call-1"}}
    result = MCPCallResult(
        success=True,
        output=output,
        metadata={"duration": 1},
        raw_result=raw,
    )
    schema["properties"]["value"]["type"] = "string"
    definition_metadata["source"]["version"] = 2
    output["items"].append(2)
    raw["transport"]["id"] = "changed"
    assert definition.name == "calculate"
    assert definition.input_schema["properties"]["value"]["type"] == "integer"
    assert definition.metadata["source"]["version"] == 1
    assert result.output == {"items": [1]}
    assert result.raw_result == {"transport": {"id": "call-1"}}


def test_mock_mcp_client_discovery_and_session_state_are_deterministic() -> None:
    client = MockMCPClient(server_id="demo", tools=_definitions())
    assert client.server_id == "demo"
    assert client.is_connected is False
    client.connect()
    assert client.is_connected is True
    first = client.list_tools()
    second = client.list_tools()
    assert first == second == _definitions()
    assert first is not second
    client.close()
    assert client.is_connected is False


def test_mock_mcp_client_records_calls_in_order_as_isolated_snapshots() -> None:
    client = MockMCPClient(tools=_definitions())
    arguments = {"query": {"value": "first"}}
    first_result = client.call_tool("lookup", arguments)
    snapshot = client.calls
    arguments["query"]["value"] = "changed"
    client.call_tool("status", {})
    assert first_result.output == {
        "tool": "lookup", "arguments": {"query": {"value": "first"}},
    }
    assert snapshot[0].arguments == {"query": {"value": "first"}}
    assert tuple(item.name for item in client.calls) == ("lookup", "status")


def test_mock_mcp_client_simulates_normalized_call_failure() -> None:
    failure = MCPCallResult(success=False, error="Remote operation failed")
    client = MockMCPClient(
        tools=_definitions(), responses={"lookup": failure},
    )
    assert client.call_tool("lookup", {"query": "x"}) == failure
    with pytest.raises(KeyError, match="Unknown MCP tool"):
        client.call_tool("unknown", {})


def test_mcp_tool_is_a_base_tool_with_stable_server_identity() -> None:
    client = MockMCPClient(server_id="demo", tools=_definitions())
    tool = MCPTool(client=client, definition=_definitions()[0])
    second = MCPTool(client=client, definition=_definitions()[0])
    assert isinstance(tool, BaseTool)
    assert tool.id == second.id == "mcp:demo:lookup"
    assert tool.name == "lookup"
    assert tool.description == "Look up a value"
    assert tool.metadata["mcp"]["server_id"] == "demo"
    assert tool.metadata["mcp"]["tool_name"] == "lookup"


def test_mcp_tool_converts_success_to_tool_result() -> None:
    response = MCPCallResult(
        success=True,
        output={"value": 42},
        metadata={"duration_ms": 3},
        raw_result={"content": [{"type": "text"}]},
    )
    definition = _definitions()[0]
    client = MockMCPClient(
        server_id="demo", tools=(definition,), responses={"lookup": response},
    )
    tool = MCPTool(client=client, definition=definition)
    result = tool.invoke({"query": "answer"})
    assert result == ToolResult(
        tool_id=tool.id,
        success=True,
        output={"value": 42},
        metadata={
            "mcp": {
                "server_id": "demo",
                "tool_name": "lookup",
                "call_metadata": {"duration_ms": 3},
                "raw_result": {"content": [{"type": "text"}]},
            },
        },
    )


def test_mcp_normalized_and_exception_failures_become_failed_tool_results() -> None:
    definition = _definitions()[0]
    normalized_client = MockMCPClient(
        tools=(definition,),
        responses={"lookup": MCPCallResult(success=False, error="Unavailable")},
    )
    normalized = MCPTool(
        client=normalized_client, definition=definition,
    ).invoke({"query": "x"})
    assert normalized.success is False
    assert normalized.error == "Unavailable"

    class FailingClient(BaseMCPClient):
        def connect(self) -> None:
            return None

        def list_tools(self) -> tuple[MCPToolDefinition, ...]:
            return (definition,)

        def call_tool(
            self, name: str, arguments: Mapping[str, Any],
        ) -> MCPCallResult:
            raise ConnectionError("transport unavailable")

        def close(self) -> None:
            return None

    failing_tool = MCPTool(
        client=FailingClient(server_id="failing"), definition=definition,
    )
    failed = failing_tool.invoke({"query": "x"})
    assert failed.success is False
    assert failed.error == "ConnectionError: transport unavailable"
    assert failed.metadata["mcp"]["error_type"] == "ConnectionError"


def test_mcp_tool_keeps_configuration_errors_distinguishable() -> None:
    definition = _definitions()[0]

    class UnknownClient(BaseMCPClient):
        def connect(self) -> None:
            return None

        def list_tools(self) -> tuple[MCPToolDefinition, ...]:
            return (definition,)

        def call_tool(
            self, name: str, arguments: Mapping[str, Any],
        ) -> MCPCallResult:
            raise KeyError(name)

        def close(self) -> None:
            return None

    tool = MCPTool(client=UnknownClient(server_id="unknown"), definition=definition)
    with pytest.raises(KeyError):
        tool.invoke({"query": "x"})
    with pytest.raises(TypeError, match="mapping"):
        tool.invoke([("query", "x")])  # type: ignore[arg-type]


def test_simple_schema_normalization_captures_required_optional_and_defaults() -> None:
    spec = normalize_mcp_input_schema(_definitions()[0].input_schema)
    query, limit = spec.parameters
    assert (query.name, query.required, query.annotation) == (
        "query", True, "string",
    )
    assert query.description == "Lookup query"
    assert query.default is None
    assert (limit.name, limit.required, limit.annotation, limit.default) == (
        "limit", False, "integer", "10",
    )
    assert spec.accepts_var_keyword is False


def test_unsupported_schema_degrades_safely_and_raw_schema_is_preserved() -> None:
    schema = {
        "type": "object",
        "properties": {
            "choice": {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
            },
        },
        "additionalProperties": True,
    }
    definition = MCPToolDefinition(name="complex", input_schema=schema)
    client = MockMCPClient(tools=(definition,))
    tool = MCPTool(client=client, definition=definition)
    assert tool.input_spec.parameters[0].annotation is None
    assert tool.input_spec.accepts_var_keyword is True
    assert tool.metadata["mcp"]["input_schema"] == schema
    assert normalize_mcp_input_schema({"type": "array"}).parameters == ()


def test_loader_preserves_discovery_order_without_hidden_registration() -> None:
    client = MockMCPClient(server_id="demo", tools=_definitions())
    registry = ToolRegistry()
    tools = MCPToolLoader(client=client).discover()
    assert tuple(tool.name for tool in tools) == ("lookup", "status")
    assert tuple(tool.id for tool in tools) == (
        "mcp:demo:lookup", "mcp:demo:status",
    )
    assert registry.tools == ()


def test_loader_registers_discovered_tools_in_order() -> None:
    client = MockMCPClient(server_id="demo", tools=_definitions())
    registry = ToolRegistry()
    loaded = MCPToolLoader(client=client).load(registry)
    assert registry.tools == loaded
    assert registry.get_by_name("lookup") is loaded[0]
    assert registry.get_by_id("mcp:demo:status") is loaded[1]


def test_loader_rejects_collisions_atomically_by_default() -> None:
    registry = ToolRegistry()
    existing = FunctionTool(name="status", function=lambda: "local")
    registry.register(existing)
    before = (registry.tools, registry.ids, registry.names)
    loader = MCPToolLoader(
        client=MockMCPClient(server_id="demo", tools=_definitions()),
    )
    with pytest.raises(ValueError, match="name collision"):
        loader.load(registry)
    assert (registry.tools, registry.ids, registry.names) == before


def test_loader_namespace_avoids_name_collision_explicitly() -> None:
    registry = ToolRegistry()
    local = FunctionTool(name="status", function=lambda: "local")
    registry.register(local)
    loaded = MCPToolLoader(
        client=MockMCPClient(server_id="demo", tools=_definitions()),
        namespace="demo",
    ).load(registry)
    assert tuple(tool.name for tool in loaded) == ("demo.lookup", "demo.status")
    assert tuple(tool.id for tool in loaded) == (
        "mcp:demo:demo.lookup", "mcp:demo:demo.status",
    )
    assert registry.get_by_name("status") is local
    assert registry.get_by_name("DEMO.STATUS") is loaded[1]


def test_function_and_mcp_tools_coexist_in_one_registry() -> None:
    definition = _definitions()[0]
    client = MockMCPClient(server_id="demo", tools=(definition,))
    function_tool = FunctionTool(name="add", function=lambda a, b: a + b)
    mcp_tool = MCPTool(client=client, definition=definition)
    registry = ToolRegistry()
    registry.register(function_tool)
    registry.register(mcp_tool)
    assert registry.tools == (function_tool, mcp_tool)
    assert isinstance(registry.get_by_name("lookup"), BaseTool)


def test_existing_bindings_and_executor_run_mcp_tool_without_special_branch() -> None:
    specialist = SpecialistAgent(name="Worker")
    definition = _definitions()[0]
    client = MockMCPClient(server_id="demo", tools=(definition,))
    tool = MCPTool(client=client, definition=definition)
    registry = ToolRegistry()
    registry.register(tool)
    bindings = ToolBindingRegistry()
    bindings.assign(specialist.id, tool.id)
    result = ToolExecutor(registry=registry, bindings=bindings).execute(
        specialist=specialist,
        tool_name="lookup",
        arguments={"query": "value"},
    )
    assert result.success is True
    assert result.tool_id == tool.id
    assert client.calls[0].name == "lookup"


def test_unauthorized_mcp_invocation_does_not_call_or_mutate_configuration() -> None:
    specialist = SpecialistAgent(name="Worker", metadata={"fixed": True})
    definition = _definitions()[0]
    client = MockMCPClient(server_id="demo", tools=(definition,))
    tool = MCPTool(client=client, definition=definition)
    registry = ToolRegistry()
    registry.register(tool)
    bindings = ToolBindingRegistry()
    before = (registry.tools, bindings.bindings, specialist.metadata.copy())
    with pytest.raises(PermissionError):
        ToolExecutor(registry=registry, bindings=bindings).execute(
            specialist=specialist,
            tool_id=tool.id,
            arguments={"query": "value"},
        )
    assert client.calls == ()
    assert (registry.tools, bindings.bindings, specialist.metadata) == before


def test_mcp_execution_reuses_tool_trace_events_with_generic_source_metadata() -> None:
    specialist = SpecialistAgent(name="Worker")
    definition = _definitions()[0]
    client = MockMCPClient(server_id="demo", tools=(definition,))
    tool = MCPTool(client=client, definition=definition)
    registry = ToolRegistry()
    registry.register(tool)
    trace = ExecutionTrace(task_id="task")
    ToolExecutor(registry=registry).execute(
        specialist=specialist,
        tool_id=tool.id,
        arguments={"query": "value"},
        trace=trace,
    )
    assert tuple(event.event_type for event in trace.events) == (
        ExecutionEventType.TOOL_EXECUTION_STARTED.value,
        ExecutionEventType.TOOL_EXECUTION_COMPLETED.value,
    )
    metadata = trace.last_event.metadata  # type: ignore[union-attr]
    assert metadata["tool_metadata"]["mcp"]["server_id"] == "demo"
    assert metadata["tool_metadata"]["mcp"]["tool_name"] == "lookup"
