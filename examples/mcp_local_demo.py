"""Optional local-only MCP tool demonstration for thesis presentations."""

from pathlib import Path
import sys

from agenttree import SpecialistAgent
from agenttree.models import ExecutionTrace
from agenttree.tools import ToolBindingRegistry, ToolExecutor, ToolRegistry
from agenttree.tools.mcp import MCPToolLoader, StdioMCPClient


SERVER = Path(__file__).resolve().parents[1] / "tests/fixtures/mcp_test_server.py"


def main() -> int:
    """Call the local fixture's add tool and close its process deterministically."""
    client = StdioMCPClient(
        server_id="thesis-local",
        command=sys.executable,
        args=(str(SERVER),),
        cwd=str(SERVER.parent),
        timeout=10,
    )
    registry = ToolRegistry()
    bindings = ToolBindingRegistry()
    specialist = SpecialistAgent(
        name="Demo Tool Specialist",
        id="demo-tool-specialist",
    )
    trace = ExecutionTrace(task_id="thesis-mcp-001")
    try:
        client.connect()
        tools = MCPToolLoader(client=client).load(registry)
        tool = next(item for item in tools if item.name == "add")
        bindings.assign(specialist.id, tool.id)
        result = ToolExecutor(registry=registry, bindings=bindings).execute(
            specialist=specialist,
            tool_id=tool.id,
            arguments={"a": 2, "b": 3},
            trace=trace,
        )
    finally:
        client.close()
    print("AgentTree Local MCP Demo")
    print(f"Specialist: {specialist.name}")
    print(f"MCP server: {client.server_id} (local stdio fixture)")
    print(f"Tool: {tool.name}")
    print(f"Success: {result.success}")
    print(f"Result: {result.output['structured_content']}")
    print(f"Trace events: {trace.event_count}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
