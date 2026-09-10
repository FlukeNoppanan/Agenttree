"""Offline normalization, sync-session ownership, and local transport tests."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import importlib.util
from pathlib import Path
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

from agenttree import SpecialistAgent
from agenttree.models import ExecutionTrace
from agenttree.tools import ToolBindingRegistry, ToolExecutor, ToolRegistry
from agenttree.tools.mcp import (
    MCPConfigurationError, MCPConnectionError, MCPDependencyError, MCPRuntimeError,
    MCPToolLoader, StdioMCPClient, StreamableHttpMCPClient,
)
from agenttree.tools.mcp.transports import _sync
from agenttree.tools.mcp.transports._normalization import call_result, definition


HAS_SDK = importlib.util.find_spec("mcp") is not None
real_sdk = pytest.mark.skipif(not HAS_SDK, reason="Optional MCP SDK is not installed")
SERVER = Path(__file__).parent / "fixtures" / "mcp_test_server.py"


def execute_stack(client: Any, tool_name: str, arguments: dict[str, Any]) -> Any:
    registry = ToolRegistry()
    tools = MCPToolLoader(client=client).load(registry)
    selected = next(tool for tool in tools if tool.name == tool_name)
    specialist = SpecialistAgent(name="Caller")
    bindings = ToolBindingRegistry()
    bindings.assign(specialist.id, selected.id)
    trace = ExecutionTrace(task_id="task")
    result = ToolExecutor(registry=registry, bindings=bindings).execute(
        specialist=specialist, tool_id=selected.id, arguments=arguments, trace=trace,
    )
    assert trace.event_count == 2
    assert trace.events[0].metadata["tool_id"] == selected.id
    return result


def test_tool_definition_preserves_schema_and_extra_metadata() -> None:
    raw = {"name": "echo", "description": "Echo", "inputSchema": {
        "type": "object", "properties": {"text": {"type": "string"}},
    }, "annotations": {"readOnlyHint": True}, "_meta": {"custom": [1]}}
    result = definition(raw)
    assert result.input_schema == raw["inputSchema"]
    assert result.metadata["annotations"] == {"readOnlyHint": True}
    result.metadata["_meta"]["custom"].append(2)
    assert raw["_meta"]["custom"] == [1]


@pytest.mark.parametrize("content, structured, failed", [
    ([{"type": "text", "text": "hello"}], None, False),
    ([{"type": "text", "text": "one"}, {"type": "text", "text": "two"}], {"x": [1]}, False),
    ([{"type": "image", "mimeType": "image/png", "data": "base64"}], None, False),
    ([{"type": "text", "text": "failure details"}], None, True),
    ([], {}, False),
])
def test_lossless_result_normalization(content: list[Any], structured: Any, failed: bool) -> None:
    raw = {"content": content, "structuredContent": structured, "isError": failed}
    result = call_result(raw, server_id="fixture", transport="stdio")
    assert result.success is not failed
    assert result.output == {"structured_content": structured, "content": content}
    assert result.raw_result == raw
    result.output["content"].append({"type": "text", "text": "extra"})
    assert result.raw_result == raw
    assert "extra" not in str(raw)


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    record: dict[str, Any] = {"calls": [], "owners": [], "closed": 0}

    class Session:
        async def __aenter__(self) -> "Session":
            record["owners"].append(asyncio.current_task())
            return self

        async def __aexit__(self, *exc: Any) -> None:
            record["owners"].append(asyncio.current_task())
            record["closed"] += 1

        async def initialize(self) -> None:
            if record.get("startup_timeout"):
                await asyncio.Event().wait()
            if record.get("startup_error"):
                raise RuntimeError("private startup detail")

        async def list_tools(self, cursor: str | None = None) -> Any:
            record["calls"].append(cursor)
            return SimpleNamespace(tools=[{
                "name": "first" if cursor is None else "second", "inputSchema": {},
            }], nextCursor="next" if cursor is None else None)

        async def call_tool(self, name: str, arguments: Any) -> Any:
            if record.get("cancelled"):
                raise asyncio.CancelledError()
            if record.get("call_timeout"):
                await asyncio.Event().wait()
            if record.get("call_error"):
                raise RuntimeError("private call detail")
            return {"content": [], "structuredContent": arguments}

    @asynccontextmanager
    async def transport(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        record["owners"].append(asyncio.current_task())
        record["parameters"] = args[0]
        try:
            yield (None, None)
        finally:
            record["owners"].append(asyncio.current_task())

    def load(name: str) -> Any:
        if name == "mcp":
            return SimpleNamespace(ClientSession=lambda *a, **kw: Session(),
                                   StdioServerParameters=lambda **kw: kw)
        return SimpleNamespace(stdio_client=transport)

    monkeypatch.setattr(_sync, "dependency", load)
    monkeypatch.setattr("agenttree.tools.mcp.transports.stdio.dependency", load)
    return record


def test_single_task_session_lifecycle_and_discovery_pagination(fake_session: dict[str, Any]) -> None:
    client = StdioMCPClient(server_id="fake", command="explicit", timeout=2)
    assert not client.is_connected and fake_session["owners"] == []
    with pytest.raises(MCPConfigurationError):
        client.list_tools()
    client.close()
    client.connect()
    client.connect()
    assert [tool.name for tool in client.list_tools()] == ["first", "second"]
    assert fake_session["calls"] == [None, "next"]
    client.close()
    client.close()
    assert len({id(task) for task in fake_session["owners"]}) == 1
    assert fake_session["closed"] == 1 and not client.is_connected
    client.connect()
    client.close()
    assert fake_session["closed"] == 2


def test_configuration_copy_and_no_secret_repr(fake_session: dict[str, Any]) -> None:
    env = {"TOKEN": "private-secret"}
    args = ["argument"]
    client = StdioMCPClient(server_id="fake", command="explicit", args=args, env=env)
    env["TOKEN"] = "changed"
    args.append("changed")
    with client:
        assert fake_session["parameters"]["env"] == {"TOKEN": "private-secret"}
        assert fake_session["parameters"]["args"] == ["argument"]
        assert "private-secret" not in repr(client)


def test_failures_close_owner_contexts_and_keep_errors_generic(fake_session: dict[str, Any]) -> None:
    client = StdioMCPClient(server_id="fake", command="explicit", timeout=2)
    fake_session["startup_error"] = True
    with pytest.raises(MCPConnectionError) as caught:
        client.connect()
    assert "private" not in str(caught.value)
    assert caught.value.__cause__ is not None
    client.close()
    assert fake_session["closed"] == 1 and not client.is_connected
    fake_session["startup_error"] = False
    fake_session["call_error"] = True
    with client:
        result = execute_stack(client, "first", {})
        assert not result.success and "MCPRuntimeError" in result.error
        assert "private call" not in result.error
    assert fake_session["closed"] == 2


def test_active_event_loop_usage_is_rejected_without_starting_worker(fake_session: dict[str, Any]) -> None:
    client = StdioMCPClient(server_id="fake", command="explicit")

    async def attempt() -> None:
        with pytest.raises(MCPRuntimeError, match="event-loop"):
            client.connect()

    asyncio.run(attempt())
    assert not client.is_connected and fake_session["owners"] == []


@pytest.mark.parametrize("failure", ["startup_timeout", "call_timeout", "cancelled"])
def test_timeouts_and_session_cancellation_release_resources(fake_session: dict[str, Any], failure: str) -> None:
    fake_session[failure] = True
    client = StdioMCPClient(server_id="fake", command="explicit", timeout=0.1)
    if failure == "startup_timeout":
        with pytest.raises(MCPConnectionError):
            client.connect()
    else:
        client.connect()
        with pytest.raises(MCPRuntimeError):
            client.call_tool("first", {})
    client.close()
    assert not client.is_connected and fake_session["closed"] == 1


@pytest.mark.parametrize("options", [
    {"command": ""}, {"args": "bad"}, {"args": [1]}, {"env": {"x": 1}},
    {"timeout": 0}, {"timeout": True}, {"timeout": float("inf")},
    {"server_id": " "}, {"cwd": "__nonexistent_agenttree_directory__"},
])
def test_invalid_stdio_configuration(options: dict[str, Any]) -> None:
    config = {"server_id": "fake", "command": "explicit", **options}
    with pytest.raises(MCPConfigurationError):
        StdioMCPClient(**config)


@pytest.mark.parametrize("options", [
    {"url": "file:///tmp/server"}, {"url": "http://user:secret@localhost/mcp"},
    {"url": "http://localhost:bad/mcp"}, {"headers": {"X": "bad\r\nvalue"}},
])
def test_invalid_http_configuration(options: dict[str, Any]) -> None:
    with pytest.raises(MCPConfigurationError):
        StreamableHttpMCPClient(**{"server_id": "fake", "url": "http://localhost/mcp", **options})


def test_optional_imports_and_mock_work_without_mcp_sdk() -> None:
    script = '''
import importlib.abc
import sys
class BlockMCP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'mcp':
            raise ModuleNotFoundError('Blocked MCP SDK')
sys.meta_path.insert(0, BlockMCP())
import agenttree
import agenttree.tools
from agenttree.tools.mcp import MockMCPClient, StdioMCPClient, StreamableHttpMCPClient, MCPDependencyError
assert MockMCPClient().list_tools() == ()
for factory in (lambda: StdioMCPClient(server_id='x', command='explicit'),
                lambda: StreamableHttpMCPClient(server_id='x', url='http://localhost/mcp')):
    try:
        factory()
    except MCPDependencyError as error:
        assert 'agenttree[mcp]' in str(error)
    else:
        raise AssertionError('Expected dependency error')
'''
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@real_sdk
def test_real_stdio_stack_and_process_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    import mcp.client.stdio as sdk_stdio

    original = sdk_stdio._create_platform_compatible_process
    processes: list[Any] = []

    async def capture(**kwargs: Any) -> Any:
        process = await original(**kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(sdk_stdio, "_create_platform_compatible_process", capture)
    client = StdioMCPClient(
        server_id="local", command=sys.executable, args=[str(SERVER.resolve())],
        cwd=str(SERVER.parent.resolve()), env={"AGENTTREE_TEST_MARKER": "configured"}, timeout=10,
    )
    process_handle = None
    try:
        client.connect()
        assert [tool.name for tool in client.list_tools()] == ["echo", "add", "fail", "process_info"]
        info = client.call_tool("process_info", {}).output["structured_content"]
        assert info["marker"] == "configured" and Path(info["cwd"]) == SERVER.parent.resolve()
        if sys.platform == "win32":
            # Windows venv launchers can have a different PID from the actual
            # server. Hold a native wait handle to verify the server also exits.
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            process_handle = kernel.OpenProcess(0x00100000, False, info["pid"])
            assert process_handle
        else:
            assert info["pid"] == processes[0].pid
        result = execute_stack(client, "add", {"a": 2, "b": 3})
        assert result.success and result.output["structured_content"] == {"sum": 5}
        assert result.metadata["mcp"]["call_metadata"]["transport"] == "stdio"
        failed = execute_stack(client, "fail", {})
        assert not failed.success
        assert failed.output["content"][0]["type"] == "text"
        with pytest.raises(MCPConfigurationError):
            client.call_tool("echo", {"bad": object()})
    finally:
        client.close()
        if process_handle is not None:
            try:
                assert kernel.WaitForSingleObject(process_handle, 0) == 0
            finally:
                kernel.CloseHandle(process_handle)
    client.close()
    assert not client.is_connected
    assert len(processes) == 1 and processes[0].returncode is not None


@real_sdk
def test_real_missing_executable_cleanup() -> None:
    client = StdioMCPClient(server_id="missing", command="__missing_agenttree_executable__", timeout=2)
    with pytest.raises(MCPConnectionError):
        client.connect()
    client.close()
    client.close()
    assert not client.is_connected


@real_sdk
def test_real_server_exit_during_initialization_releases_process(monkeypatch: pytest.MonkeyPatch) -> None:
    import mcp.client.stdio as sdk_stdio

    original = sdk_stdio._create_platform_compatible_process
    processes: list[Any] = []

    async def capture(**kwargs: Any) -> Any:
        process = await original(**kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(sdk_stdio, "_create_platform_compatible_process", capture)
    client = StdioMCPClient(server_id="exiting", command=sys.executable,
                            args=[str(SERVER.resolve()), "--fail-startup"], timeout=5)
    with pytest.raises(MCPConnectionError):
        client.connect()
    client.close()
    assert processes and all(process.returncode is not None for process in processes)


@real_sdk
def test_real_http_stack_and_shutdown() -> None:
    import uvicorn
    from tests.fixtures.mcp_test_server import server as fixture_server

    ready = threading.Event()

    class Server(uvicorn.Server):
        async def startup(self, sockets: Any = None) -> None:
            await super().startup(sockets)
            ready.set()

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app = fixture_server.streamable_http_app()
    http_server = Server(uvicorn.Config(app, log_level="error", access_log=False))
    worker = threading.Thread(target=http_server.run, kwargs={"sockets": [sock]}, daemon=True)
    worker.start()
    client = StreamableHttpMCPClient(
        server_id="http-local", url=f"http://127.0.0.1:{port}/mcp",
        headers={"X-Test": "private-header"}, timeout=10,
    )
    try:
        assert ready.wait(10), "Local HTTP server did not start"
        assert "private-header" not in repr(client)
        with client:
            result = execute_stack(client, "echo", {"text": "local"})
            assert result.success and result.output["structured_content"] == {"text": "local"}
            assert "private-header" not in str(result.metadata)
            assert result.metadata["mcp"]["call_metadata"]["transport"] == "streamable_http"
            assert not execute_stack(client, "fail", {}).success
    finally:
        client.close()
        http_server.should_exit = True
        worker.join(10)
        sock.close()
    assert not worker.is_alive() and not client.is_connected
