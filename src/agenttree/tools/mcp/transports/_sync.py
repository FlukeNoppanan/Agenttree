"""Single-owner asynchronous MCP sessions behind a synchronous client API."""

import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from copy import deepcopy
from datetime import timedelta
import importlib
import json
import math
import threading
from typing import Any, AsyncContextManager, Mapping

from agenttree.tools.mcp.base import BaseMCPClient
from agenttree.tools.mcp.exceptions import (
    MCPConfigurationError, MCPConnectionError, MCPDependencyError, MCPRuntimeError,
)
from agenttree.tools.mcp.models import MCPCallResult, MCPToolDefinition
from agenttree.tools.mcp.transports._normalization import call_result, definition


def dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise MCPDependencyError(
            'Real MCP clients require optional dependencies: pip install "agenttree[mcp]"',
        ) from error


def nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise MCPConfigurationError(f"{name} must be nonempty text without NUL")
    return value


class _SyncMCPClient(BaseMCPClient):
    """Internal serialized actor; its one task owns transport/session contexts."""

    _transport_name: str

    def __init__(self, *, server_id: str, timeout: float) -> None:
        nonempty(server_id, "server_id")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not 0 < timeout < math.inf):
            raise MCPConfigurationError("timeout must be a positive finite number")
        super().__init__(server_id=server_id)
        self._timeout = timeout
        self._sdk = dependency("mcp")
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owner: asyncio.Task[Any] | None = None
        self._queue: asyncio.Queue[Any] | None = None
        self._ready: Future[None] = Future()
        self._current: Future[Any] | None = None
        self._failure: BaseException | None = None

    def _transport(self) -> AsyncContextManager[Any]:
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        """Return whether an initialized worker/session is currently alive."""
        return bool(self._thread and self._thread.is_alive() and self._ready.done()
                    and self._ready.exception() is None and self._failure is None)

    @staticmethod
    def _check_sync() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise MCPRuntimeError(
            "Synchronous MCP methods cannot run on an active event-loop thread; "
            "call them from a synchronous worker instead",
        )

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._owner = asyncio.current_task()
        self._queue = asyncio.Queue()
        # Covers transport entry as well as initialize; cancellation unwinds the
        # same owner task's contexts rather than exiting AnyIO scopes elsewhere.
        deadline = self._loop.call_later(self._timeout, self._owner.cancel)
        try:
            async with self._transport() as streams:
                async with self._sdk.ClientSession(
                    streams[0], streams[1],
                    read_timeout_seconds=timedelta(seconds=self._timeout),
                ) as session:
                    await session.initialize()
                    deadline.cancel()
                    self._ready.set_result(None)
                    while True:
                        command = await self._queue.get()
                        if command is None:
                            break
                        operation, arguments, future = command
                        self._current = future
                        try:
                            value = await asyncio.wait_for(
                                self._dispatch(session, operation, arguments), self._timeout,
                            )
                        except Exception as error:
                            future.set_exception(error)
                        else:
                            future.set_result(value)
                        finally:
                            if future.done():
                                self._current = None
        finally:
            deadline.cancel()

    async def _dispatch(self, session: Any, operation: str, arguments: Any) -> Any:
        if operation == "list":
            tools: list[MCPToolDefinition] = []
            cursor = None
            cursors: set[str] = set()
            while True:
                page = await session.list_tools(cursor=cursor)
                tools.extend(definition(item) for item in page.tools)
                cursor = page.nextCursor
                if cursor is None:
                    return tuple(tools)
                if cursor in cursors:
                    raise MCPConnectionError("MCP discovery repeated a pagination cursor")
                cursors.add(cursor)
        result = await session.call_tool(arguments[0], arguments=arguments[1])
        return call_result(result, server_id=self.server_id, transport=self._transport_name)

    def _worker(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self._failure = error
        finally:
            error = self._failure or MCPConnectionError("MCP session closed")
            if not self._ready.done():
                self._ready.set_exception(error)
            if self._current is not None and not self._current.done():
                self._current.set_exception(error)

    def _stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        if thread.is_alive() and self._loop is not None and self._queue is not None:
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            except RuntimeError:
                pass  # Worker completed concurrently.
        thread.join(self._timeout + 5)
        if thread.is_alive() and self._loop is not None and self._owner is not None:
            try:
                self._loop.call_soon_threadsafe(self._owner.cancel)
            except RuntimeError:
                pass
            thread.join(self._timeout + 5)
        if thread.is_alive():
            raise MCPRuntimeError("MCP worker did not finish resource cleanup")
        self._thread = None
        self._loop = None
        self._owner = None
        self._queue = None

    def connect(self) -> None:
        """Explicitly initialize a session; repeated connect is a no-op."""
        self._check_sync()
        with self._lock:
            if self.is_connected:
                return
            self._stop()
            self._ready = Future()
            self._failure = None
            self._thread = threading.Thread(target=self._worker, name="agenttree-mcp", daemon=True)
            self._thread.start()
            try:
                self._ready.result(self._timeout + 5)
            except BaseException as error:
                self._stop()
                if not isinstance(error, (Exception, asyncio.CancelledError)):
                    raise
                raise MCPConnectionError("MCP connection or initialization failed") from error

    def _request(self, operation: str, arguments: Any = None) -> Any:
        self._check_sync()
        with self._lock:
            if not self.is_connected:
                if self._failure is not None:
                    raise MCPConnectionError("MCP session is unavailable") from self._failure
                raise MCPConfigurationError("Call connect() before using the MCP client")
            future: Future[Any] = Future()
            assert self._loop is not None and self._queue is not None
            try:
                self._loop.call_soon_threadsafe(
                    self._queue.put_nowait, (operation, arguments, future),
                )
                return future.result(self._timeout + 1)
            except BaseException as error:
                if not isinstance(error, (Exception, asyncio.CancelledError)):
                    raise
                if isinstance(error, FutureTimeoutError):
                    self._stop()
                kind = MCPConnectionError if operation == "list" else MCPRuntimeError
                raise kind("MCP discovery failed" if operation == "list" else "MCP tool call failed") from error

    def list_tools(self) -> tuple[MCPToolDefinition, ...]:
        """Discover all pages in server order after explicit connection."""
        return self._request("list")

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> MCPCallResult:
        """Invoke one explicitly named tool and preserve normalized content."""
        nonempty(name, "name")
        if not isinstance(arguments, Mapping) or any(not isinstance(k, str) for k in arguments):
            raise MCPConfigurationError("arguments must be a mapping with string keys")
        copied = deepcopy(dict(arguments))
        try:
            json.dumps(copied, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise MCPConfigurationError("arguments must contain JSON-compatible values") from error
        return self._request("call", (name, copied))

    def close(self) -> None:
        """Close session/process resources; repeated close is a no-op."""
        self._check_sync()
        with self._lock:
            previous_failure = self._failure
            self._stop()
            if self._failure is not None and previous_failure is None:
                raise MCPRuntimeError("MCP resource cleanup failed") from self._failure

    def __enter__(self) -> "_SyncMCPClient":
        """Connect for a synchronous with block."""
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Close on either normal completion or a raised caller error."""
        self.close()
