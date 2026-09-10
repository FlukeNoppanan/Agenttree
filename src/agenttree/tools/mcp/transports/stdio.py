"""Optional stdio MCP transport using the installed SDK's process lifecycle."""

from contextlib import asynccontextmanager
import os
from typing import Any, AsyncIterator, Mapping, Sequence

from agenttree.tools.mcp.exceptions import MCPConfigurationError
from agenttree.tools.mcp.transports._sync import _SyncMCPClient, dependency, nonempty


class StdioMCPClient(_SyncMCPClient):
    """Connect to an explicitly configured executable without a shell.

    The SDK inherits its minimal safe environment and adds supplied overrides.
    No executable is downloaded or discovered. Configuration is not included in
    repr or trace metadata. Use connect()/close() or a synchronous with block.
    """

    _transport_name = "stdio"

    def __init__(
        self, *, server_id: str, command: str, args: Sequence[str] = (),
        env: Mapping[str, str] | None = None, cwd: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._command = nonempty(command, "command")
        if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
            raise MCPConfigurationError("args must be a sequence of strings")
        if any(not isinstance(arg, str) or "\0" in arg for arg in args):
            raise MCPConfigurationError("args must contain strings without NUL")
        if env is not None and (not isinstance(env, Mapping) or any(
            not isinstance(k, str) or not k or "=" in k or "\0" in k
            or not isinstance(v, str) or "\0" in v for k, v in env.items()
        )):
            raise MCPConfigurationError("env must map valid environment names to strings")
        if cwd is not None:
            nonempty(cwd, "cwd")
            if not os.path.isdir(cwd):
                raise MCPConfigurationError("cwd must be an existing directory")
        self._args = tuple(args)
        self._env = dict(env) if env is not None else None
        self._cwd = cwd
        super().__init__(server_id=server_id, timeout=timeout)
        self._stdio = dependency("mcp.client.stdio")

    @asynccontextmanager
    async def _transport(self) -> AsyncIterator[Any]:
        params = self._sdk.StdioServerParameters(
            command=self._command, args=list(self._args), env=self._env, cwd=self._cwd,
        )
        # Server diagnostic output is deliberately excluded from framework logs.
        with open(os.devnull, "w", encoding="utf-8") as errlog:
            async with self._stdio.stdio_client(params, errlog=errlog) as streams:
                yield streams
