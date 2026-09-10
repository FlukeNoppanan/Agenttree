"""Optional Streamable HTTP MCP client with explicit endpoint configuration."""

from contextlib import asynccontextmanager
import re
from typing import Any, AsyncIterator, Mapping
from urllib.parse import urlsplit

from agenttree.tools.mcp.exceptions import MCPConfigurationError
from agenttree.tools.mcp.transports._sync import _SyncMCPClient, dependency, nonempty


class StreamableHttpMCPClient(_SyncMCPClient):
    """Connect to a developer-supplied HTTP(S) endpoint and optional headers.

    Headers are copied, never placed in repr/trace metadata, and only sent to
    the configured endpoint. Redirects and environment proxy discovery are
    disabled. No OAuth, endpoint discovery, or automatic connection is added.
    """

    _transport_name = "streamable_http"

    def __init__(
        self, *, server_id: str, url: str,
        headers: Mapping[str, str] | None = None, timeout: float = 30.0,
    ) -> None:
        nonempty(url, "url")
        if any(character.isspace() or ord(character) < 32 for character in url):
            raise MCPConfigurationError("url must not contain whitespace or control characters")
        try:
            parsed = urlsplit(url)
            valid = (parsed.scheme in ("http", "https") and parsed.hostname
                     and not parsed.username and not parsed.password and not parsed.fragment)
            parsed.port
        except ValueError as error:
            raise MCPConfigurationError("Invalid MCP endpoint URL") from error
        if not valid:
            raise MCPConfigurationError("url must be an HTTP(S) endpoint without credentials or fragment")
        if headers is not None and (not isinstance(headers, Mapping) or any(
            not isinstance(k, str) or re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", k) is None
            or not isinstance(v, str) or not v.isascii()
            or any(c in k + v for c in "\r\n\0") for k, v in headers.items()
        )):
            raise MCPConfigurationError("headers must map valid names to text values")
        self._url = url
        self._headers = dict(headers or {})
        super().__init__(server_id=server_id, timeout=timeout)
        self._http = dependency("httpx")
        self._http_transport = dependency("mcp.client.streamable_http")

    @asynccontextmanager
    async def _transport(self) -> AsyncIterator[Any]:
        async with self._http.AsyncClient(
            headers=self._headers, timeout=self._timeout,
            follow_redirects=False, trust_env=False,
        ) as client:
            async with self._http_transport.streamable_http_client(
                self._url, http_client=client, terminate_on_close=True,
            ) as streams:
                yield streams
