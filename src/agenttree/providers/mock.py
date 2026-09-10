"""Deterministic provider for local tests and framework development."""

from typing import Any

from agenttree.providers.base import BaseProvider
from agenttree.providers.models import (
    ProviderConfig,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
)


class MockProvider(BaseProvider):
    """Return deterministic responses without network or model calls.

    A complete static response may be supplied. Otherwise each call returns a
    normalized response using ``response_content`` and the configured identity.
    Requests are exposed as an insertion-ordered tuple for test assertions.
    """

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        response_content: str = "Mock response",
        response: ProviderResponse | None = None,
        usage: ProviderUsage | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config or ProviderConfig(provider_name="mock", model="mock-model"))
        if response is not None and not isinstance(response, ProviderResponse):
            raise TypeError("response must be a ProviderResponse")
        self._response_content = response_content
        self._response = response
        self._usage = usage
        self._response_metadata = dict(response_metadata or {})
        self._requests: list[ProviderRequest] = []

    @property
    def requests(self) -> tuple[ProviderRequest, ...]:
        """Return a snapshot of requests in call order."""
        return tuple(self._requests)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Record ``request`` and return the configured deterministic response."""
        if not isinstance(request, ProviderRequest):
            raise TypeError("request must be a ProviderRequest")
        self._requests.append(request)
        if self._response is not None:
            return self._response
        return ProviderResponse(
            content=self._response_content,
            provider=self.name,
            model=request.model or self.config.model,
            usage=self._usage,
            metadata=dict(self._response_metadata),
        )
