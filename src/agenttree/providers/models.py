"""Vendor-neutral model provider data contracts."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderUsage:
    """Optional token counts reported by a model provider."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ProviderConfig:
    """Non-secret configuration shared by provider implementations.

    Credentials are deliberately excluded because this object may later appear
    in logs or traces.
    """

    provider_name: str
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_name = self.provider_name.strip()
        if not normalized_name:
            raise ValueError("provider_name cannot be empty")
        object.__setattr__(self, "provider_name", normalized_name)


@dataclass(frozen=True)
class ProviderRequest:
    """A vendor-neutral generation request with structured caller context."""

    prompt: str
    system_prompt: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized model output independent of any provider SDK."""

    content: str
    provider: str
    model: str | None = None
    usage: ProviderUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None
