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
    timeout: float | None = None
    response_format: dict[str, Any] | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: str | None = None
    tool_history: tuple[dict[str, Any], ...] = ()
    # Host-controlled per-declaration transport bounds; never sent to the model.
    # Unknown/ordinary calls retain the generic bound. ToolSession still validates
    # authorization, schema, content and metadata before invocation.
    tool_argument_limits: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized model output independent of any provider SDK."""

    content: str
    provider: str
    model: str | None = None
    usage: ProviderUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None
    finish_reason: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ProviderStreamChunk:
    """One transient text delta or the complete normalized response."""

    delta_text: str = ""
    response: ProviderResponse | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.delta_text, str):
            raise TypeError("delta_text must be text")
        if self.response is not None and not isinstance(self.response, ProviderResponse):
            raise TypeError("response must be a ProviderResponse")


@dataclass(frozen=True)
class ProviderCapabilities:
    """Provider-level support; None means unknown or model-dependent."""

    chat: bool | None = True
    streaming: bool | None = None
    structured_output: bool | None = None
    tool_calling: bool | None = None
    vision: bool | None = None
    reasoning: bool | None = None
    model_discovery: bool = False


@dataclass(frozen=True)
class ProviderModel:
    """One discovered model with only known fields populated."""

    id: str
    provider: str
    name: str | None = None
    context_window: int | None = None
    capabilities: ProviderCapabilities | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderValidationResult:
    valid: bool
    provider: str
    message: str
    error_category: str | None = None
