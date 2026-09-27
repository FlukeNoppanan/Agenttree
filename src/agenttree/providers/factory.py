"""Provider construction at the provider boundary."""

from typing import Any

from agenttree.providers.base import BaseProvider
from agenttree.providers.cerebras import CerebrasProvider
from agenttree.providers.compatible import OpenAICompatibleProvider
from agenttree.providers.gemini import GeminiProvider
from agenttree.providers.groq import GroqProvider
from agenttree.providers.models import ProviderConfig
from agenttree.providers.openrouter import OpenRouterProvider


_TYPES: dict[str, type[BaseProvider]] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "cerebras": CerebrasProvider,
    "openai-compatible": OpenAICompatibleProvider,
}


def create_provider(provider_type: str, config: ProviderConfig, **options: Any) -> BaseProvider:
    """Construct one named provider without runtime-specific branches."""
    if not isinstance(provider_type, str):
        raise TypeError("provider_type must be text")
    try:
        provider_class = _TYPES[provider_type.strip().casefold()]
    except KeyError:
        raise ValueError("Unknown provider type") from None
    return provider_class(config, **options)
