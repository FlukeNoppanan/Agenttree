"""OpenRouter chat provider."""

from typing import Mapping

from agenttree.providers.compatible import OpenAICompatibleProvider
from agenttree.providers.models import ProviderConfig


class OpenRouterProvider(OpenAICompatibleProvider):
    _allowed_options = frozenset({"top_p", "stop", "seed", "frequency_penalty", "presence_penalty"})

    def __init__(self, config: ProviderConfig, *, api_key: str,
                 headers: Mapping[str, str] | None = None,
                 timeout: float = 15, model_cache_ttl: float = 300) -> None:
        super().__init__(config, base_url="https://openrouter.ai/api/v1",
                         api_key=api_key, headers=headers, timeout=timeout,
                         model_cache_ttl=model_cache_ttl, streaming=True)

    @property
    def provider_type(self) -> str:
        return "openrouter"
