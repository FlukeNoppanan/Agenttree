"""Cerebras chat provider."""

from agenttree.providers.compatible import OpenAICompatibleProvider
from agenttree.providers.models import ProviderConfig


class CerebrasProvider(OpenAICompatibleProvider):
    _allowed_options = frozenset({"top_p", "stop", "seed"})

    def __init__(self, config: ProviderConfig, *, api_key: str,
                 timeout: float = 15, model_cache_ttl: float = 300) -> None:
        super().__init__(config, base_url="https://api.cerebras.ai/v1",
                         api_key=api_key, timeout=timeout, model_cache_ttl=model_cache_ttl,
                         streaming=True)

    @property
    def provider_type(self) -> str:
        return "cerebras"
