"""Common model provider interface."""

from abc import ABC, abstractmethod

from agenttree.providers.models import (
    ProviderConfig,
    ProviderRequest,
    ProviderResponse,
)


class BaseProvider(ABC):
    """Base interface implemented by model backends.

    Providers normalize their output into ``ProviderResponse`` and do not make
    orchestration decisions.
    """

    def __init__(self, config: ProviderConfig) -> None:
        if not isinstance(config, ProviderConfig):
            raise TypeError("config must be a ProviderConfig")
        self._config = config

    @property
    def name(self) -> str:
        """Return the stable configured provider identity."""
        return self._config.provider_name

    @property
    def config(self) -> ProviderConfig:
        """Return this provider's non-secret configuration."""
        return self._config

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate and normalize a response for ``request``."""
        raise NotImplementedError
