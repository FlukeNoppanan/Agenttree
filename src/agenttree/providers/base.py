"""Common model provider interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from agenttree.providers.models import (
    ProviderConfig,
    ProviderRequest,
    ProviderResponse,
    ProviderCapabilities, ProviderModel, ProviderValidationResult,
    ProviderStreamChunk,
)
from agenttree.providers.exceptions import ProviderConfigurationError, ProviderError


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

    @property
    def provider_type(self) -> str:
        """Implementation family, distinct from the configured instance name."""
        return "custom"

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Conservative default for custom provider implementations."""
        return ProviderCapabilities()

    def list_models(self, *, refresh: bool = False) -> tuple[ProviderModel, ...]:
        """Discover models when supported by the implementation."""
        raise ProviderConfigurationError("Provider does not support model discovery")

    def validate_connection(self) -> ProviderValidationResult:
        """Use model discovery when available; report safe, structured failure."""
        if not self.capabilities.model_discovery:
            return ProviderValidationResult(False, self.name,
                "Provider does not support connection validation", "unsupported")
        try:
            self.list_models(refresh=True)
        except ProviderError as error:
            return ProviderValidationResult(False, self.name, str(error),
                                            type(error).__name__)
        except Exception:
            return ProviderValidationResult(False, self.name,
                                            "Provider validation failed", "ProviderRuntimeError")
        return ProviderValidationResult(True, self.name, "Connection validated")

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate and normalize a response for ``request``."""
        raise NotImplementedError

    def generate_stream(self, request: ProviderRequest) -> Iterator[ProviderStreamChunk]:
        """Optional native stream; subclasses declaring streaming support override this."""
        raise ProviderConfigurationError("Provider does not support streaming")
