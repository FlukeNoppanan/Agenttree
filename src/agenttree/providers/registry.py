"""Ordered storage for configured provider instances."""

from types import MappingProxyType
from typing import Mapping

from agenttree.providers.base import BaseProvider


def _provider_key(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Provider names must be strings")
    key = name.strip().casefold()
    if not key:
        raise ValueError("Provider names cannot be empty")
    return key


class ProviderRegistry:
    """Register and retrieve providers by normalized name.

    Names are trimmed and compared case-insensitively. Registration order is
    preserved. The registry performs no model routing, fallback, or generation.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    @property
    def providers(self) -> tuple[BaseProvider, ...]:
        """Return providers in registration order as a snapshot."""
        return tuple(self._providers.values())

    @property
    def names(self) -> tuple[str, ...]:
        """Return normalized registered names in registration order."""
        return tuple(self._providers)

    @property
    def provider_map(self) -> Mapping[str, BaseProvider]:
        """Return a read-only snapshot keyed by normalized provider name."""
        return MappingProxyType(dict(self._providers))

    def register(self, provider: BaseProvider) -> None:
        """Register a provider, rejecting non-providers and duplicate names."""
        if not isinstance(provider, BaseProvider):
            raise TypeError("Only BaseProvider instances can be registered")
        key = _provider_key(provider.name)
        if key in self._providers:
            raise ValueError(f"Provider name already registered: {provider.name}")
        self._providers[key] = provider

    def get(self, name: str) -> BaseProvider:
        """Return a provider by name, raising ``KeyError`` if absent."""
        return self._providers[_provider_key(name)]

    def unregister(self, name: str) -> BaseProvider:
        """Remove and return a provider by name, raising ``KeyError`` if absent."""
        return self._providers.pop(_provider_key(name))
