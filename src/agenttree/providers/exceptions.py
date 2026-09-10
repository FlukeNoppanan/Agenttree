"""SDK-independent provider failures."""


class ProviderError(Exception):
    """Base class for errors at the provider boundary."""


class ProviderConfigurationError(ProviderError):
    """Invalid provider configuration or request."""


class ProviderDependencyError(ProviderConfigurationError):
    """The requested optional SDK is unavailable."""


class ProviderRuntimeError(ProviderError):
    """A provider call or response normalization failed."""
