"""SDK-independent provider failures."""


class ProviderError(Exception):
    """Base class for errors at the provider boundary."""


class ProviderConfigurationError(ProviderError):
    """Invalid provider configuration or request."""


class ProviderDependencyError(ProviderConfigurationError):
    """The requested optional SDK is unavailable."""


class ProviderRuntimeError(ProviderError):
    """A provider call or response normalization failed."""


class ProviderAuthenticationError(ProviderRuntimeError):
    """Credentials were rejected by the provider."""


class ProviderRateLimitError(ProviderRuntimeError):
    """The provider rejected a call due to rate limits."""

    def __init__(self, message: str, *, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderUnavailableError(ProviderRuntimeError):
    """The provider service was unavailable."""


class ProviderTimeoutError(ProviderRuntimeError):
    """A provider call timed out."""


class ProviderInvalidRequestError(ProviderRuntimeError):
    """The provider rejected the request."""


class ProviderModelNotFoundError(ProviderRuntimeError):
    """The requested model was unavailable."""


class MalformedProviderResponseError(ProviderRuntimeError):
    """The provider response could not be normalized."""


def normalize_provider_error(error: Exception) -> ProviderRuntimeError:
    """Classify common SDK errors using safe status information only."""
    status = getattr(error, "status_code", getattr(error, "code", None))
    if status in (401, 403):
        return ProviderAuthenticationError("Provider authentication failed")
    if status == 429:
        headers = getattr(error, "headers", None)
        retry_value = headers.get("Retry-After") if headers is not None else None
        retry_after = None
        if isinstance(retry_value, str) and len(retry_value) <= 5 and retry_value.isdecimal():
            retry_after = min(int(retry_value), 3600)
        return ProviderRateLimitError("Provider rate limit exceeded", retry_after=retry_after)
    if status == 404:
        return ProviderModelNotFoundError("Provider model was not found")
    if status in (408, 504) or isinstance(error, TimeoutError):
        return ProviderTimeoutError("Provider request timed out")
    if (isinstance(status, int) and 500 <= status < 600) or isinstance(error, ConnectionError):
        return ProviderUnavailableError("Provider service is unavailable")
    if status in (400, 422):
        return ProviderInvalidRequestError("Provider rejected the request")
    return ProviderRuntimeError("Provider generation failed")
