"""SDK-independent provider failures."""

import math
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime



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

    def __init__(self, message: str, *, retry_after: float | None = None,
                 failure_scope: str = "provider", quota_exhausted: bool = False, signal=None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.failure_scope = failure_scope
        self.quota_exhausted = quota_exhausted
        self.signal = signal


class ProviderUnavailableError(ProviderRuntimeError):
    """The provider service was unavailable."""


class ProviderNetworkError(ProviderUnavailableError):
    """A network failure, distinct from Provider capacity."""
    def __init__(self, message, *, stage="connect"):
        super().__init__(message)
        self.stage = stage


class ProviderTimeoutError(ProviderRuntimeError):
    """A provider call timed out."""


class ProviderInvalidRequestError(ProviderRuntimeError):
    """Rejected request, with an optional allowlisted feature diagnostic."""

    def __init__(self, message: str, *, unsupported_parameter: str | None = None):
        super().__init__(message)
        self.unsupported_parameter = unsupported_parameter


class ProviderModelNotFoundError(ProviderRuntimeError):
    """The requested model was unavailable."""


class MalformedProviderResponseError(ProviderRuntimeError):
    """The provider response could not be normalized."""


def _normalize_provider_error(error: Exception) -> ProviderRuntimeError:
    """Classify common SDK errors using safe status information only."""
    status = (getattr(error, "status_code", None) or getattr(error, "code", None)
              or getattr(getattr(error, "response", None), "status_code", None))
    try:
        status = int(status) if status is not None else None
    except (ValueError, TypeError):
        status = None
    if type(error).__name__ == "ResourceExhausted" or getattr(error,"status",None) == "RESOURCE_EXHAUSTED":
        status = 429
    if status in (401, 403):
        return ProviderAuthenticationError("Provider authentication failed")
    if status == 429:
        from agenttree.providers.traffic_failure import rate_limit_signal
        signal = rate_limit_signal(error, status)
        return ProviderRateLimitError("Provider rate limit exceeded", retry_after=signal.retry_after,
                                      failure_scope=signal.scope, quota_exhausted=signal.quota_exhausted,
                                      signal=signal)
    if status == 404:
        return ProviderModelNotFoundError("Provider model was not found")
    if (status in (408, 504) or isinstance(error, TimeoutError)
            or any(base.__name__ == "TimeoutException" and base.__module__.startswith("httpx")
                   for base in type(error).__mro__)):
        return ProviderTimeoutError("Provider request timed out")
    from agenttree.providers.traffic_failure import failure_from_exception
    network = failure_from_exception(error)
    if status is None and (network.category == "network" or isinstance(error, OSError)):
        return ProviderNetworkError("Provider network connection failed", stage=network.transport_stage or "connect")
    if (isinstance(status, int) and 500 <= status < 600) or isinstance(error, ConnectionError):
        return ProviderUnavailableError("Provider service is unavailable")
    if status in (400, 422):
        payload = getattr(error, "body", None)
        if not isinstance(payload, dict) and callable(getattr(error, "read", None)):
            try:
                payload = json.loads(error.read(65_537))
            except (ValueError, TypeError, OSError):
                payload = None
        detail = payload.get("error", payload) if isinstance(payload, dict) else None
        unsupported = None
        if isinstance(detail, dict) and detail.get("code") in {"unsupported_parameter", "unsupported_value"}:
            parameter = detail.get("param")
            if parameter in {"response_format", "temperature"}:
                unsupported = parameter
        return ProviderInvalidRequestError("Provider rejected the request", unsupported_parameter=unsupported)
    return ProviderRuntimeError("Provider generation failed")


def normalize_provider_error(error: Exception) -> ProviderRuntimeError:
    """Preserve allowlisted HTTP status on the safe generic failure contract."""
    from dataclasses import replace
    from agenttree.providers.traffic_failure import failure_from_exception
    normalized = _normalize_provider_error(error)
    failure = failure_from_exception(normalized)
    status = (getattr(error, "status_code", None) or getattr(error, "code", None)
              or getattr(getattr(error, "response", None), "status_code", None))
    try:
        status = int(status) if status is not None else None
    except (ValueError, TypeError):
        status = None
    if failure.http_status is None and status is not None and 100 <= status <= 599:
        failure = replace(failure, http_status=status)
    normalized.failure = failure
    return normalized
