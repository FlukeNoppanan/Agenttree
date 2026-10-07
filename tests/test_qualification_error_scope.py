"""Safe quota scope survives the SDK adapter boundary without raw data."""
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from agenttree.providers.exceptions import normalize_provider_error, ProviderRateLimitError


def error(violations, retry="18824s"):
    return SimpleNamespace(code=429, details={"error": {"message": "SECRET_SENTINEL", "details": [
        {"@type":"type.googleapis.com/google.rpc.QuotaFailure", "violations":violations},
        {"@type":"type.googleapis.com/google.rpc.RetryInfo", "retryDelay":retry}]}})


def test_model_quota_scope_and_long_retry_delay_are_preserved():
    result=normalize_provider_error(error([{"quotaDimensions":{"model":"real-model"},"quotaId":"RequestsPerDayPerModel"}]))
    assert isinstance(result,ProviderRateLimitError)
    assert result.failure_scope=="model" and result.quota_exhausted
    assert result.retry_after==18824
    assert "SECRET" not in str(result) and not hasattr(result,"details")


def test_unscoped_or_mixed_quota_remains_unknown():
    for entries in [[], [{"quotaId":"RequestsPerDay"}], [{"quotaDimensions":{"model":"m"}},{"quotaId":"global"}]]:
        assert normalize_provider_error(error(entries)).failure_scope=="unknown"


def test_retry_after_headers_and_dates():
    for value in ["7",format_datetime(datetime.now(timezone.utc)+timedelta(seconds=9),usegmt=True)]:
        result=normalize_provider_error(SimpleNamespace(code=429,response=SimpleNamespace(headers={"Retry-After":value})))
        assert 7 <= result.retry_after <= 9


def test_malformed_delay_does_not_expose_response():
    result=normalize_provider_error(error([],"SECRET_SENTINEL"))
    assert result.retry_after is None and "SECRET" not in str(result)


def test_http_status_exception_uses_response_status_and_retry_header():
    import httpx
    response=httpx.Response(429,headers={"Retry-After":"9"},request=httpx.Request("POST","https://provider.invalid"))
    try:response.raise_for_status()
    except httpx.HTTPStatusError as error:result=normalize_provider_error(error)
    assert isinstance(result,ProviderRateLimitError) and result.retry_after==9
