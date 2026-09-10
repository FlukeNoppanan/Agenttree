"""Optional synchronous OpenAIProvider SDK adapter."""

from typing import Any

from agenttree.providers.base import BaseProvider
from agenttree.providers._adapter import create_client, field, prepare, response, validate_config
from agenttree.providers.exceptions import ProviderConfigurationError, ProviderRuntimeError
from agenttree.providers.models import ProviderConfig, ProviderRequest, ProviderResponse, ProviderUsage


class OpenAIProvider(BaseProvider):
    """Generate text through an optional SDK; injected clients support offline tests.

    Credentials belong to the SDK client, never ProviderConfig. No model is
    hardcoded. SDK construction does not perform generation or health checks.
    """

    def __init__(
        self, config: ProviderConfig, *, api_key: str | None = None, base_url: str | None = None,
        client: Any = None, keep_raw_response: bool = False,
    ) -> None:
        validate_config(config)
        if not isinstance(keep_raw_response, bool):
            raise ProviderConfigurationError("keep_raw_response must be a bool")
        super().__init__(config)
        if client is None:
            kwargs = {"api_key": api_key} if api_key is not None else {}
            if base_url is not None:
                kwargs["base_url"] = base_url
            client = create_client("openai", "OpenAI", "openai", **kwargs)
        self._client = client
        self._keep_raw = keep_raw_response

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Translate a generic request and normalize SDK output and failures."""
        model, prompt, temperature, tokens = prepare(self.config, request)
        kwargs: dict[str, Any] = {"model": model, "input": prompt}
        if request.system_prompt is not None:
            kwargs["instructions"] = request.system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tokens is not None:
            kwargs["max_output_tokens"] = tokens
        try:
            raw = self._client.responses.create(**kwargs)
            if field(raw, "error") is not None or field(raw, "status") in ("failed", "cancelled"):
                raise ProviderRuntimeError("OpenAI returned an unsuccessful response")
            reported = field(raw, "usage")
            usage = None if reported is None else ProviderUsage(
                input_tokens=field(reported, "input_tokens"),
                output_tokens=field(reported, "output_tokens"),
                total_tokens=field(reported, "total_tokens"),
            )
            return response(field(raw, "output_text"), field(raw, "model") or model,
                            self.name, usage, request, raw, self._keep_raw)
        except ProviderRuntimeError:
            raise
        except Exception as error:
            raise ProviderRuntimeError("OpenAIProvider generation failed") from error

