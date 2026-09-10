"""Optional synchronous GeminiProvider SDK adapter."""

from typing import Any

from agenttree.providers.base import BaseProvider
from agenttree.providers._adapter import create_client, field, prepare, response, validate_config
from agenttree.providers.exceptions import ProviderConfigurationError, ProviderRuntimeError
from agenttree.providers.models import ProviderConfig, ProviderRequest, ProviderResponse, ProviderUsage


class GeminiProvider(BaseProvider):
    """Generate text through an optional SDK; injected clients support offline tests.

    Credentials belong to the SDK client, never ProviderConfig. No model is
    hardcoded. SDK construction does not perform generation or health checks.
    """

    def __init__(
        self, config: ProviderConfig, *, api_key: str | None = None,
        client: Any = None, keep_raw_response: bool = False,
    ) -> None:
        validate_config(config)
        if not isinstance(keep_raw_response, bool):
            raise ProviderConfigurationError("keep_raw_response must be a bool")
        super().__init__(config)
        if client is None:
            kwargs = {"api_key": api_key} if api_key is not None else {}
            client = create_client("google.genai", "Client", "gemini", **kwargs)
        self._client = client
        self._keep_raw = keep_raw_response

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Translate a generic request and normalize SDK output and failures."""
        model, prompt, temperature, tokens = prepare(self.config, request)
        options: dict[str, Any] = {}
        if request.system_prompt is not None:
            options["system_instruction"] = request.system_prompt
        if temperature is not None:
            options["temperature"] = temperature
        if tokens is not None:
            options["max_output_tokens"] = tokens
        try:
            raw = self._client.models.generate_content(model=model, contents=prompt, config=options)
            reported = field(raw, "usage_metadata")
            usage = None if reported is None else ProviderUsage(
                input_tokens=field(reported, "prompt_token_count"),
                output_tokens=field(reported, "candidates_token_count"),
                total_tokens=field(reported, "total_token_count"),
            )
            return response(field(raw, "text"), field(raw, "model_version") or model,
                            self.name, usage, request, raw, self._keep_raw)
        except ProviderRuntimeError:
            raise
        except Exception as error:
            raise ProviderRuntimeError("GeminiProvider generation failed") from error

