"""Optional synchronous OllamaProvider SDK adapter."""

from typing import Any

from agenttree.providers.base import BaseProvider
from agenttree.providers._adapter import create_client, field, prepare, response, validate_config
from agenttree.providers.exceptions import ProviderConfigurationError, ProviderRuntimeError
from agenttree.providers.models import ProviderConfig, ProviderRequest, ProviderResponse, ProviderUsage


class OllamaProvider(BaseProvider):
    """Generate text through an optional SDK; injected clients support offline tests.

    Credentials belong to the SDK client, never ProviderConfig. No model is
    hardcoded. SDK construction does not perform generation or health checks.
    """

    def __init__(
        self, config: ProviderConfig, *, host: str | None = None,
        client: Any = None, keep_raw_response: bool = False,
    ) -> None:
        validate_config(config)
        if not isinstance(keep_raw_response, bool):
            raise ProviderConfigurationError("keep_raw_response must be a bool")
        super().__init__(config)
        if client is None:
            kwargs = {"host": host} if host is not None else {}
            client = create_client("ollama", "Client", "ollama", **kwargs)
        self._client = client
        self._keep_raw = keep_raw_response

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Translate a generic request and normalize SDK output and failures."""
        model, prompt, temperature, tokens = prepare(self.config, request)
        kwargs: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if request.system_prompt is not None:
            kwargs["system"] = request.system_prompt
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if tokens is not None:
            options["num_predict"] = tokens
        if options:
            kwargs["options"] = options
        try:
            raw = self._client.generate(**kwargs)
            if field(raw, "error"):
                raise ProviderRuntimeError("Ollama returned an unsuccessful response")
            input_tokens = field(raw, "prompt_eval_count")
            output_tokens = field(raw, "eval_count")
            usage = ProviderUsage(input_tokens=input_tokens, output_tokens=output_tokens)
            return response(field(raw, "response"), field(raw, "model") or model,
                            self.name, usage, request, raw, self._keep_raw)
        except ProviderRuntimeError:
            raise
        except Exception as error:
            raise ProviderRuntimeError("OllamaProvider generation failed") from error

