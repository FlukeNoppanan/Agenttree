"""Optional synchronous GeminiProvider SDK adapter."""

from typing import Any
from threading import RLock
from time import monotonic
import math
from uuid import uuid4

from agenttree.providers.base import BaseProvider
from agenttree.providers._adapter import create_client, field, prepare, response, validate_config
from agenttree.providers.exceptions import ProviderConfigurationError, ProviderRuntimeError, normalize_provider_error
from agenttree.providers.models import (
    ProviderCapabilities, ProviderConfig, ProviderModel, ProviderRequest,
    ProviderResponse, ProviderUsage,
    ProviderStreamChunk,
)
from agenttree.core.execution_control import check_execution


class GeminiProvider(BaseProvider):
    """Generate text through an optional SDK; injected clients support offline tests.

    Credentials belong to the SDK client, never ProviderConfig. No model is
    hardcoded. SDK construction does not perform generation or health checks.
    """

    def __init__(
        self, config: ProviderConfig, *, api_key: str | None = None,
        client: Any = None, keep_raw_response: bool = False,
        model_cache_ttl: float = 300,
        timeout: float = 15,
    ) -> None:
        validate_config(config)
        if not isinstance(keep_raw_response, bool):
            raise ProviderConfigurationError("keep_raw_response must be a bool")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
                not math.isfinite(timeout) or timeout <= 0):
            raise ProviderConfigurationError("timeout must be positive")
        super().__init__(config)
        if client is None:
            kwargs = {"api_key": api_key} if api_key is not None else {}
            kwargs["http_options"] = {"timeout": max(1, int(timeout * 1000))}
            client = create_client("google.genai", "Client", "gemini", **kwargs)
        self._client = client
        self._keep_raw = keep_raw_response
        if (isinstance(model_cache_ttl, bool) or not isinstance(model_cache_ttl, (int, float))
                or not math.isfinite(model_cache_ttl) or model_cache_ttl <= 0):
            raise ProviderConfigurationError("model_cache_ttl must be positive")
        self._cache_ttl = model_cache_ttl
        self._cache_lock = RLock()
        self._model_cache: tuple[ProviderModel, ...] | None = None
        self._cache_expires = 0.0
        # Keep native model content (including thought signatures) inside this
        # adapter for the next function-response turn; never expose it in traces.
        self._tool_content_by_id: dict[str, Any] = {}
        self._tool_original_ids: dict[str, str | None] = {}

    @property
    def provider_type(self) -> str:
        return "gemini"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(chat=True, streaming=True, tool_calling=True,
                                    model_discovery=True)

    def list_models(self, *, refresh: bool = False) -> tuple[ProviderModel, ...]:
        with self._cache_lock:
            if not refresh and self._model_cache is not None and monotonic() < self._cache_expires:
                return self._model_cache
            try:
                raw_models = self._client.models.list()
                models: list[ProviderModel] = []
                for item in raw_models:
                    model_id = field(item, "name")
                    if not isinstance(model_id, str) or not model_id:
                        continue
                    actions = field(item, "supported_actions")
                    chat = None if actions is None else "generateContent" in actions
                    context = field(item, "input_token_limit")
                    if not isinstance(context, int) or context <= 0:
                        context = None
                    models.append(ProviderModel(
                        id=model_id, provider=self.name,
                        name=field(item, "display_name") if isinstance(field(item, "display_name"), str) else None,
                        context_window=context,
                        capabilities=ProviderCapabilities(chat=chat),
                    ))
            except Exception as error:
                normalized = normalize_provider_error(error)
            else:
                self._model_cache = tuple(models)
                self._cache_expires = monotonic() + self._cache_ttl
                return self._model_cache
        raise normalized

    def _generation_args(self, request: ProviderRequest) -> tuple[str, Any, dict[str, Any]]:
        model, prompt, temperature, tokens = prepare(self.config, request)
        if request.provider_options or request.response_format is not None or request.timeout is not None:
            raise ProviderConfigurationError("Gemini adapter does not support these optional request fields")
        options: dict[str, Any] = {}
        if request.system_prompt is not None:
            options["system_instruction"] = request.system_prompt
        if temperature is not None:
            options["temperature"] = temperature
        if tokens is not None:
            options["max_output_tokens"] = tokens
        contents: Any = prompt
        if request.tools:
            options["tools"] = [{"function_declarations": list(request.tools)}]
            options["automatic_function_calling"] = {"disable": True}
            contents = [prompt]
            for turn in request.tool_history:
                calls = turn.get("calls", ())
                cached = self._tool_content_by_id.get(calls[0]["id"]) if calls else None
                if cached is not None:
                    contents.append(cached)
                else:
                    model_parts = [{"function_call": {"name": call["name"],
                                                      "args": call["arguments"] if isinstance(call["arguments"], dict) else {},
                                                      "id": call["id"]}}
                                   for call in calls]
                    contents.append({"role": "model", "parts": model_parts})
                result_parts = []
                for result in turn.get("results", ()):
                    function_response = {"name": result["name"],
                                         "response": {"success": result["success"],
                                                      "output": result["output"],
                                                      "error": result["error"]}}
                    original_id = self._tool_original_ids.get(result["call_id"], result["call_id"])
                    if original_id is not None:
                        function_response["id"] = original_id
                    result_parts.append({"function_response": function_response})
                contents.append({"role": "user", "parts": result_parts})
        return model, contents, options

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Translate a generic request and normalize SDK output and failures."""
        model, contents, options = self._generation_args(request)
        try:
            raw = self._client.models.generate_content(model=model, contents=contents, config=options)
            reported = field(raw, "usage_metadata")
            usage = None if reported is None else ProviderUsage(
                input_tokens=field(reported, "prompt_token_count"),
                output_tokens=field(reported, "candidates_token_count"),
                total_tokens=field(reported, "total_token_count"),
            )
            calls = []
            candidates = field(raw, "candidates") or ()
            for candidate in candidates[:1]:
                content = field(candidate, "content")
                for part in field(content, "parts") or ():
                    function = field(part, "function_call")
                    if function is not None:
                        original_id = field(function, "id")
                        if not isinstance(original_id, str) or not original_id:
                            original_id = None
                        call_id = original_id or str(uuid4())
                        self._tool_original_ids[call_id] = original_id
                        calls.append({"id": call_id,
                                      "function": {"name": field(function, "name"),
                                                   "arguments": field(function, "args") or {}}})
            try:
                text_content = field(raw, "text")
            except (ValueError, AttributeError):
                text_content = None
            if calls:
                native_content = field(candidates[0], "content")
                for call in calls:
                    self._tool_content_by_id[call["id"]] = native_content
                while len(self._tool_content_by_id) > 128:
                    old = next(iter(self._tool_content_by_id))
                    self._tool_content_by_id.pop(old)
                    self._tool_original_ids.pop(old, None)
                return ProviderResponse(content=text_content if isinstance(text_content, str) else "",
                                        provider=self.name, model=field(raw, "model_version") or model,
                                        usage=usage, tool_calls=tuple(calls))
            return response(text_content, field(raw, "model_version") or model,
                            self.name, usage, request, raw, self._keep_raw)
        except ProviderRuntimeError:
            raise
        except Exception as error:
            normalized = normalize_provider_error(error)
        raise normalized

    def generate_stream(self, request: ProviderRequest):
        model, contents, options = self._generation_args(request)
        text_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        usage = None
        model_name = model
        try:
            for chunk in self._client.models.generate_content_stream(
                    model=model, contents=contents, config=options):
                check_execution()
                reported = field(chunk, "usage_metadata")
                if reported is not None:
                    usage = ProviderUsage(field(reported, "prompt_token_count"),
                                          field(reported, "candidates_token_count"),
                                          field(reported, "total_token_count"))
                model_name = field(chunk, "model_version") or model_name
                candidates = field(chunk, "candidates") or ()
                for candidate in candidates[:1]:
                    native_content = field(candidate, "content")
                    for part in field(native_content, "parts") or ():
                        function = field(part, "function_call")
                        if function is not None:
                            original_id = field(function, "id")
                            if not isinstance(original_id, str) or not original_id:
                                original_id = None
                            call_id = original_id or str(uuid4())
                            arguments = field(function, "args") or {}
                            from json import dumps
                            if len(dumps(arguments, default=str).encode("utf-8")) > 16_384:
                                raise ProviderRuntimeError("Provider tool call exceeded size limit")
                            self._tool_original_ids[call_id] = original_id
                            self._tool_content_by_id[call_id] = native_content
                            calls.append({"id": call_id, "function": {
                                "name": field(function, "name"),
                                "arguments": arguments}})
                            if len(calls) > 32:
                                raise ProviderRuntimeError("Provider stream has too many tool calls")
                        part_text = field(part, "text")
                        if isinstance(part_text, str) and part_text:
                            text_parts.append(part_text)
                            yield ProviderStreamChunk(delta_text=part_text)
            if not calls and not "".join(text_parts).strip():
                raise ProviderRuntimeError("Provider returned no text response")
            yield ProviderStreamChunk(response=ProviderResponse(
                content="".join(text_parts).strip(), provider=self.name,
                model=model_name, usage=usage, tool_calls=tuple(calls)))
        except ProviderRuntimeError:
            raise
        except Exception as error:
            raise normalize_provider_error(error) from None
