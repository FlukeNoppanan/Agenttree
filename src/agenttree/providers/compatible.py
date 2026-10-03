"""Shared synchronous OpenAI-compatible chat and model discovery transport."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import re
import socket
from threading import RLock
from time import monotonic
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from agenttree.providers._adapter import prepare, validate_config
from agenttree.providers.base import BaseProvider
from agenttree.providers.exceptions import (
    MalformedProviderResponseError, ProviderConfigurationError,
    normalize_provider_error,
)
from agenttree.providers.models import (
    ProviderCapabilities, ProviderConfig, ProviderModel, ProviderRequest,
    ProviderResponse, ProviderUsage, ProviderStreamChunk,
)
from agenttree.core.execution_control import check_execution


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request: Request, fp: Any, code: int,
                         message: str, headers: Any, new_url: str) -> None:
        return None


def _safe_url(value: str) -> str:
    if not isinstance(value, str):
        raise ProviderConfigurationError("base_url must be a URL")
    if value != value.strip() or any(ord(character) < 33 for character in value):
        raise ProviderConfigurationError("base_url contains invalid whitespace")
    try:
        parts = urlsplit(value)
        parts.port
    except ValueError:
        raise ProviderConfigurationError("base_url is invalid") from None
    if (not parts.hostname or parts.username or parts.password or parts.query or
            parts.fragment or parts.scheme not in ("http", "https")):
        raise ProviderConfigurationError("base_url must be an absolute HTTP(S) URL without credentials")
    if parts.scheme == "http" and parts.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ProviderConfigurationError("Non-loopback provider URLs require HTTPS")
    return value.rstrip("/")


def _safe_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in (headers or {}).items():
        if (not isinstance(key, str) or not isinstance(value, str) or
                re.fullmatch(r"[A-Za-z0-9-]+", key) is None or
                "\r" in value or "\n" in value or
                key.casefold() in {"authorization", "proxy-authorization", "host", "cookie",
                                    "content-type", "x-api-key", "api-key"}):
            raise ProviderConfigurationError("Invalid custom provider header")
        result[key] = value
    return result


def _positive_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ProviderConfigurationError("timeout must be a positive finite number")
    return float(value)


class OpenAICompatibleProvider(BaseProvider):
    """HTTP chat adapter with no implicit redirects or automatic retries.

    Credentials and custom headers stay private on the instance. Model results
    are cached per instance for a bounded TTL and can be refreshed explicitly.
    """

    _allowed_options: frozenset[str] = frozenset()

    def __init__(
        self, config: ProviderConfig, *, base_url: str, api_key: str | None = None,
        headers: Mapping[str, str] | None = None, timeout: float = 15,
        model_discovery: bool = True, model_cache_ttl: float = 300,
        allowed_options: frozenset[str] | None = None,
        streaming: bool = False,
    ) -> None:
        validate_config(config)
        super().__init__(config)
        self._base_url = _safe_url(base_url)
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip() or
                                     "\r" in api_key or "\n" in api_key):
            raise ProviderConfigurationError("api_key must be nonempty text")
        self._api_key = api_key
        self._headers = _safe_headers(headers)
        self._timeout = _positive_timeout(timeout)
        if not isinstance(model_discovery, bool):
            raise ProviderConfigurationError("model_discovery must be a bool")
        self._model_discovery = model_discovery
        if not isinstance(streaming, bool):
            raise ProviderConfigurationError("streaming must be a bool")
        self._streaming = streaming
        self._cache_ttl = _positive_timeout(model_cache_ttl)
        self._model_cache: tuple[ProviderModel, ...] | None = None
        self._cache_expires = 0.0
        self._cache_lock = RLock()
        if allowed_options is not None and (
            not isinstance(allowed_options, frozenset) or
            not all(isinstance(item, str) for item in allowed_options)
        ):
            raise ProviderConfigurationError("allowed_options must be a frozenset of names")
        self._allowed_options = allowed_options if allowed_options is not None else type(self)._allowed_options
        if self._allowed_options & {"model", "messages", "stream", "tools", "tool_choice",
                                    "response_format", "api_key", "base_url", "headers",
                                    "authorization", "url"}:
            raise ProviderConfigurationError("Unsafe provider option allowlist")
        self._opener = build_opener(_NoRedirect())

    @property
    def provider_type(self) -> str:
        return "openai-compatible"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(chat=True, tool_calling=True, streaming=self._streaming,
                                    model_discovery=self._model_discovery)

    def _json_request(self, path: str, payload: dict[str, Any] | None = None,
                      *, timeout: float | None = None) -> dict[str, Any]:
        headers = {"Accept": "application/json", "User-Agent": "AgentTree",
                   **self._headers}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = None
        if payload is not None:
            try:
                body = json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
            except (TypeError, ValueError):
                raise ProviderConfigurationError("Request contains non-JSON values") from None
            headers["Content-Type"] = "application/json"
        request = Request(self._base_url + path, data=body, headers=headers,
                          method="POST" if body is not None else "GET")
        try:
            with self._opener.open(request, timeout=timeout or self._timeout) as response:
                raw = response.read(2_000_001)
        except HTTPError as error:
            normalized = normalize_provider_error(error)
        except (URLError, TimeoutError, socket.timeout, OSError) as error:
            reason = error.reason if isinstance(error, URLError) else error
            normalized = normalize_provider_error(reason if isinstance(reason, Exception) else error)
        else:
            if len(raw) > 2_000_000:
                raise MalformedProviderResponseError("Provider response exceeded size limit")
            try:
                parsed = json.loads(raw)
            except (UnicodeDecodeError, ValueError):
                raise MalformedProviderResponseError("Provider returned invalid JSON") from None
            if not isinstance(parsed, dict):
                raise MalformedProviderResponseError("Provider response must be an object")
            return parsed
        raise normalized

    def _chat_payload(self, request: ProviderRequest) -> tuple[str, dict[str, Any]]:
        model, prompt, temperature, tokens = prepare(self.config, request)
        if request.timeout is not None:
            _positive_timeout(request.timeout)
        if not isinstance(request.provider_options, dict):
            raise ProviderConfigurationError("provider_options must be an object")
        forbidden = set(request.provider_options) - self._allowed_options
        if forbidden:
            raise ProviderConfigurationError("Unsupported or unsafe provider option")
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": prompt})
        for turn in request.tool_history:
            calls = turn.get("calls", ())
            results = turn.get("results", ())
            messages.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": call["id"], "type": "function", "function": {
                    "name": call["name"], "arguments": json.dumps(call["arguments"], allow_nan=False)
                    if not isinstance(call["arguments"], str) else call["arguments"]}}
                for call in calls]})
            for result in results:
                messages.append({"role": "tool", "tool_call_id": result["call_id"],
                                 "content": json.dumps({"success": result["success"],
                                                        "output": result["output"],
                                                        "error": result["error"]}, allow_nan=False)})
        payload: dict[str, Any] = {"model": model, "messages": messages,
                                   "stream": False}
        if request.tools:
            payload["tools"] = [{"type": "function", "function": deepcopy(item)}
                                for item in request.tools]
            payload["tool_choice"] = request.tool_choice or "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if tokens is not None:
            payload["max_tokens"] = tokens
        if request.response_format is not None:
            if self.capabilities.structured_output is False or not isinstance(request.response_format, dict):
                raise ProviderConfigurationError("response_format is unsupported or invalid")
            known_model = next((item for item in self._model_cache or () if item.id == model), None)
            if (known_model is not None and known_model.capabilities is not None and
                    known_model.capabilities.structured_output is False):
                raise ProviderConfigurationError("Selected model does not support structured output")
            payload["response_format"] = deepcopy(request.response_format)
        payload.update(deepcopy(request.provider_options))
        return model, payload

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        model, payload = self._chat_payload(request)
        started = monotonic()
        data = self._json_request("/chat/completions", payload, timeout=request.timeout)
        latency = round((monotonic() - started) * 1000, 2)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError
            # Function-call-only messages may omit the optional text field.
            # The checks below still reject responses without text or calls.
            content = message.get("content")
            model_name = data.get("model", model)
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError):
            raise MalformedProviderResponseError("Provider chat response is malformed") from None
        tool_calls = _tool_calls(message.get("tool_calls"))
        if not isinstance(content, (str, type(None))) or (not tool_calls and not (content or "").strip()) or not isinstance(model_name, str):
            raise MalformedProviderResponseError("Provider returned no text response")
        usage_data = data.get("usage")
        usage = None
        if isinstance(usage_data, dict):
            usage = ProviderUsage(
                input_tokens=_token_count(usage_data.get("prompt_tokens")),
                output_tokens=_token_count(usage_data.get("completion_tokens")),
                total_tokens=_token_count(usage_data.get("total_tokens")),
            )
        return ProviderResponse(content=(content or "").strip(), provider=self.name,
                                model=model_name, usage=usage,
                                finish_reason=finish_reason if isinstance(finish_reason, str) else None,
                                tool_calls=tool_calls,
                                metadata={"latency_ms": latency})

    def generate_stream(self, request: ProviderRequest):
        if not self._streaming:
            raise ProviderConfigurationError("Provider streaming is disabled")
        model, payload = self._chat_payload(request)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        headers = {"Accept": "text/event-stream", "Content-Type": "application/json",
                   "User-Agent": "AgentTree", **self._headers}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            body = json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError):
            raise ProviderConfigurationError("Request contains non-JSON values") from None
        http_request = Request(self._base_url + "/chat/completions", data=body,
                               headers=headers, method="POST")
        content: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish_reason = None
        usage = None
        model_name = model
        received = 0
        done = False
        try:
            with self._opener.open(http_request, timeout=request.timeout or self._timeout) as stream:
                for raw_line in stream:
                    check_execution()
                    received += len(raw_line)
                    if received > 2_000_000 or len(raw_line) > 65_536:
                        raise MalformedProviderResponseError("Provider stream exceeded size limit")
                    if not raw_line.startswith(b"data: "):
                        continue
                    data = raw_line[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    try:
                        item = json.loads(data)
                        if not isinstance(item, dict):
                            raise ValueError()
                        choices = item.get("choices", [])
                        if choices:
                            choice = choices[0]
                            delta = choice.get("delta", {})
                            text_delta = delta.get("content") or ""
                            if not isinstance(text_delta, str):
                                raise ValueError()
                            if text_delta:
                                content.append(text_delta)
                                yield ProviderStreamChunk(delta_text=text_delta)
                            for part in delta.get("tool_calls") or ():
                                index = part["index"]
                                if not isinstance(index, int) or index < 0 or index > 32:
                                    raise ValueError()
                                current = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                                current["id"] += part.get("id") or ""
                                function = part.get("function") or {}
                                current["name"] += function.get("name") or ""
                                current["arguments"] += function.get("arguments") or ""
                                from agenttree.providers._adapter import tool_argument_limit
                                if (len(current["arguments"].encode("utf-8")) > tool_argument_limit(request, current["name"]) or
                                        len(current["name"].encode("utf-8")) > 256 or
                                        len(current["id"].encode("utf-8")) > 256):
                                    raise MalformedProviderResponseError("Provider tool call exceeded size limit")
                            finish_reason = choice.get("finish_reason") or finish_reason
                        reported = item.get("usage")
                        if isinstance(reported, dict):
                            usage = ProviderUsage(_token_count(reported.get("prompt_tokens")),
                                                  _token_count(reported.get("completion_tokens")),
                                                  _token_count(reported.get("total_tokens")))
                        if isinstance(item.get("model"), str):
                            model_name = item["model"]
                    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                        raise MalformedProviderResponseError("Provider stream chunk is malformed") from None
        except HTTPError as error:
            raise normalize_provider_error(error) from None
        except (URLError, TimeoutError, socket.timeout, OSError) as error:
            reason = error.reason if isinstance(error, URLError) else error
            raise normalize_provider_error(reason if isinstance(reason, Exception) else error) from None
        if not done:
            raise MalformedProviderResponseError("Provider stream ended before completion")
        tool_calls = tuple({"id": value["id"], "type": "function",
                            "function": {"name": value["name"],
                                         "arguments": value["arguments"]}}
                           for _, value in sorted(calls.items()))
        if not tool_calls and not "".join(content).strip():
            raise MalformedProviderResponseError("Provider returned no text response")
        if any(not value["id"] or not value["name"] for value in calls.values()):
            raise MalformedProviderResponseError("Provider tool call is incomplete")
        yield ProviderStreamChunk(response=ProviderResponse(
            content="".join(content).strip(), provider=self.name, model=model_name,
            usage=usage, finish_reason=finish_reason, tool_calls=tool_calls))

    def list_models(self, *, refresh: bool = False) -> tuple[ProviderModel, ...]:
        if not self._model_discovery:
            raise ProviderConfigurationError("Model discovery is disabled")
        with self._cache_lock:
            if not refresh and self._model_cache is not None and monotonic() < self._cache_expires:
                return self._model_cache
            data = self._json_request("/models")
            items = data.get("data")
            if not isinstance(items, list):
                raise MalformedProviderResponseError("Provider model list is malformed")
            models = tuple(_model(item, self.name) for item in items)
            self._model_cache = models
            self._cache_expires = monotonic() + self._cache_ttl
            return models


def _token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _model(item: Any, provider: str) -> ProviderModel:
    if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
        raise MalformedProviderResponseError("Provider model entry is malformed")
    limits = item.get("limits") if isinstance(item.get("limits"), dict) else {}
    context = item.get("context_length", item.get("context_window", limits.get("max_context_length")))
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        context = None
    reported = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
    chat = _chat_capability(item)
    capabilities = None
    if reported or chat is not None:
        capabilities = ProviderCapabilities(
            chat=chat,
            streaming=_boolean(reported.get("streaming")),
            structured_output=_boolean(reported.get("structured_outputs")),
            tool_calling=_boolean(reported.get("function_calling", reported.get("tools"))),
            vision=_boolean(reported.get("vision")),
            reasoning=_boolean(reported.get("reasoning")),
        )
    safe_fields = ("pricing", "supported_parameters", "input_modalities",
                   "output_modalities", "architecture", "description", "owned_by")
    metadata = {key: deepcopy(item[key]) for key in safe_fields if key in item}
    return ProviderModel(id=item["id"], provider=provider,
                         name=item.get("name") if isinstance(item.get("name"), str) else None,
                         context_window=context, capabilities=capabilities, metadata=metadata)


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _tool_calls(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise MalformedProviderResponseError("Provider tool calls are malformed")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise MalformedProviderResponseError("Provider tool calls are malformed")
        function = item.get("function")
        if not isinstance(function, dict):
            raise MalformedProviderResponseError("Provider tool calls are malformed")
        result.append({"id": item.get("id"), "type": item.get("type"),
                       "function": {"name": function.get("name"),
                                    "arguments": function.get("arguments")}})
    return tuple(result)


def _chat_capability(item: dict[str, Any]) -> bool | None:
    inputs, outputs = item.get("input_modalities"), item.get("output_modalities")
    if isinstance(inputs, list) and isinstance(outputs, list):
        return "text" in inputs and "text" in outputs
    architecture = item.get("architecture")
    if isinstance(architecture, dict) and architecture.get("modality") == "text":
        return True
    return None
