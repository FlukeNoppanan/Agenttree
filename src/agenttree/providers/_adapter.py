"""Shared validation and normalization for optional synchronous SDK adapters."""

from copy import deepcopy
from importlib import import_module
import json
import math
from typing import Any

from agenttree.providers.exceptions import (
    ProviderConfigurationError, ProviderDependencyError, MalformedProviderResponseError,
)
from agenttree.providers.models import ProviderConfig, ProviderRequest, ProviderResponse, ProviderUsage


def validate_options(model: str | None, temperature: float | None, tokens: int | None) -> None:
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ProviderConfigurationError("model must be a non-empty string")
    if temperature is not None and (
        isinstance(temperature, bool) or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature) or temperature < 0
    ):
        raise ProviderConfigurationError("temperature must be a finite non-negative number")
    if tokens is not None and (
        isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0
    ):
        raise ProviderConfigurationError("max_tokens must be a positive integer")


def validate_config(config: ProviderConfig) -> None:
    if not isinstance(config, ProviderConfig):
        raise ProviderConfigurationError("config must be a ProviderConfig")
    validate_options(config.model, config.temperature, config.max_tokens)


def prepare(config: ProviderConfig, request: ProviderRequest) -> tuple[str, str, float | None, int | None]:
    if not isinstance(request, ProviderRequest):
        raise ProviderConfigurationError("request must be a ProviderRequest")
    validate_options(request.model, request.temperature, request.max_tokens)
    model = request.model if request.model is not None else config.model
    if model is None:
        raise ProviderConfigurationError("Configure a model or supply ProviderRequest.model")
    if not isinstance(request.prompt, str):
        raise ProviderConfigurationError("prompt must be a string")
    if request.system_prompt is not None and not isinstance(request.system_prompt, str):
        raise ProviderConfigurationError("system_prompt must be a string or None")
    if not isinstance(request.context, dict) or not isinstance(request.metadata, dict):
        raise ProviderConfigurationError("context and metadata must be dictionaries")
    prompt = request.prompt
    if request.context:
        try:
            context = json.dumps(request.context, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ProviderConfigurationError("context must contain JSON-compatible values") from error
        prompt += "\n\nContext (JSON):\n" + context
    return (
        model, prompt,
        request.temperature if request.temperature is not None else config.temperature,
        request.max_tokens if request.max_tokens is not None else config.max_tokens,
    )


def create_client(module: str, constructor: str, extra: str, **kwargs: Any) -> Any:
    try:
        sdk = import_module(module)
    except ImportError as error:
        raise ProviderDependencyError(
            f"Install the optional SDK with pip install 'agenttree[{extra}]'",
        ) from error
    try:
        return getattr(sdk, constructor)(**kwargs)
    except Exception:
        pass
    raise ProviderConfigurationError(f"Could not initialize {extra} client")


def field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _redact_raw(value: Any) -> Any:
    forbidden = {"authorization", "api_key", "apikey", "secret", "token",
                 "reasoning", "reasoning_content", "thinking", "thoughts",
                 "chain_of_thought"}
    if isinstance(value, dict):
        if value.get("thought") is True or value.get("type") in {"reasoning", "thinking", "analysis"}:
            return {"reasoning_present": True}
        return {key: _redact_raw(item) for key, item in value.items()
                if isinstance(key, str) and key.casefold() not in forbidden}
    if isinstance(value, list):
        return [_redact_raw(item) for item in value]
    return deepcopy(value)



def final_content(value: Any, *, depth: int = 0) -> str | dict | None:
    """Extract only explicit answer parts, never hidden reasoning or SDK trees.

    Vendor adapters choose their answer field first. This helper accepts common
    text/content/parts carriers inside that field; unrelated metadata is ignored.
    Multiple text parts concatenate in order. Native objects remain structured.
    """
    if depth > 8:
        raise MalformedProviderResponseError("Answer content nesting exceeds limit")
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, list):
        if len(value) > 1024:
            raise MalformedProviderResponseError("Answer content exceeds part limit")
        parts = [final_content(item, depth=depth + 1) for item in value]
        if any(isinstance(item, dict) for item in parts):
            raise MalformedProviderResponseError("Ambiguous structured answer parts")
        return "".join(item or "" for item in parts)
    if isinstance(value, dict):
        if value.get("thought") is True or value.get("type") in {
            "reasoning", "thinking", "analysis", "reasoning_content", "thought"}:
            return None
        for key in ("text", "content", "parts"):
            if key in value:
                return final_content(value[key], depth=depth + 1)
        # A protocol object is an answer only when the caller explicitly passes
        # the structured field. Do not remove/rename its canonical fields.
        return value
    raise MalformedProviderResponseError("Unsupported answer content representation")


def response(content: Any, model: Any, provider: str, usage: ProviderUsage | None,
             request: ProviderRequest, raw: Any, keep_raw: bool) -> ProviderResponse:
    content = final_content(content)
    structured_content = None
    if isinstance(content, dict) and request.response_format is not None:
        try:
            # An explicit structured field supplied by the adapter, not raw SDK data.
            content_text = json.dumps(content, allow_nan=False, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError) as error:
            raise MalformedProviderResponseError("Invalid native structured content") from error
        structured_content = json.loads(content_text)
        content = content_text
    if not isinstance(content, str) or not content.strip():
        raise MalformedProviderResponseError("Provider returned no normalized text content")
    if not isinstance(model, str) or not model:
        raise MalformedProviderResponseError("Provider returned an invalid model identity")
    raw_data = None
    if keep_raw:
        if isinstance(raw, dict):
            raw_data = deepcopy(raw)
        elif callable(getattr(raw, "model_dump", None)):
            raw_data = raw.model_dump(mode="json")
        # Unknown injected response types are intentionally not leaked.
        if raw_data is not None:
            raw_data = _redact_raw(raw_data)
    return ProviderResponse(
        content=content, model=model, provider=provider, usage=usage,
        metadata={"request_metadata": deepcopy(request.metadata)}, raw_response=raw_data,
        structured_content=structured_content,
    )


def tool_argument_bytes(arguments: Any) -> int:
    """Bound native UTF-8 JSON, or the raw JSON text before parsing."""
    try:
        if isinstance(arguments, str):
            return len(arguments.encode("utf-8"))
        return len(json.dumps(arguments, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        raise MalformedProviderResponseError("Provider Tool arguments are not valid JSON") from None


def tool_argument_limit(request: ProviderRequest, name: str | None, default: int = 16_384) -> int:
    limit = request.tool_argument_limits.get(name, default)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ProviderConfigurationError("Invalid Tool argument byte limit")
    return limit
