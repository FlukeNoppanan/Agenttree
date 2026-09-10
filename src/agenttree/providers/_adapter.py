"""Shared validation and normalization for optional synchronous SDK adapters."""

from copy import deepcopy
from importlib import import_module
import json
import math
from typing import Any

from agenttree.providers.exceptions import (
    ProviderConfigurationError, ProviderDependencyError, ProviderRuntimeError,
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
    except Exception as error:
        raise ProviderConfigurationError(f"Could not initialize {extra} client") from error


def field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def response(content: Any, model: Any, provider: str, usage: ProviderUsage | None,
             request: ProviderRequest, raw: Any, keep_raw: bool) -> ProviderResponse:
    if not isinstance(content, str) or not content.strip():
        raise ProviderRuntimeError("Provider returned no normalized text content")
    if not isinstance(model, str) or not model:
        raise ProviderRuntimeError("Provider returned an invalid model identity")
    raw_data = None
    if keep_raw:
        if isinstance(raw, dict):
            raw_data = deepcopy(raw)
        elif callable(getattr(raw, "model_dump", None)):
            raw_data = raw.model_dump(mode="json")
        # Unknown injected response types are intentionally not leaked.
    return ProviderResponse(
        content=content, model=model, provider=provider, usage=usage,
        metadata={"request_metadata": deepcopy(request.metadata)}, raw_response=raw_data,
    )
