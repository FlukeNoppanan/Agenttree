"""Offline SDK contract tests for all real provider adapters."""

from copy import deepcopy
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from agenttree.agents import SpecialistAgent
from agenttree.core import ProviderSpecialistExecutor
from agenttree.models import Subtask, Task
from agenttree.providers import (
    GeminiProvider, OllamaProvider, OpenAIProvider, ProviderConfig,
    ProviderConfigurationError, ProviderDependencyError, ProviderRegistry,
    ProviderRequest, ProviderResponse, ProviderRuntimeError, ProviderUsage,
)


ADAPTERS = (OpenAIProvider, GeminiProvider, OllamaProvider)


class FakeResponse(SimpleNamespace):
    """SDK-style attributes and plain-data dump."""

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return deepcopy(vars(self))


class FakeClient:
    """SDK-shaped client recording calls without I/O."""

    def __init__(self, result: Any, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.responses = SimpleNamespace(create=self.call)
        self.models = SimpleNamespace(generate_content=self.call)
        self.generate = self.call

    def call(self, **kwargs: Any) -> Any:
        self.calls.append(deepcopy(kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def sdk_result(adapter: type, *, usage: bool = True) -> FakeResponse:
    if adapter is OpenAIProvider:
        return FakeResponse(
            output_text="answer", model="reported-model", status="completed",
            usage={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
            if usage else None,
        )
    if adapter is GeminiProvider:
        return FakeResponse(
            text="answer", model_version="reported-model",
            usage_metadata={
                "prompt_token_count": 3, "candidates_token_count": 4,
                "total_token_count": 9,
            } if usage else None,
        )
    return FakeResponse(
        response="answer", model="reported-model",
        prompt_eval_count=3 if usage else None,
        eval_count=4 if usage else None,
    )


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_same_request_normalizes_across_providers(adapter: type) -> None:
    client = FakeClient(sdk_result(adapter))
    config = ProviderConfig(provider_name=adapter.__name__, model="default-model")
    provider = adapter(config, client=client, keep_raw_response=True)
    request = ProviderRequest(
        prompt="Question", system_prompt="Instructions", model="override",
        temperature=0, max_tokens=12, context={"z": [1], "a": True},
        metadata={"local": {"id": 1}},
    )
    before = deepcopy(request)
    result = provider.generate(request)
    assert isinstance(result, ProviderResponse)
    assert result.content == "answer"
    assert result.model == "reported-model"
    assert result.provider == config.provider_name
    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 4
    assert result.usage.total_tokens == (
        7 if adapter is OpenAIProvider else 9 if adapter is GeminiProvider else None
    )
    assert isinstance(result.raw_response, dict)
    result.raw_response["changed"] = True
    assert not hasattr(client.result, "changed")
    assert request == before
    call = client.calls[0]
    assert call["model"] == "override"
    expected_prompt = 'Question\n\nContext (JSON):\n{"a": true, "z": [1]}'
    if adapter is OpenAIProvider:
        assert call == {
            "model": "override", "input": expected_prompt,
            "instructions": "Instructions", "temperature": 0, "max_output_tokens": 12,
        }
    elif adapter is GeminiProvider:
        assert call == {
            "model": "override", "contents": expected_prompt,
            "config": {"system_instruction": "Instructions", "temperature": 0, "max_output_tokens": 12},
        }
    else:
        assert call == {
            "model": "override", "prompt": expected_prompt, "system": "Instructions",
            "stream": False, "options": {"temperature": 0, "num_predict": 12},
        }
    result.metadata["request_metadata"]["local"]["id"] = 2
    assert request.metadata["local"]["id"] == 1


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_config_defaults_and_missing_usage(adapter: type) -> None:
    client = FakeClient(sdk_result(adapter, usage=False))
    provider = adapter(
        ProviderConfig(provider_name="fixture", model="configured", temperature=0.2, max_tokens=8),
        client=client,
    )
    result = provider.generate(ProviderRequest(prompt="plain"))
    assert client.calls[0]["model"] == "configured"
    options = client.calls[0].get("options", client.calls[0].get("config", client.calls[0]))
    assert options["temperature"] == 0.2
    assert options["num_predict" if adapter is OllamaProvider else "max_output_tokens"] == 8
    assert result.usage is None or result.usage == ProviderUsage()
    assert result.raw_response is None


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_omitted_options_and_request_only_model(adapter: type) -> None:
    client = FakeClient(sdk_result(adapter))
    provider = adapter(ProviderConfig(provider_name="fixture"), client=client)
    provider.generate(ProviderRequest(prompt="plain", model="request-only"))
    call = client.calls[0]
    assert call["model"] == "request-only"
    assert "temperature" not in call
    assert "max_output_tokens" not in call
    assert "instructions" not in call and "system" not in call
    assert call.get("config", {}) == {}
    assert call.get("options", {}) == {}


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_missing_model_and_invalid_context_fail_before_sdk_call(adapter: type) -> None:
    client = FakeClient(sdk_result(adapter))
    provider = adapter(ProviderConfig(provider_name="fixture"), client=client)
    with pytest.raises(ProviderConfigurationError, match="model"):
        provider.generate(ProviderRequest(prompt="x"))
    with pytest.raises(ProviderConfigurationError, match="JSON"):
        provider.generate(ProviderRequest(prompt="x", model="m", context={"x": object()}))
    with pytest.raises(ProviderConfigurationError, match="max_tokens"):
        provider.generate(ProviderRequest(prompt="x", model="m", max_tokens=True))
    assert client.calls == []
    with pytest.raises(ProviderConfigurationError):
        adapter(object(), client=client)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_sdk_failure_has_safe_provider_runtime_error(adapter: type) -> None:
    original = RuntimeError("fake SDK rate limit")
    client = FakeClient(None, original)
    provider = adapter(ProviderConfig(provider_name="fixture", model="m"), client=client)
    with pytest.raises(ProviderRuntimeError) as caught:
        provider.generate(ProviderRequest(prompt="x"))
    assert caught.value.__context__ is None
    assert "fake SDK rate limit" not in str(caught.value)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_malformed_or_blocked_response_is_not_success(adapter: type) -> None:
    provider = adapter(
        ProviderConfig(provider_name="fixture", model="m"),
        client=FakeClient(FakeResponse()),
    )
    with pytest.raises(ProviderRuntimeError):
        provider.generate(ProviderRequest(prompt="x"))


@pytest.mark.parametrize("adapter,module,extra", (
    (OpenAIProvider, "openai", "openai"),
    (GeminiProvider, "google.genai", "gemini"),
    (OllamaProvider, "ollama", "ollama"),
))
def test_missing_optional_dependency(adapter: type, module: str, extra: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from agenttree.providers import _adapter

    def missing(name: str) -> None:
        assert name == module
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(_adapter, "import_module", missing)
    with pytest.raises(ProviderDependencyError, match=extra) as caught:
        adapter(ProviderConfig(provider_name="fixture", model="m"))
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_sdk_construction_parameters_no_generation(adapter: type, monkeypatch: pytest.MonkeyPatch) -> None:
    from agenttree.providers import _adapter

    constructed: list[dict[str, Any]] = []
    fake = FakeClient(sdk_result(adapter))

    def constructor(**kwargs: Any) -> FakeClient:
        constructed.append(kwargs)
        return fake

    monkeypatch.setattr(_adapter, "import_module", lambda _: SimpleNamespace(
        OpenAI=constructor, Client=constructor,
    ))
    options = (
        {"host": "http://example.invalid:11434"} if adapter is OllamaProvider
        else {"api_key": "fake-key"}
    )
    provider = adapter(ProviderConfig(provider_name="fixture", model="m"), **options)
    if adapter is OpenAIProvider:
        options = {**options, "max_retries": 0}
    expected = ({**options, "http_options": {"timeout": 15000, "retry_options": {"attempts": 1}}}
                if adapter is GeminiProvider else options)
    assert constructed == [expected]
    assert fake.calls == []
    assert "api_key" not in vars(provider.config)
    constructed.clear()
    adapter(ProviderConfig(provider_name="environment", model="m"))
    assert constructed == ([{"http_options": {"timeout": 15000, "retry_options": {"attempts": 1}}}]
                           if adapter is GeminiProvider else [{"max_retries": 0}] if adapter is OpenAIProvider else [{}])


def test_three_specialists_use_external_provider_bindings() -> None:
    registry = ProviderRegistry()
    bindings: dict[str, str] = {}
    specialists = []
    clients = []
    for index, adapter in enumerate(ADAPTERS):
        specialist = SpecialistAgent(name=f"Specialist {index}")
        client = FakeClient(sdk_result(adapter))
        provider = adapter(
            ProviderConfig(provider_name=f"provider-{index}", model="configured"),
            client=client,
        )
        registry.register(provider)
        bindings[specialist.id] = provider.name
        specialists.append(specialist)
        clients.append(client)
    executor = ProviderSpecialistExecutor(provider_registry=registry, provider_bindings=bindings)
    task = Task(objective="Parent")
    for specialist, client in zip(specialists, clients):
        subtask = Subtask(parent_task_id=task.id, manager_id="manager", objective="Work")
        result = executor.execute(task, subtask, specialist)
        assert result.success and result.output == "answer"
        assert result.metadata["provider"] == bindings[specialist.id]
        assert len(client.calls) == 1
        client.error = RuntimeError("offline failure")
        assert executor.execute(task, subtask, specialist).success is False


def test_public_imports_with_sdk_imports_forbidden() -> None:
    script = """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('openai', 'google', 'ollama'):
        raise AssertionError('SDK imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import agenttree
from agenttree.providers import OpenAIProvider, GeminiProvider, OllamaProvider
print('imports without SDKs passed')
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "imports without SDKs passed" in completed.stdout


def test_openai_custom_base_url_and_constructor_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from agenttree.providers import _adapter

    seen = []

    def constructor(**kwargs: Any) -> FakeClient:
        seen.append(kwargs)
        return FakeClient(sdk_result(OpenAIProvider))

    monkeypatch.setattr(_adapter, "import_module", lambda _: SimpleNamespace(OpenAI=constructor))
    OpenAIProvider(
        ProviderConfig(provider_name="custom", model="m"),
        api_key="fake", base_url="https://example.invalid/v1",
    )
    assert seen == [{"api_key": "fake", "base_url": "https://example.invalid/v1", "max_retries": 0}]
    error = ValueError("fake missing credentials")

    def broken(**kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(_adapter, "import_module", lambda _: SimpleNamespace(OpenAI=broken))
    with pytest.raises(ProviderConfigurationError) as caught:
        OpenAIProvider(ProviderConfig(provider_name="custom", model="m"))
    assert caught.value.__context__ is None
    assert "fake missing credentials" not in str(caught.value)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_plain_dict_raw_response_and_model_fallback(adapter: type) -> None:
    raw = vars(sdk_result(adapter)).copy()
    raw.pop("model", None)
    raw.pop("model_version", None)
    provider = adapter(
        ProviderConfig(provider_name="fixture", model="configured"),
        client=FakeClient(raw), keep_raw_response=True,
    )
    result = provider.generate(ProviderRequest(prompt="x"))
    assert result.model == "configured"
    assert result.raw_response == raw
    assert result.raw_response is not raw


def test_raw_response_redacts_sensitive_fields():
    from agenttree.providers._adapter import response

    raw = {"visible": "ok", "reasoning": "hidden text",
           "nested": {"authorization": "Bearer private-key", "value": 1}}
    result = response("ok", "m", "fixture", None,
                      ProviderRequest(prompt="x"), raw, True)
    assert result.raw_response == {"visible": "ok", "nested": {"value": 1}}
    assert "private-key" not in repr(result)
