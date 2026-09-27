"""Offline HTTP contracts for compatible providers and discovery."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from time import sleep
from types import SimpleNamespace

import pytest

from agenttree.providers import (
    CerebrasProvider, GeminiProvider, GroqProvider, OpenAICompatibleProvider,
    OpenRouterProvider, ProviderConfig, ProviderRequest,
    ProviderAuthenticationError, ProviderRateLimitError,
    ProviderTimeoutError, ProviderUnavailableError,
    MalformedProviderResponseError, create_provider,
)
from agenttree.providers.exceptions import ProviderConfigurationError


@contextmanager
def server():
    state = {"calls": [], "status": 200, "body": None, "delay": 0.0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length) if length else b""
            state["calls"].append((self.command, self.path,
                                   self.headers.get("Authorization"), payload,
                                   self.headers.get("User-Agent")))
            if state["delay"]:
                sleep(state["delay"])
            status = state["status"]
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if status == 429:
                self.send_header("Retry-After", "7")
            if status == 302:
                self.send_header("Location", state.get("redirect", "https://example.invalid/steal"))
            self.end_headers()
            body = state["body"]
            if body is None:
                body = ({"data": [{"id": "provider/model:free", "name": "Text",
                                   "context_length": 1234, "pricing": {"prompt": "0"},
                                   "supported_parameters": ["temperature"],
                                   "input_modalities": ["text"], "output_modalities": ["text"]}]}
                        if self.path.endswith("/models") else
                        {"model": "provider/model:free", "choices": [{"message": {"content": "Hello"},
                          "finish_reason": "stop"}], "usage": {"prompt_tokens": 2,
                          "completion_tokens": 3, "total_tokens": 5}})
            try:
                self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())
            except BrokenPipeError:
                pass

        do_GET = respond
        do_POST = respond

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/v1", state
    finally:
        http.shutdown()
        thread.join()
        http.server_close()


def test_custom_chat_models_cache_validation_and_secret_redaction():
    with server() as (url, state):
        provider = OpenAICompatibleProvider(
            ProviderConfig("custom", model="provider/model:free"),
            base_url=url, api_key="private-key", timeout=1)
        assert provider.provider_type == "openai-compatible"
        assert "private-key" not in repr(provider)
        models = provider.list_models()
        assert provider.list_models() == models
        assert models[0].id == "provider/model:free"
        assert models[0].context_window == 1234
        assert models[0].metadata["pricing"] == {"prompt": "0"}
        assert provider.validate_connection().valid
        assert [call[1] for call in state["calls"]] == ["/v1/models", "/v1/models"]
        response = provider.generate(ProviderRequest(prompt="Hello", provider_options={}))
        assert response.content == "Hello" and response.model == "provider/model:free"
        assert response.usage.total_tokens == 5 and response.finish_reason == "stop"
        assert response.raw_response is None
        assert "private-key" not in repr(response)
        assert state["calls"][-1][2] == "Bearer private-key"
        assert state["calls"][-1][4] == "AgentTree"
        assert json.loads(state["calls"][-1][3])["messages"][-1]["content"] == "Hello"
        provider.list_models(refresh=True)
        assert len([call for call in state["calls"] if call[1] == "/v1/models"]) == 3


@pytest.mark.parametrize("status,error", [(401, ProviderAuthenticationError),
    (429, ProviderRateLimitError), (500, ProviderUnavailableError)])
def test_http_errors_are_normalized_without_secret(status, error):
    with server() as (url, state):
        provider = OpenAICompatibleProvider(ProviderConfig("custom", model="m"),
                                             base_url=url, api_key="private-key")
        state["status"] = status
        with pytest.raises(error) as caught:
            provider.generate(ProviderRequest(prompt="x"))
        assert "private-key" not in str(caught.value)
        assert caught.value.__context__ is None
        if status == 429:
            assert caught.value.retry_after == 7
        validation = provider.validate_connection()
        assert not validation.valid and validation.error_category == error.__name__
        assert "private-key" not in repr(validation)


def test_timeout_invalid_json_and_redirect_are_safe():
    with server() as (url, state):
        provider = OpenAICompatibleProvider(ProviderConfig("custom", model="m"),
                                             base_url=url, api_key="private-key", timeout=0.05)
        state["delay"] = 0.15
        with pytest.raises(ProviderTimeoutError):
            provider.generate(ProviderRequest(prompt="x"))
        state["delay"] = 0
        state["body"] = b"not JSON"
        with pytest.raises(MalformedProviderResponseError):
            provider.generate(ProviderRequest(prompt="x"))
        with server() as (redirect_url, destination):
            state["status"] = 302
            state["redirect"] = redirect_url + "/chat/completions"
            with pytest.raises(Exception):
                provider.generate(ProviderRequest(prompt="x"))
            assert destination["calls"] == []
        assert len(state["calls"]) == 3


def test_failed_discovery_is_not_cached_and_known_model_capability_is_checked():
    with server() as (url, state):
        provider = OpenAICompatibleProvider(ProviderConfig("custom", model="m"), base_url=url)
        state["status"] = 500
        with pytest.raises(ProviderUnavailableError):
            provider.list_models()
        state["status"] = 200
        state["body"] = {"data": [{"id": "m", "capabilities": {"structured_outputs": False}}]}
        assert provider.list_models()[0].capabilities.structured_output is False
        with pytest.raises(ProviderConfigurationError):
            provider.generate(ProviderRequest(prompt="x", response_format={"type": "json_object"}))
        assert len(state["calls"]) == 2


def test_response_discards_reasoning_and_keeps_safe_tool_call_fields():
    with server() as (url, state):
        state["body"] = {"model": "m", "choices": [{"message": {
            "content": "Visible answer", "reasoning": "private reasoning",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {
                "name": "lookup", "arguments": "{}"}, "hidden": "secret"}]},
            "finish_reason": "tool_calls"}]}
        provider = OpenAICompatibleProvider(ProviderConfig("custom", model="m"), base_url=url)
        result = provider.generate(ProviderRequest(prompt="x"))
        assert result.content == "Visible answer"
        assert result.tool_calls[0]["function"]["name"] == "lookup"
        assert "private reasoning" not in repr(result)
        assert "hidden" not in repr(result)


def test_custom_configuration_and_options_are_bounded():
    with server() as (url, state):
        for bad in ("http://example.com/v1", "https://user:pass@example.com/v1",
                    "https://example.com/v1?key=secret"):
            with pytest.raises(ProviderConfigurationError):
                OpenAICompatibleProvider(ProviderConfig("x"), base_url=bad)
        with pytest.raises(ProviderConfigurationError):
            OpenAICompatibleProvider(ProviderConfig("x"), base_url=url,
                                     headers={"Authorization": "Bearer secret"})
        with pytest.raises(ProviderConfigurationError):
            OpenAICompatibleProvider(ProviderConfig("x"), base_url=url,
                                     allowed_options=frozenset({"model"}))
        provider = OpenAICompatibleProvider(ProviderConfig("x", model="m"), base_url=url)
        with pytest.raises(ProviderConfigurationError):
            provider.generate(ProviderRequest(prompt="x", provider_options={"model": "override"}))
        assert state["calls"] == []
        disabled = OpenAICompatibleProvider(ProviderConfig("off", model="m"),
                                             base_url=url, model_discovery=False)
        assert not disabled.capabilities.model_discovery
        assert disabled.validate_connection().error_category == "unsupported"


def test_first_class_endpoints_and_factory(monkeypatch):
    expected = [(GroqProvider, "groq", "https://api.groq.com/openai/v1"),
                (OpenRouterProvider, "openrouter", "https://openrouter.ai/api/v1"),
                (CerebrasProvider, "cerebras", "https://api.cerebras.ai/v1")]
    for cls, kind, url in expected:
        provider = create_provider(kind, ProviderConfig(kind, model="m"), api_key="fake")
        assert isinstance(provider, cls) and provider.provider_type == kind
        assert provider._base_url == url
        assert provider.capabilities.model_discovery
        assert "fake" not in repr(provider)
    with pytest.raises(ValueError):
        create_provider("unknown", ProviderConfig("x"))


def test_native_gemini_discovery_cache_and_validation():
    class Models:
        def __init__(self):
            self.calls = 0

        def list(self):
            self.calls += 1
            return [SimpleNamespace(name="models/gemini-test", display_name="Test",
                 input_token_limit=1000, supported_actions=("generateContent",))]

    models = Models()
    provider = GeminiProvider(ProviderConfig("gemini"),
                              client=SimpleNamespace(models=models))
    discovered = provider.list_models()
    assert discovered[0].id == "models/gemini-test"
    assert discovered[0].capabilities.chat is True
    assert provider.list_models() == discovered and models.calls == 1
    assert provider.validate_connection().valid and models.calls == 2
    assert provider.list_models(refresh=True) == discovered and models.calls == 3
