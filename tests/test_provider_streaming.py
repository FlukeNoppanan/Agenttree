"""Offline stream normalization and atomic tool-call assembly."""

from contextlib import contextmanager
from io import BytesIO
from types import SimpleNamespace

import pytest

from agenttree.providers import (
    BaseProvider, OpenAICompatibleProvider, ProviderCapabilities,
    ProviderConfig, ProviderRequest, ProviderResponse, ProviderStreamChunk,
    GeminiProvider,
)
from agenttree.providers.exceptions import MalformedProviderResponseError
from agenttree.tools.runtime import _provider_response


class StreamingProvider(BaseProvider):
    @property
    def capabilities(self):
        return ProviderCapabilities(streaming=True)

    def generate(self, request):
        raise AssertionError("synchronous path used")

    def generate_stream(self, request):
        yield ProviderStreamChunk(delta_text="partial")
        yield ProviderStreamChunk(response=ProviderResponse("complete", self.name))


def test_stream_consumes_complete_response():
    provider = StreamingProvider(ProviderConfig("fake"))
    assert _provider_response(provider, ProviderRequest("test"), True).content == "complete"


def test_stream_rejects_missing_final_response():
    provider = StreamingProvider(ProviderConfig("fake"))
    provider.generate_stream = lambda request: iter((ProviderStreamChunk(delta_text="partial"),))
    with pytest.raises(ValueError, match="final response"):
        _provider_response(provider, ProviderRequest("test"), True)


def test_stream_rejects_oversized_text_and_tool_arguments():
    provider = StreamingProvider(ProviderConfig("fake"))
    provider.generate_stream = lambda request: iter((ProviderStreamChunk(delta_text="12345"),))
    with pytest.raises(ValueError, match="text limit"):
        _provider_response(provider, ProviderRequest("test"), True, max_bytes=4)
    provider.generate_stream = lambda request: iter((ProviderStreamChunk(
        response=ProviderResponse("", "fake", tool_calls=({"function": {
            "name": "tool", "arguments": "12345"}},))),))
    with pytest.raises(ValueError, match="argument limit"):
        _provider_response(provider, ProviderRequest("test"), True, max_tool_bytes=4)


def test_stream_failure_never_returns_partial_response():
    provider = StreamingProvider(ProviderConfig("fake"))
    def fail(request):
        yield ProviderStreamChunk(delta_text="visible but incomplete")
        raise OSError("stream disconnected")
    provider.generate_stream = fail
    with pytest.raises(OSError, match="disconnected"):
        _provider_response(provider, ProviderRequest("test"), True)


class FakeOpener:
    def __init__(self, body):
        self.body = body

    def open(self, request, timeout=None):
        return BytesIO(self.body)


def test_compatible_stream_assembles_tool_arguments():
    provider = OpenAICompatibleProvider(ProviderConfig("custom", model="m"),
                                        base_url="http://127.0.0.1:1234", streaming=True)
    provider._opener = FakeOpener(b''.join((
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"abc","function":{"name":"do_","arguments":"{\\\"x\\\":"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"work","arguments":"1}"}}]},"finish_reason":"tool_calls"}]}\n\n',
        b'data: [DONE]\n\n',
    )))
    chunks = list(provider.generate_stream(ProviderRequest("test")))
    assert len(chunks) == 1
    call = chunks[-1].response.tool_calls[0]
    assert call["function"] == {"name": "do_work", "arguments": '{"x":1}'}


def test_compatible_stream_requires_done_marker():
    provider = OpenAICompatibleProvider(ProviderConfig("custom", model="m"),
                                        base_url="http://127.0.0.1:1234", streaming=True)
    provider._opener = FakeOpener(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
    with pytest.raises(MalformedProviderResponseError, match="before completion"):
        list(provider.generate_stream(ProviderRequest("test")))


def test_gemini_stream_preserves_complete_function_call():
    class Models:
        def generate_content_stream(self, **kwargs):
            assert kwargs["model"] == "gemini-test"
            yield SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(
                parts=[SimpleNamespace(text="Working", function_call=None)]))],
                usage_metadata=None, model_version="gemini-test")
            yield SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(
                parts=[SimpleNamespace(text=None, function_call=SimpleNamespace(
                    id="call-1", name="write", args={"path": "health.py"}))]))],
                usage_metadata=SimpleNamespace(prompt_token_count=3,
                    candidates_token_count=2, total_token_count=5),
                model_version="gemini-test")
    provider = GeminiProvider(ProviderConfig("gemini", model="gemini-test"),
                              client=SimpleNamespace(models=Models()))
    chunks = list(provider.generate_stream(ProviderRequest("test", tools=({
        "name": "write", "parameters": {}},))))
    assert chunks[0].delta_text == "Working"
    response = chunks[-1].response
    assert response.tool_calls[0]["function"] == {
        "name": "write", "arguments": {"path": "health.py"}}
    assert response.usage.total_tokens == 5
