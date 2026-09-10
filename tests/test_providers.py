"""Contracts, mock behavior, and registry tests for model providers."""

from types import MappingProxyType

import pytest

from agenttree.providers import (
    BaseProvider,
    MockProvider,
    ProviderConfig,
    ProviderRegistry,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
)


def test_provider_request_supports_structured_data_and_options() -> None:
    request = ProviderRequest(
        prompt="Create a result",
        system_prompt="Follow the supplied constraints",
        context={"inputs": [{"value": 3}], "flags": {"enabled": True}},
        metadata={"request_id": "request-1"},
        model="model-a",
        temperature=0.25,
        max_tokens=256,
    )
    assert request.prompt == "Create a result"
    assert request.context["inputs"][0]["value"] == 3
    assert request.metadata == {"request_id": "request-1"}
    assert (request.model, request.temperature, request.max_tokens) == (
        "model-a", 0.25, 256,
    )


def test_provider_request_defaults_are_independent() -> None:
    first = ProviderRequest(prompt="First")
    second = ProviderRequest(prompt="Second")
    first.context["value"] = 1
    first.metadata["label"] = "first"
    assert second.context == {}
    assert second.metadata == {}
    assert second.system_prompt is None


def test_provider_response_supports_normalized_and_raw_data() -> None:
    usage = ProviderUsage(input_tokens=4, output_tokens=6, total_tokens=10)
    raw = {"vendor_payload": {"id": "response-1"}}
    response = ProviderResponse(
        content="Result", provider="example", model="model-a", usage=usage,
        metadata={"finish_reason": "complete"}, raw_response=raw,
    )
    assert response.content == "Result"
    assert response.provider == "example"
    assert response.usage is usage
    assert response.raw_response is raw


def test_provider_usage_allows_partial_or_missing_counts() -> None:
    assert ProviderUsage() == ProviderUsage(None, None, None)
    assert ProviderUsage(input_tokens=7).total_tokens is None


def test_provider_config_is_non_secret_and_normalizes_name() -> None:
    config = ProviderConfig(
        provider_name="  Example  ", model="model-a", temperature=0.5,
        max_tokens=128, metadata={"region": "local"},
    )
    assert config.provider_name == "Example"
    assert config.model == "model-a"
    assert config.metadata == {"region": "local"}
    assert "api_key" not in config.__dataclass_fields__
    with pytest.raises(ValueError, match="provider_name"):
        ProviderConfig(provider_name="   ")


def test_base_provider_is_an_abstract_contract() -> None:
    with pytest.raises(TypeError):
        BaseProvider(ProviderConfig(provider_name="base"))  # type: ignore[abstract]

    class IncompleteProvider(BaseProvider):
        pass

    with pytest.raises(TypeError):
        IncompleteProvider(ProviderConfig(provider_name="incomplete"))


def test_custom_provider_implements_generate_contract_and_identity() -> None:
    class LocalProvider(BaseProvider):
        def generate(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(content=request.prompt, provider=self.name)

    config = ProviderConfig(provider_name="local", model="local-model")
    provider = LocalProvider(config)
    response = provider.generate(ProviderRequest(prompt="deterministic"))
    assert provider.name == "local"
    assert provider.config is config
    assert response == ProviderResponse(content="deterministic", provider="local")


def test_mock_provider_default_response_is_deterministic() -> None:
    provider = MockProvider()
    request = ProviderRequest(prompt="Any prompt")
    first = provider.generate(request)
    second = provider.generate(request)
    expected = ProviderResponse(
        content="Mock response", provider="mock", model="mock-model",
    )
    assert first == expected
    assert second == expected


def test_mock_provider_uses_request_model_and_configurable_content() -> None:
    usage = ProviderUsage(input_tokens=1, output_tokens=2, total_tokens=3)
    provider = MockProvider(
        ProviderConfig(provider_name="fixture", model="default-model"),
        response_content="Fixed output", usage=usage,
        response_metadata={"fixture": {"version": 1}},
    )
    response = provider.generate(
        ProviderRequest(prompt="Input", model="request-model"),
    )
    assert response == ProviderResponse(
        content="Fixed output", provider="fixture", model="request-model",
        usage=usage, metadata={"fixture": {"version": 1}},
    )


def test_mock_provider_can_return_complete_static_response() -> None:
    static = ProviderResponse(
        content="Static", provider="fixture", raw_response={"fixed": True},
    )
    provider = MockProvider(
        ProviderConfig(provider_name="fixture"), response=static,
    )
    assert provider.generate(ProviderRequest(prompt="Ignored")) is static


def test_mock_provider_records_requests_in_order_as_snapshots() -> None:
    provider = MockProvider()
    first = ProviderRequest(prompt="First")
    second = ProviderRequest(prompt="Second")
    provider.generate(first)
    snapshot = provider.requests
    provider.generate(second)
    assert snapshot == (first,)
    assert provider.requests == (first, second)


def test_mock_provider_rejects_non_requests() -> None:
    with pytest.raises(TypeError, match="ProviderRequest"):
        MockProvider().generate("prompt")  # type: ignore[arg-type]


def test_provider_registry_registers_retrieves_and_unregisters() -> None:
    registry = ProviderRegistry()
    provider = MockProvider(ProviderConfig(provider_name="  Fixture  "))
    registry.register(provider)
    assert registry.get("fixture") is provider
    assert registry.get(" FIXTURE ") is provider
    assert registry.unregister("Fixture") is provider
    assert registry.providers == ()
    with pytest.raises(KeyError):
        registry.get("fixture")


def test_provider_registry_rejects_duplicate_normalized_names() -> None:
    registry = ProviderRegistry()
    original = MockProvider(ProviderConfig(provider_name="Fixture"))
    registry.register(original)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(MockProvider(ProviderConfig(provider_name=" fixture ")))
    assert registry.providers == (original,)


@pytest.mark.parametrize("invalid", [object(), "provider", None])
def test_provider_registry_rejects_non_providers(invalid: object) -> None:
    registry = ProviderRegistry()
    with pytest.raises(TypeError, match="BaseProvider"):
        registry.register(invalid)  # type: ignore[arg-type]


def test_provider_registry_preserves_order_and_returns_snapshots() -> None:
    registry = ProviderRegistry()
    first = MockProvider(ProviderConfig(provider_name="First"))
    second = MockProvider(ProviderConfig(provider_name="Second"))
    registry.register(first)
    providers_snapshot = registry.providers
    names_snapshot = registry.names
    map_snapshot = registry.provider_map
    registry.register(second)
    assert providers_snapshot == (first,)
    assert names_snapshot == ("first",)
    assert isinstance(map_snapshot, MappingProxyType)
    assert map_snapshot == {"first": first}
    with pytest.raises(TypeError):
        map_snapshot["third"] = second  # type: ignore[index]
    assert registry.providers == (first, second)
    assert registry.names == ("first", "second")
