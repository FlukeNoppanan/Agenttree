"""Regressions for real Gemini runtime adapter boundary failures."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from agenttree.providers import GeminiProvider, ProviderConfig, ProviderRequest, ProviderTimeoutError
from agenttree.providers.exceptions import normalize_provider_error


def test_gemini_function_declaration_uses_json_schema_and_preserves_input():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": False, "required": ["name"]}
    tool = {"name": "create_artifact", "description": "Safe artifact", "parameters": schema}
    original = deepcopy(tool)
    provider = GeminiProvider(ProviderConfig("gemini", model="fixture"), client=SimpleNamespace())
    _, _, options = provider._generation_args(ProviderRequest(prompt="work", tools=(tool,)))
    declaration = options["tools"][0]["function_declarations"][0]
    assert declaration["parameters_json_schema"] == schema
    assert "parameters" not in declaration
    assert tool == original


def test_gemini_native_json_schema_passes_through_unchanged():
    declaration = {"name": "safe", "parameters_json_schema": {"type": "object", "properties": {}}}
    provider = GeminiProvider(ProviderConfig("gemini", model="fixture"), client=SimpleNamespace())
    _, _, options = provider._generation_args(ProviderRequest(prompt="work", tools=(declaration,)))
    assert options["tools"][0]["function_declarations"][0] == declaration


@pytest.mark.parametrize("name", ["ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout"])
def test_httpx_timeout_is_normalized_without_leaking_sdk_message(name):
    # Optional SDK-independent recognition, even without importing httpx.
    base = type("TimeoutException", (Exception,), {"__module__": "httpx"})
    error_type = type(name, (base,), {"__module__": "httpx"})
    normalized = normalize_provider_error(error_type("SECRET_SENTINEL"))
    assert isinstance(normalized, ProviderTimeoutError)
    assert str(normalized) == "Provider request timed out"


@pytest.mark.parametrize("value", [True, False, "false"])
def test_mcp_primitive_union_schema_accepts_declared_types(value):
    from agenttree.tools.runtime import _validate_schema
    _validate_schema({"nextThoughtNeeded": value}, {"type": "object", "properties": {
        "nextThoughtNeeded": {"type": ["boolean", "string"]}}, "required": ["nextThoughtNeeded"]})


@pytest.mark.parametrize("value", [1, None, {}])
def test_mcp_primitive_union_schema_rejects_other_types(value):
    from agenttree.tools.runtime import _validate_schema
    with pytest.raises(ValueError, match="invalid type"):
        _validate_schema(value, {"type": ["boolean", "string"]})


@pytest.mark.parametrize("schema_type", [[], ["unknown"], [["boolean"]], 42])
def test_malformed_mcp_union_schema_fails_closed_without_typeerror(schema_type):
    from agenttree.tools.runtime import _validate_schema
    with pytest.raises(ValueError, match="Unsupported"):
        _validate_schema(True, {"type": schema_type})
