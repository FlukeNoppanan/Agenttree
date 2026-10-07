"""Representation tolerance must not change orchestration semantics."""
import json
import pytest
from agenttree.core.structured_output import parse_decision_output, review_output, required
from agenttree.core.decision_contracts import decision_format
from agenttree.providers import OllamaProvider, ProviderConfig, ProviderRequest
from agenttree.providers.exceptions import ProviderConfigurationError
from agenttree.exceptions import DecisionOutputError
from tests.test_structured_decision_repair import SequenceProvider, journal_context, manager_review, final_review, structured_events

@pytest.mark.parametrize("content", [
    '{"decision":"pass","feedback":"ok"}',
    '```json\n{"decision":"pass","feedback":"ok"}\n```',
    'Here is the result:\n```JSON\n{"decision":"pass","feedback":"ok"}\n```\nDone.',
])
def test_unambiguous_representation(content):
    assert review_output(parse_decision_output(content), "trusted").reviewer_id == "trusted"

@pytest.mark.parametrize("content", [
    'Maybe ask a manager', '{"decision":"pass","decision":"fail"}',
    '{"decision":"pass"} {"decision":"fail"}', '[{"decision":"pass"}]',
    '```json\n{"decision":"pass"}\n```\n{"decision":"fail"}',
    '{"decision":"pass","feedback":NaN}',
    '{"wrapper":{"decision":"pass","feedback":"ok"}}',
    '{"decision":"PASS","feedback":"ok"}',
])
def test_ambiguous_or_semantically_invalid_content_stays_invalid(content):
    with pytest.raises(DecisionOutputError):
        review_output(parse_decision_output(content), "trusted")

def test_required_diagnostic_does_not_invent_objective():
    with pytest.raises(DecisionOutputError) as caught:
        required({"notes":"potential secret"}, "objective")
    assert caught.value.reason_code == "missing_required_field"
    assert caught.value.field_name == "objective"

def test_two_independent_decisions_have_separate_safe_repair_identity():
    provider = SequenceProvider(['{"decision":"approve","feedback":"sentinel-secret"}', '{"decision":"pass","feedback":"accepted"}', '{"feedback":"missing decision"}', '{"decision":"approve","feedback":"still invalid"}'])
    with journal_context() as store:
        manager_review(provider)
        with pytest.raises(DecisionOutputError):
            final_review(provider)
    events=structured_events(store)
    repaired=next(e for e in events if e.event_type.endswith("repair.succeeded"))
    failed=next(e for e in events if e.event_type.endswith("repair.failed"))
    assert repaired.metadata["decision_id"] != failed.metadata["decision_id"]
    assert failed.metadata["reason_code"] == "invalid_enum"
    assert "sentinel-secret" not in json.dumps([e.metadata for e in events])
    assert len(provider.requests) == 4
    assert provider.requests[-1].context["structured_decision_repair"]["field"] == "decision"

def test_ollama_native_schema_maps_without_thinking_leakage():
    class Client:
        def generate(self, **kwargs):
            self.kwargs=kwargs
            return {"response":'{"decision":"pass","feedback":"ok"}',"model":"native-id","thinking":"sentinel-private"}
    client=Client()
    provider=OllamaProvider(ProviderConfig("local",model="native-id"),client=client,keep_raw_response=True)
    format=decision_format("final_review",{})
    response=provider.generate(ProviderRequest(prompt="Review",response_format=format))
    assert client.kwargs["format"] == format["json_schema"]["schema"]
    assert client.kwargs["think"] is False
    assert provider.capabilities.structured_output is True
    assert "sentinel-private" not in json.dumps(response.raw_response)
    provider.generate(ProviderRequest(prompt="JSON",response_format={"type":"json_object"}))
    assert client.kwargs["format"] == "json"
    with pytest.raises(ProviderConfigurationError):
        provider.generate(ProviderRequest(prompt="JSON",response_format={"type":"unknown"}))

def test_capability_schema_does_not_select_or_invent_targets():
    format=decision_format("triage",{"available_manager_capabilities":["comparison"]})
    item=format["json_schema"]["schema"]["properties"]["required_capabilities"]["items"]
    assert item["enum"] == ["comparison"]
    assert decision_format("triage",{"available_manager_capabilities":[]})["json_schema"]["schema"]["properties"]["required_capabilities"]["maxItems"] == 0
    assert decision_format("unknown",{}) is None


@pytest.mark.parametrize("format", ["json", {"type":"json_schema","json_schema":None}, {"type":"json_schema","json_schema":[]}, {"type":"json_schema","json_schema":{"schema":[]}}])
def test_invalid_native_format_configuration_is_rejected_before_generation(format):
    class Client:
        def generate(self, **kwargs):
            raise AssertionError("Invalid configuration must not contact Ollama")
    provider = OllamaProvider(ProviderConfig("local", model="native-id"), client=Client())
    with pytest.raises(ProviderConfigurationError):
        provider.generate(ProviderRequest(prompt="JSON", response_format=format))
