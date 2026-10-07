"""Shared answer extraction preserves representation flexibility and strict semantics."""
import json
import pytest
from agenttree.core.structured_output import normalize_decision_output, required, review_output
from agenttree.exceptions import DecisionOutputError
from agenttree.providers._adapter import final_content, _redact_raw
from agenttree.providers.compatible import OpenAICompatibleProvider
from agenttree.providers import ProviderConfig, ProviderRequest
from agenttree.providers.exceptions import ProviderInvalidRequestError, MalformedProviderResponseError, normalize_provider_error

@pytest.mark.parametrize('value', [
 {'decision':'pass','feedback':'OK'},
 'Result: {"decision":"pass","feedback":"OK"} Done.',
 {'parts':[{'type':'text','text':'{"decision":"pass",'},{'type':'text','text':'"feedback":"OK"}'}]},
 {'function':{'name':'decision','arguments':'{"decision":"pass","feedback":"OK"}'}},
 {'arguments':{'decision':'pass','feedback':'OK'}},
])
def test_supported_shapes(value):
 data,method=normalize_decision_output(value)
 assert review_output(data,'trusted').decision.value=='pass'
 assert method

@pytest.mark.parametrize('value', [
 'Result: {"decision":"pass"} or {"decision":"fail"}',
 '<think>{"decision":"pass","feedback":"private"}</think>',
 '{"delegate":true,"delegate":false}',
 '{"decision":"approve","feedback":"OK"}',
])
def test_ambiguity_and_semantics_remain_invalid(value):
 with pytest.raises(DecisionOutputError):
  review_output(normalize_decision_output(value)[0],'trusted')

def test_absent_required_semantics_not_invented():
 with pytest.raises(DecisionOutputError) as caught:
  required(normalize_decision_output('Answer: {"notes":"hello"}')[0],'objective')
 assert caught.value.field_name=='objective'

def test_parts_discard_hidden_reasoning():
 parts=[{'type':'reasoning','text':'private-sentinel'},{'thought':True,'text':'private-sentinel'},{'type':'text','text':'final'}]
 assert final_content(parts)=='final'
 assert 'private-sentinel' not in json.dumps(_redact_raw(parts))

def test_optional_feature_fallback_never_changes_model():
 provider=OpenAICompatibleProvider(ProviderConfig('neutral',model='native'),base_url='https://example.test')
 calls=[]
 def send(path,payload,**kwargs):
  calls.append(dict(payload))
  if len(calls)==1: raise ProviderInvalidRequestError('safe',unsupported_parameter='response_format')
  return {'choices':[{'message':{'content':[{'type':'text','text':'final'}]},'finish_reason':'stop'}]}
 provider._json_request=send
 response=provider.generate(ProviderRequest(prompt='input',response_format={'type':'json_object'}))
 assert response.content=='final'
 assert len(calls)==2 and calls[0]['model']==calls[1]['model']=='native'
 assert 'response_format' not in calls[1]
 assert response.metadata['request_feature_fallback']=='response_format'

def test_unknown_request_rejection_does_not_retry():
 provider=OpenAICompatibleProvider(ProviderConfig('neutral',model='native'),base_url='https://example.test')
 calls=[]
 def send(*args,**kwargs):
  calls.append(1);raise ProviderInvalidRequestError('safe')
 provider._json_request=send
 with pytest.raises(ProviderInvalidRequestError): provider.generate(ProviderRequest(prompt='input'))
 assert len(calls)==1

def test_incomplete_final_has_safe_diagnostics_only():
 provider=OpenAICompatibleProvider(ProviderConfig('neutral',model='native'),base_url='https://example.test')
 provider._json_request=lambda *a,**kw: {'choices':[{'message':{'content':None,'reasoning':'private-sentinel'},'finish_reason':'length'}]}
 with pytest.raises(MalformedProviderResponseError) as caught: provider.generate(ProviderRequest(prompt='input'))
 assert caught.value.diagnostics['finish_reason']=='length'
 assert caught.value.diagnostics['reasoning_present'] is True
 assert 'private-sentinel' not in json.dumps(caught.value.diagnostics)

def test_feature_error_keeps_only_allowlisted_parameter():
 class Error(Exception):
  status_code=400
  body={'error':{'code':'unsupported_parameter','param':'temperature','message':'private-key'}}
 error=normalize_provider_error(Error())
 assert error.unsupported_parameter=='temperature'
 assert 'private-key' not in str(error)
