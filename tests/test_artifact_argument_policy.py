"""Bounded Artifact body policy; ordinary Tool authorization/budgets unchanged."""
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from agenttree import AgentTreeConfig, RootAgent
from agenttree.core import create_artifact_tool
from agenttree.core.artifacts import ArtifactSession, _active_artifact_session
from agenttree.core.artifact_store import InMemoryArtifactStore
from agenttree.providers import GeminiProvider, ProviderConfig, ProviderRequest, ProviderResponse
from agenttree.providers.exceptions import ProviderRuntimeError
from agenttree.tools import FunctionTool, ToolRegistry, ToolBindingRegistry, ToolCall
from agenttree.tools.runtime import (ToolSession, _validate_arguments, _argument_limit,
                                    _schema, _provider_response)

def setup(config=None):
    config = config or AgentTreeConfig()
    agent = RootAgent(name='Root')
    tool = create_artifact_tool()
    registry, bindings = ToolRegistry(), ToolBindingRegistry()
    registry.register(tool)
    bindings.assign(agent.id, tool.id)
    return tool, agent, ToolSession(registry, bindings, (agent,), config, 'artifact-policy')

def args(content):
    return dict(name='report.txt', type='text', content=content, operation='none')

@pytest.mark.parametrize('content', ['small', 'รายงานสำหรับมหาวิทยาลัย\n' * 400, 'a' * 999_999, 'a' * 1_000_000], ids=['small','thai-report','near-limit','at-limit'])
def test_artifact_body_small_normal_near_and_exact_boundary(content):
    tool, agent, session = setup()
    store = InMemoryArtifactStore()
    artifact_session = ArtifactSession('artifact-policy', store)
    token = _active_artifact_session.set(artifact_session)
    try:
        result = session.invoke(ToolCall('call', tool.name, agent.id, 'root', args(content), 'artifact-policy'))
    finally:
        _active_artifact_session.reset(token)
    assert result.success
    ref = artifact_session.refs()[0]
    assert artifact_session.read(ref.artifact_id) == content.encode('utf-8')
    assert ref.producer_agent_id == agent.id
    assert ref.size_bytes == len(content.encode('utf-8'))

@pytest.mark.parametrize('content', ['a' * 1_000_001, 'ก' * 333_334], ids=['ascii-over-limit','thai-over-limit'])
def test_artifact_above_utf8_body_bound_rejected_before_invocation(content):
    tool, agent, session = setup()
    result = session.invoke(ToolCall('call', tool.name, agent.id, 'root', args(content), 'artifact-policy'))
    assert not result.success and result.error_type == 'ToolArgumentValidationError'
    assert 'shorten' in result.error and '1000000' in result.error
    assert not session.metrics['executed']

def test_artifact_metadata_does_not_receive_body_allowance():
    tool, agent, session = setup()
    value = args('small') | {'name': 'SECRET_SENTINEL' * 2000}
    result = session.invoke(ToolCall('call', tool.name, agent.id, 'root', value, 'artifact-policy'))
    assert not result.success and 'remove unrelated context' in result.error
    assert 'SECRET_SENTINEL' not in result.error
    assert not session.metrics['executed']

@pytest.mark.parametrize('name', ['ordinary', 'create_artifact'])
def test_ordinary_tool_even_with_artifact_name_still_has_generic_bound(name):
    tool = FunctionTool(name=name, function=lambda content: 'should not execute')
    assert _argument_limit(tool, AgentTreeConfig()) == 16_384
    with pytest.raises(ValueError):
        _validate_arguments(tool, {'content': 'a' * 16_384}, AgentTreeConfig())

def test_configured_artifact_bound_and_escaped_json_string():
    tool, _, session = setup(AgentTreeConfig(max_artifact_bytes=20_000))
    content = '\x01' * 20_000
    value = json.dumps(args(content))
    assert len(value.encode()) > 100_000
    assert _validate_arguments(tool, value, session.config)['content'] == content
    with pytest.raises(ValueError, match='shorten'):
        _validate_arguments(tool, args(content + 'x'), session.config)

@pytest.mark.parametrize('extra', ['conversation_history', 'execution_trace', 'state'])
def test_artifact_schema_rejects_irrelevant_context(extra):
    tool, _, session = setup()
    with pytest.raises(ValueError):
        _validate_arguments(tool, args('report') | {extra: 'context'}, session.config)
    schema = _schema(tool)
    assert schema['additionalProperties'] is False
    assert schema['properties']['content']['type'] == ['string', 'null']
    assert 'JSON-encoded string' in schema['properties']['content']['description']
    assert schema['properties']['operation']['enum'] == ['none', 'create', 'modify', 'delete']

def gemini(arguments, name='create_artifact'):
    native = SimpleNamespace(parts=[SimpleNamespace(function_call=SimpleNamespace(name=name, args=arguments, id='call'), text=None)])
    chunk = SimpleNamespace(candidates=[SimpleNamespace(content=native)], usage_metadata=None, model_version='fake')
    models = SimpleNamespace(generate_content_stream=lambda **kwargs: iter([chunk]))
    return GeminiProvider(ProviderConfig('gemini', model='fake'), client=SimpleNamespace(models=models))

def test_gemini_thai_size_uses_utf8_not_ascii_escapes():
    value = args('ก' * 4000)
    assert len(json.dumps(value).encode()) > 16_384
    assert len(json.dumps(value, ensure_ascii=False).encode()) < 16_384
    chunks = list(gemini(value).generate_stream(ProviderRequest('test')))
    assert chunks[-1].response.tool_calls[0]['function']['arguments'] == value

def test_gemini_and_session_stream_use_trusted_artifact_transport_policy():
    tool, _, session = setup()
    value = args('ก' * 9000)
    request = ProviderRequest('test', tools=session.definitions(next(iter(session.agents))),
        tool_argument_limits={tool.name: _argument_limit(tool, session.config)})
    result = _provider_response(gemini(value), request, True)
    assert result.tool_calls[0]['function']['arguments'] == value
    with pytest.raises(ProviderRuntimeError, match='size limit'):
        list(gemini(value, 'unassigned').generate_stream(request))

def test_gemini_transport_above_explicit_bound_rejected():
    request = ProviderRequest('test', tool_argument_limits={'create_artifact': 100})
    with pytest.raises(ProviderRuntimeError, match='size limit'):
        list(gemini(args('a' * 101)).generate_stream(request))

def test_oversized_ordinary_session_tool_is_denied_and_not_invoked():
    tool, agent, session = setup()
    ordinary = FunctionTool(name='ordinary', function=lambda content: pytest.fail('unsafe invocation'))
    session.registry.register(ordinary)
    session.bindings.assign(agent.id, ordinary.id)
    result = session.invoke(ToolCall('call', ordinary.name, agent.id, 'root', {'content':'a'*16_384}, 'artifact-policy'))
    assert not result.success and result.error_type == 'ToolArgumentValidationError'
    assert not session.metrics['executed']

def test_artifact_model_continuation_retains_real_body_and_small_result():
    from agenttree.providers import BaseProvider, ProviderCapabilities
    tool, agent, session = setup()
    content = 'report\n' * 4000
    class Provider(BaseProvider):
        requests = []
        @property
        def capabilities(self): return ProviderCapabilities(tool_calling=True)
        def generate(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return ProviderResponse('', self.name, tool_calls=({'id':'call','function':{'name':tool.name,'arguments':args(content)}},))
            assert request.tool_history[0]['calls'][0]['arguments']['content'] == content
            assert request.tool_history[0]['results'][0]['success'] is True
            assert set(request.tool_history[0]['results'][0]['output']) == {'artifact_id','sha256'}
            return ProviderResponse('continued', self.name)
    provider = Provider(ProviderConfig('test'))
    token = _active_artifact_session.set(ArtifactSession('artifact-policy', InMemoryArtifactStore()))
    try:
        assert session.generate(agent.id, provider, ProviderRequest('test'), 'root').content == 'continued'
    finally:
        _active_artifact_session.reset(token)
    assert provider.requests[0].tool_argument_limits[tool.name] > 16_384

def test_compatible_stream_respects_explicit_artifact_transport_bound():
    from io import BytesIO
    from agenttree.providers import OpenAICompatibleProvider
    from agenttree.providers.exceptions import MalformedProviderResponseError
    payload = json.dumps(args('report' * 4000))
    delta = {'choices':[{'delta':{'tool_calls':[{'index':0,'id':'call','function':{'name':'create_artifact','arguments':payload}}]},'finish_reason':'tool_calls'}]}
    wire = b'data: ' + json.dumps(delta).encode() + b'\n\n' + b'data: [DONE]\n\n'
    class Opener:
        def open(self, request, timeout=None): return BytesIO(wire)
    provider = OpenAICompatibleProvider(ProviderConfig('custom', model='fake'), base_url='http://localhost:1234', streaming=True)
    provider._opener=Opener()
    request = ProviderRequest('test', tool_argument_limits={'create_artifact':50_000})
    assert list(provider.generate_stream(request))[-1].response.tool_calls[0]['function']['arguments'] == payload
    with pytest.raises(MalformedProviderResponseError, match='size limit'):
        list(provider.generate_stream(ProviderRequest('test')))
