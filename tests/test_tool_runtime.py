"""Offline model to tool to model paths through the public SDK."""

import json
from time import sleep

import pytest

from agenttree import AgentTree, AgentTreeConfig, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (ProviderManagerReviewer, ProviderRootPlanner,
                            ProviderRootSynthesizer, ProviderTaskDecomposer,
                            RuleBasedTaskTriage, StaticFinalReviewer,
                            StaticManagerReviewer, StaticTaskDecomposer)
from agenttree.models import SubtaskTemplate
from agenttree.models import ReviewDecision
from agenttree.providers import (BaseProvider, ProviderCapabilities, ProviderConfig,
                                 ProviderRequest, ProviderResponse)
from agenttree.providers.compatible import OpenAICompatibleProvider
from agenttree.providers.gemini import GeminiProvider
from agenttree.providers.groq import GroqProvider
from agenttree.providers.openrouter import OpenRouterProvider
from agenttree.providers.cerebras import CerebrasProvider
from agenttree.tools import FunctionTool
from agenttree.tools.runtime import ToolSession
from agenttree.tools import ToolRegistry, ToolBindingRegistry, ToolCall
from agenttree.tools.mcp import MCPTool, MCPToolDefinition, MockMCPClient


class ScriptProvider(BaseProvider):
    def __init__(self, responses, name="script"):
        super().__init__(ProviderConfig(name, model="fixture"))
        self.responses = list(responses)
        self.requests = []

    @property
    def capabilities(self):
        return ProviderCapabilities(tool_calling=True)

    def generate(self, request):
        self.requests.append(request)
        item = self.responses.pop(0)
        return ProviderResponse(content=item[0], provider=self.name,
                                tool_calls=item[1] if len(item) > 1 else ())


def call(name="lookup", args='{"value": 3}', call_id="call-1"):
    return {"id": call_id, "function": {"name": name, "arguments": args}}


def tree(**kwargs):
    options = dict(root_agent=RootAgent(name="Root"),
                   triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
                   decomposer=StaticTaskDecomposer((SubtaskTemplate("Work", ("work",)),)),
                   manager_reviewer=StaticManagerReviewer(),
                   final_reviewer=StaticFinalReviewer())
    options.update(kwargs)
    return AgentTree(**options)


def pair(target):
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    specialist = SpecialistAgent(name="Worker", capabilities=("work",))
    target.register_manager(manager)
    target.register_specialist(manager, specialist)
    return manager, specialist


def bind(target, agent, provider, tool):
    target.register_provider(provider)
    target.register_tool(tool)
    target.bind_provider(agent, provider)
    target.bind_tool(agent, tool)


def test_root_direct_tool_roundtrip_and_trace():
    provider = ScriptProvider([("", (call(),)), ('{"delegate":false,"direct_output":"value is 6"}',)])
    target = tree(root_planner=ProviderRootPlanner(provider))
    tool = FunctionTool(name="lookup", function=lambda value: value * 2)
    bind(target, target.root_agent, provider, tool)
    result = target.run(Task(objective="Find value"))
    assert result.success and result.final_output == "value is 6"
    assert provider.requests[0].tools[0]["name"] == "lookup"
    assert provider.requests[1].tool_history[0]["results"][0]["output"] == 6
    assert provider.requests[1].tool_history[0]["results"][0]["call_id"] == "call-1"
    assert [event.event_type for event in result.trace.events if ".tool." in event.event_type] == [
        "root.tool.requested", "root.tool.authorized", "root.tool.started", "root.tool.completed"]
    assert result.metadata["tool_metrics"]["completed"] == 1


def test_root_synthesis_tool_roundtrip():
    provider = ScriptProvider([('{"delegate":true}',), ("", (call(),)),
                               ("Final answer with value 6",)])
    target = tree(root_synthesizer=ProviderRootSynthesizer(provider))
    _, specialist = pair(target)
    worker = ScriptProvider([("Worker output",)], name="worker")
    target.register_provider(worker)
    target.bind_provider(specialist, worker)
    tool = FunctionTool(name="lookup", function=lambda value: value * 2)
    bind(target, target.root_agent, provider, tool)
    result = target.run(Task(objective="Work"))
    assert result.final_output == "Final answer with value 6"
    assert "root.tool.completed" in [event.event_type for event in result.trace.events]
    assert result.trace.events[-1].event_type == "execution.completed"


def test_manager_decomposition_and_review_use_same_runtime():
    decomposition = ScriptProvider([("", (call(),)),
        ('{"subtasks":[{"objective":"Work","required_capabilities":["work"]}]}',)])
    reviewer = ScriptProvider([("", (call(),)), ('{"decision":"pass","feedback":"ok"}',)], name="review")
    target = tree(decomposer=ProviderTaskDecomposer(decomposition),
                  manager_reviewer=ProviderManagerReviewer(reviewer))
    manager, specialist = pair(target)
    worker = ScriptProvider([("finished",)], name="worker")
    for provider in (decomposition, reviewer, worker):
        target.register_provider(provider)
    target.bind_provider(manager, decomposition)
    target.bind_provider(specialist, worker)
    tool = FunctionTool(name="lookup", function=lambda value: value * 2)
    target.register_tool(tool)
    target.bind_tool(manager, tool)
    # Manager routing overrides the reviewer's fallback provider.
    decomposition.responses.extend([("", (call(call_id="review-call"),)),
        ('{"decision":"pass","feedback":"ok"}',)])
    result = target.run(Task(objective="Work"))
    assert result.success
    assert len([e for e in result.trace.events if e.event_type == "manager.tool.completed"]) == 2
    assert len(decomposition.requests) == 4


def test_specialist_tool_failure_can_recover_without_tree_failure():
    provider = ScriptProvider([("", (call("lookup", "{bad"),)), ("I can answer without it",)])
    target = tree()
    _, specialist = pair(target)
    tool = FunctionTool(name="lookup", function=lambda value: value * 2)
    bind(target, specialist, provider, tool)
    result = target.run(Task(objective="Work"))
    assert result.success
    assert provider.requests[1].tool_history[0]["results"][0]["error"] == "ToolArgumentValidationError"
    assert any(e.event_type == "specialist.tool.denied" for e in result.trace.events)


def test_unassigned_hallucinated_tool_is_denied():
    provider = ScriptProvider([("", (call("terminal"),)), ("safe answer",)])
    target = tree()
    _, specialist = pair(target)
    target.register_provider(provider)
    target.bind_provider(specialist, provider)
    dangerous = FunctionTool(name="terminal", function=lambda value: pytest.fail("executed"))
    target.register_tool(dangerous)
    result = target.run(Task(objective="Work"))
    assert result.success
    assert provider.requests[1].tool_history[0]["results"][0]["error"] == "ToolNotAssigned"
    assert result.metadata["tool_metrics"]["denied"] == 1


@pytest.mark.parametrize("arguments,error", [
    ('{"value":"bad"}', "ToolArgumentValidationError"),
    ('{"value":1,"extra":2}', "ToolArgumentValidationError"),
    ('{"value":null}', "ToolArgumentValidationError"),
    ('{"value":1', "ToolArgumentValidationError"),
    ('{"value":1,"value":2}', "ToolArgumentValidationError"),
])
def test_arguments_rejected_before_function(arguments, error):
    calls = []
    def lookup(value: int):
        calls.append(value)
    tool = FunctionTool(name="lookup", function=lookup)
    registry, bindings = ToolRegistry(), ToolBindingRegistry()
    registry.register(tool)
    root = RootAgent(name="Root")
    bindings.assign(root.id, tool.id)
    session = ToolSession(registry, bindings, (root,), AgentTreeConfig(), "execution")
    outcome = session.invoke(ToolCall("c", "lookup", root.id, "root", arguments, "execution"))
    assert outcome.error_type == error and not calls


def test_timeout_result_size_redaction_and_budget():
    root = RootAgent(name="Root")
    registry, bindings = ToolRegistry(), ToolBindingRegistry()
    slow = FunctionTool(name="slow", function=lambda: sleep(0.2))
    large = FunctionTool(name="large", function=lambda: "x" * 100)
    secret = FunctionTool(name="secret", function=lambda: {"api_key": "topsecret", "answer": 1})
    for tool in (slow, large, secret):
        registry.register(tool)
        bindings.assign(root.id, tool.id)
    session = ToolSession(registry, bindings, (root,), AgentTreeConfig(
        tool_timeout=0.01, max_tool_result_bytes=50, max_tool_calls=3), "execution")
    def invoke(name, index):
        return session.invoke(ToolCall(str(index), name, root.id, "root", {}, "execution"))
    assert invoke("slow", 1).error_type == "ToolTimeout"
    assert invoke("large", 2).error_type == "ToolResultTooLarge"
    assert invoke("secret", 3).output == {"api_key": "[REDACTED]", "answer": 1}
    assert invoke("secret", 4).error_type == "ToolBudgetExceeded"
    assert "topsecret" not in repr(session.events)


def test_openai_compatible_roundtrip_payload():
    class FakeCompatible(OpenAICompatibleProvider):
        def __init__(self):
            super().__init__(ProviderConfig("compatible", model="fixture"),
                             base_url="http://localhost:1234", model_discovery=False)
            self.payloads = []
        def _json_request(self, path, payload=None, **kwargs):
            self.payloads.append(payload)
            message = ({"content": None, "tool_calls": [{"id": "abc", "type": "function",
                "function": {"name": "lookup", "arguments": '{"value":3}'}}]}
                       if len(self.payloads) == 1 else {"content": "done"})
            return {"model": "fixture", "choices": [{"message": message}]}
    provider = FakeCompatible()
    root = RootAgent(name="Root")
    tool = FunctionTool(name="lookup", function=lambda value: value * 2)
    registry, bindings = ToolRegistry(), ToolBindingRegistry()
    registry.register(tool); bindings.assign(root.id, tool.id)
    session = ToolSession(registry, bindings, (root,), AgentTreeConfig(), "execution")
    assert session.generate(root.id, provider, ProviderRequest(prompt="Work"), "root").content == "done"
    assert provider.payloads[0]["tools"][0]["function"]["name"] == "lookup"
    assert provider.payloads[1]["messages"][-1]["tool_call_id"] == "abc"
    assert json.loads(provider.payloads[1]["messages"][-1]["content"])["output"] == 6


def test_gemini_roundtrip_contents():
    class Models:
        def __init__(self):
            self.calls = []
        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {"candidates": [{"content": {"parts": [{"function_call": {
                    "id": "g1", "name": "lookup", "args": {"value": 3}}}]}}],
                    "model_version": "fixture"}
            return {"text": "done", "model_version": "fixture"}
    class Client:
        def __init__(self): self.models = Models()
    client = Client()
    provider = GeminiProvider(ProviderConfig("gemini", model="fixture"), client=client)
    root = RootAgent(name="Root")
    tool = FunctionTool(name="lookup", function=lambda value: value * 2)
    registry, bindings = ToolRegistry(), ToolBindingRegistry()
    registry.register(tool); bindings.assign(root.id, tool.id)
    session = ToolSession(registry, bindings, (root,), AgentTreeConfig(), "execution")
    assert session.generate(root.id, provider, ProviderRequest(prompt="Work"), "root").content == "done"
    assert client.models.calls[0]["config"]["tools"][0]["function_declarations"][0]["name"] == "lookup"
    assert client.models.calls[1]["contents"][-1]["parts"][0]["function_response"]["id"] == "g1"


@pytest.mark.parametrize("provider_class", [GroqProvider, OpenRouterProvider, CerebrasProvider])
def test_named_compatible_adapters_roundtrip(provider_class):
    provider = provider_class(ProviderConfig(provider_class.__name__, model="fixture"), api_key="placeholder")
    payloads = []
    def fake_request(path, payload=None, **kwargs):
        payloads.append(payload)
        message = ({"content": None, "tool_calls": [{"id": "call-1", "type": "function",
                    "function": {"name": "lookup", "arguments": '{"value":3}'}}]}
                   if len(payloads) == 1 else {"content": "done"})
        return {"model": "fixture", "choices": [{"message": message}]}
    provider._json_request = fake_request
    root = RootAgent(name="Root")
    tool = FunctionTool(name="lookup", function=lambda value: value * 2)
    registry, bindings = ToolRegistry(), ToolBindingRegistry()
    registry.register(tool); bindings.assign(root.id, tool.id)
    session = ToolSession(registry, bindings, (root,), AgentTreeConfig(), "execution")
    assert session.generate(root.id, provider, ProviderRequest(prompt="Work"), "root").content == "done"
    assert payloads[1]["messages"][-1]["tool_call_id"] == "call-1"


def test_multiple_calls_are_sequential_and_ids_preserved():
    seen = []
    provider = ScriptProvider([("", (call(args='{"value":1}', call_id="first"),
                                      call(args='{"value":2}', call_id="second"))), ("done",)])
    root = RootAgent(name="Root")
    tool = FunctionTool(name="lookup", function=lambda value: seen.append(value) or value * 2)
    registry, bindings = ToolRegistry(), ToolBindingRegistry()
    registry.register(tool); bindings.assign(root.id, tool.id)
    session = ToolSession(registry, bindings, (root,), AgentTreeConfig(), "execution")
    assert session.generate(root.id, provider, ProviderRequest(prompt="Work"), "root").content == "done"
    assert seen == [1, 2]
    assert [item["call_id"] for item in provider.requests[1].tool_history[0]["results"]] == ["first", "second"]


def test_round_budget_is_deterministic_and_traced():
    provider = ScriptProvider([("", (call(call_id="one"),)),
                               ("", (call(call_id="two"),))])
    root = RootAgent(name="Root")
    tool = FunctionTool(name="lookup", function=lambda value: value)
    registry, bindings = ToolRegistry(), ToolBindingRegistry()
    registry.register(tool); bindings.assign(root.id, tool.id)
    session = ToolSession(registry, bindings, (root,),
                          AgentTreeConfig(max_tool_rounds=1), "execution")
    with pytest.raises(Exception, match="Tool round budget exhausted"):
        session.generate(root.id, provider, ProviderRequest(prompt="Work"), "root")
    assert len(provider.requests) == 2
    assert session.events[-1].event_type == "root.tool.budget_exhausted"


def test_disabled_tool_and_secret_exception_are_safe():
    root = RootAgent(name="Root")
    def fail():
        raise RuntimeError("Bearer private-token password=hidden")
    disabled = FunctionTool(name="disabled", function=fail, enabled=False)
    failing = FunctionTool(name="failing", function=fail)
    registry, bindings = ToolRegistry(), ToolBindingRegistry()
    for tool in (disabled, failing):
        registry.register(tool); bindings.assign(root.id, tool.id)
    session = ToolSession(registry, bindings, (root,), AgentTreeConfig(), "execution")
    assert session.definitions(root.id)[0]["name"] == "failing"
    denied = session.invoke(ToolCall("a", "disabled", root.id, "root", {}, "execution"))
    failed = session.invoke(ToolCall("b", "failing", root.id, "root", {}, "execution"))
    assert denied.error_type == "ToolDisabled"
    assert failed.error_type == "ToolExecutionError"
    assert "private-token" not in repr((denied, failed, session.events))


def test_root_and_manager_unassigned_tools_are_denied():
    root = RootAgent(name="Root")
    manager = ManagerAgent(name="Manager")
    tool = FunctionTool(name="lookup", function=lambda value: pytest.fail("executed"))
    registry = ToolRegistry(); registry.register(tool)
    session = ToolSession(registry, ToolBindingRegistry(), (root, manager),
                          AgentTreeConfig(), "execution")
    for agent, role in ((root, "root"), (manager, "manager")):
        result = session.invoke(ToolCall(role, "lookup", agent.id, role,
                                         {"value": 3}, "execution"))
        assert result.error_type == "ToolNotAssigned"


def test_mixed_provider_and_tool_hierarchy():
    class GeminiModels:
        def __init__(self): self.calls = []
        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            index = len(self.calls)
            if index in (1, 3):
                return {"candidates": [{"content": {"parts": [{"function_call": {
                    "id": f"root-{index}", "name": "project_context", "args": {}}}]}}]}
            return {"text": '{"delegate":true}' if index == 2 else "Integrated answer"}
    class GeminiClient:
        def __init__(self): self.models = GeminiModels()
    def compatible(provider, replies):
        payloads = []
        def fake(path, payload=None, **kwargs):
            payloads.append(payload)
            answer = replies.pop(0)
            message = ({"content": None, "tool_calls": [{"id": f"{provider.name}-{len(payloads)}",
                "type": "function", "function": {"name": answer[1], "arguments": "{}"}}]}
                       if isinstance(answer, tuple) else {"content": answer})
            return {"model": "fixture", "choices": [{"message": message}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}
        provider._json_request = fake
        return payloads

    gemini_client = GeminiClient()
    root_provider = GeminiProvider(ProviderConfig("gemini", model="fixture"), client=gemini_client)
    manager_provider = GroqProvider(ProviderConfig("groq", model="fixture"), api_key="placeholder")
    a_provider = OpenRouterProvider(ProviderConfig("openrouter", model="fixture"), api_key="placeholder")
    b_provider = CerebrasProvider(ProviderConfig("cerebras", model="fixture"), api_key="placeholder")
    manager_payloads = compatible(manager_provider, [("call", "docs_lookup"),
        '{"subtasks":[{"objective":"A","required_capabilities":["a"]},'
        '{"objective":"B","required_capabilities":["b"]}]}',
        ("call", "docs_lookup"), '{"decision":"pass","feedback":"ok"}',
        '{"decision":"pass","feedback":"ok"}'])
    a_payloads = compatible(a_provider, [("call", "calculator"), "A complete"])
    b_payloads = compatible(b_provider, [("call", "data_lookup"), "B complete"])
    target = tree(decomposer=ProviderTaskDecomposer(manager_provider),
                  manager_reviewer=ProviderManagerReviewer(manager_provider))
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    a = SpecialistAgent(name="A", capabilities=("a",))
    b = SpecialistAgent(name="B", capabilities=("b",))
    target.register_manager(manager)
    target.register_specialist(manager, a)
    target.register_specialist(manager, b)
    for provider in (root_provider, manager_provider, a_provider, b_provider):
        target.register_provider(provider)
    for agent, provider, name in ((target.root_agent, root_provider, "project_context"),
                                  (manager, manager_provider, "docs_lookup"),
                                  (a, a_provider, "calculator"),
                                  (b, b_provider, "data_lookup")):
        target.bind_provider(agent, provider)
        tool = FunctionTool(name=name, function=lambda: {"value": 1})
        target.register_tool(tool)
        target.bind_tool(agent, tool)
    result = target.run(Task(objective="Work"))
    assert result.success and result.final_output == "Integrated answer"
    assert result.metadata["tool_metrics"]["completed"] == 6
    assert len([event for event in result.trace.events if event.event_type.endswith(".tool.completed")]) == 6
    assert result.trace.events[-1].event_type == "execution.completed"
    assert manager_payloads[1]["messages"][-1]["tool_call_id"] == "groq-1"
    assert a_payloads[1]["messages"][-1]["tool_call_id"] == "openrouter-1"
    assert b_payloads[1]["messages"][-1]["tool_call_id"] == "cerebras-1"
    assert gemini_client.models.calls[1]["contents"][-1]["parts"][0]["function_response"]["id"] == "root-1"
    assert result.usage["total_tokens"] >= 15
    assert "placeholder" not in repr(result)


def test_revision_recovers_after_historical_tool_failure():
    provider = ScriptProvider([("", (call(call_id="first"),)), ("initial",),
                               ("", (call(call_id="second"),)), ("revised",)])
    target = tree(manager_reviewer=StaticManagerReviewer(outcomes=(
        (ReviewDecision.REVISE, "Try again"), (ReviewDecision.PASS, "Accepted"))))
    _, specialist = pair(target)
    attempts = []
    def flaky(value: int):
        attempts.append(value)
        if len(attempts) == 1:
            raise RuntimeError("Bearer private-token")
        return value * 2
    tool = FunctionTool(name="lookup", function=flaky)
    bind(target, specialist, provider, tool)
    result = target.run(Task(objective="Work"))
    assert result.success
    assert result.metadata["tool_metrics"]["failed"] == 1
    assert result.metadata["tool_metrics"]["completed"] == 1
    assert [e.event_type for e in result.trace.events if e.event_type in (
        "specialist.tool.failed", "specialist.tool.completed")] == [
            "specialist.tool.failed", "specialist.tool.completed"]
    assert "private-token" not in repr(result)


def test_root_round_budget_failure_is_in_last_state_trace():
    provider = ScriptProvider([("", (call(call_id="first"),)),
                               ("", (call(call_id="second"),))])
    target = tree(config=AgentTreeConfig(max_tool_rounds=1),
                  root_planner=ProviderRootPlanner(provider))
    tool = FunctionTool(name="lookup", function=lambda value: value)
    bind(target, target.root_agent, provider, tool)
    with pytest.raises(Exception, match="Tool round budget exhausted"):
        target.run(Task(objective="Work"))
    assert target.last_state is not None
    assert any(event.event_type == "root.tool.budget_exhausted"
               for event in target.last_state.trace.events)


def test_mcp_nested_schema_is_validated_before_transport():
    root = RootAgent(name="Root")
    definition = MCPToolDefinition(name="remote", input_schema={
        "type": "object", "properties": {"query": {"type": "object",
            "properties": {"term": {"type": "string", "minLength": 2}},
            "required": ["term"], "additionalProperties": False}},
        "required": ["query"], "additionalProperties": False})
    client = MockMCPClient(tools=(definition,))
    tool = MCPTool(client=client, definition=definition)
    registry, bindings = ToolRegistry(), ToolBindingRegistry()
    registry.register(tool); bindings.assign(root.id, tool.id)
    session = ToolSession(registry, bindings, (root,), AgentTreeConfig(), "execution")
    bad = session.invoke(ToolCall("one", "remote", root.id, "root",
                                  {"query": {"term": "x"}}, "execution"))
    assert bad.error_type == "ToolArgumentValidationError"
    assert not client.calls
