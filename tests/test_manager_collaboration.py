"""Execution-scoped Manager coordination through the public AgentTree runtime."""

import json
import importlib.util
from time import monotonic, sleep

import pytest

from agenttree import AgentTree, AgentTreeConfig, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (ProviderManagerReviewer, ProviderTaskDecomposer,
                            RuleBasedTaskTriage, StaticFinalReviewer,
                            StaticManagerReviewer, StaticTaskDecomposer)
from agenttree.core.collaboration import ManagerCollaborationSession, _active_collaboration_session
from agenttree.models import (ManagerMessage, ManagerMessageType, ManagerMessageStatus,
                              ReviewDecision, SubtaskTemplate)
from agenttree.providers import (BaseProvider, ProviderConfig, ProviderRequest,
                                 ProviderResponse, ProviderUsage, ProviderCapabilities)
from agenttree.providers.gemini import GeminiProvider
from agenttree.providers.groq import GroqProvider
from agenttree.providers.openrouter import OpenRouterProvider
from agenttree.providers.cerebras import CerebrasProvider
from agenttree.tools import FunctionTool
from agenttree.orchestration.backends import LangGraphOrchestrationBackend


class CallbackProvider(BaseProvider):
    def __init__(self, name, callback):
        super().__init__(ProviderConfig(name, model="fixture"))
        self.callback = callback
        self.requests = []

    @property
    def capabilities(self):
        return ProviderCapabilities(tool_calling=True)

    def generate(self, request):
        self.requests.append(request)
        outcome = self.callback(request)
        if isinstance(outcome, ProviderResponse):
            return outcome
        return ProviderResponse(content=json.dumps(outcome) if isinstance(outcome, dict) else outcome,
                                provider=self.name, model=request.model or "fixture",
                                usage=ProviderUsage(1, 1, 2))


def setup_pair(*, decomposer=None, reviewer=None, config=None, backend=None):
    fallback = CallbackProvider("fallback", lambda request: {"subtasks": []})
    target = AgentTree(root_agent=RootAgent(name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        decomposer=decomposer or ProviderTaskDecomposer(fallback),
        manager_reviewer=reviewer or StaticManagerReviewer(),
        final_reviewer=StaticFinalReviewer(), config=config,
        orchestration_backend=backend)
    a = ManagerAgent(name="Backend", capabilities=("manage",))
    b = ManagerAgent(name="Frontend", capabilities=("manage",))
    sa = SpecialistAgent(name="Backend worker", capabilities=("work",))
    sb = SpecialistAgent(name="Frontend worker", capabilities=("work",))
    target.register_manager(a); target.register_specialist(a, sa)
    target.register_manager(b); target.register_specialist(b, sb)
    for specialist in (sa, sb):
        provider = CallbackProvider(f"worker-{specialist.id}", lambda request: request.prompt)
        target.register_provider(provider)
        target.bind_provider(specialist, provider)
    return target, a, b


def bind_manager(target, agent, callback):
    provider = CallbackProvider(f"manager-{agent.id}", callback)
    target.register_provider(provider)
    target.bind_provider(agent, provider)
    return provider


def request(peer, *, kind="request", subject="API contract", content="Provide endpoint",
            thread_id=None):
    payload = {"target_manager_id": peer.id if isinstance(peer, ManagerAgent) else peer,
               "type": kind, "subject": subject, "content": content}
    if thread_id is not None:
        payload["thread_id"] = thread_id
    return {"collaboration_request": payload}


def subtask(objective):
    return {"subtasks": [{"objective": objective, "required_capabilities": ["work"]}]}


def test_request_response_during_decomposition_reaches_root_output():
    target, backend, frontend = setup_pair()
    def backend_reply(req):
        if req.metadata["strategy"] == "decomposition":
            result = req.context.get("collaboration_result")
            return request(frontend) if result is None else subtask(
                "Backend uses " + result["response"]["content"])
        return {"content": "Backend answer"}
    def frontend_reply(req):
        if req.metadata["strategy"] == "manager_collaboration":
            return {"content": "POST /api/login"}
        return subtask("Frontend login page")
    backend_provider = bind_manager(target, backend, backend_reply)
    frontend_provider = bind_manager(target, frontend, frontend_reply)
    target.allow_manager_communication(backend, frontend)
    result = target.run(Task(objective="Build login"))
    assert result.success and "POST /api/login" in result.final_output
    first_result = backend_provider.requests[1].context["collaboration_result"]
    received = frontend_provider.requests[0].context["message"]
    assert first_result["message_id"] == received["message_id"]
    assert first_result["thread_id"] == received["thread_id"]
    assert first_result["response"]["reply_to"] == received["message_id"]
    history = target.last_collaboration_messages
    assert len(history) == 2
    assert history[1].reply_to == history[0].message_id
    assert history[1].thread_id == history[0].thread_id
    history[0].metadata["local"] = True
    assert "local" not in target.last_collaboration_messages[0].metadata
    assert result.metadata["collaboration_metrics"]["messages_sent"] == 2
    events = [event.event_type for event in result.trace.events]
    assert events.index("manager.collaboration.requested") < events.index("manager.collaboration.responded")
    assert events.index("manager.collaboration.responded") < events.index("orchestration.execution_started")
    assert events[-1] == "execution.completed"
    assert any(call["stage"] == "manager_collaboration" for call in result.usage["calls"])


def test_message_contract_correlation_redaction_and_serialization():
    message = ManagerMessage(execution_id="task", from_manager_id="a", to_manager_id="b",
        type=ManagerMessageType.REQUEST, subject="API", content="Authorization: Bearer SECRET",
        metadata={"api_key": "SECRET"})
    response = ManagerMessage(execution_id="task", from_manager_id="b", to_manager_id="a",
        type=ManagerMessageType.RESPONSE, subject="API", content="done",
        thread_id=message.thread_id, reply_to=message.message_id,
        correlation_id=message.message_id, status=ManagerMessageStatus.RESPONDED)
    assert message.message_id != response.message_id
    assert response.thread_id == message.thread_id
    assert response.reply_to == message.message_id
    assert "SECRET" not in repr((message, response))
    assert "SECRET" not in json.dumps(message.to_dict())
    with pytest.raises(ValueError):
        ManagerMessage(execution_id="task", from_manager_id="a", to_manager_id="b",
                       type=ManagerMessageType.RESPONSE, subject="API", content="done")


def test_directional_permission_and_unknown_peer_denials_recover():
    target, a, b = setup_pair()
    a_provider = bind_manager(target, a, lambda req: subtask("A work") if req.metadata["strategy"] == "decomposition" else {"content": "A"})
    def b_reply(req):
        if req.metadata["strategy"] == "manager_collaboration":
            return {"content": "B"}
        result = req.context.get("collaboration_result")
        if result is None:
            return request(a)
        return subtask("B recovered from " + result["error_type"])
    b_provider = bind_manager(target, b, b_reply)
    target.allow_manager_communication(a, b)
    result = target.run(Task(objective="Work"))
    assert result.success and "ManagerCollaborationDisabled" in result.final_output
    assert not any(req.metadata["strategy"] == "manager_collaboration" for req in a_provider.requests)
    assert len(b_provider.requests) == 2
    assert "manager.collaboration.denied" in [event.event_type for event in result.trace.events]
    assert target.manager_permissions[a.id] == (b.id,)
    with pytest.raises(ValueError):
        target.allow_manager_communication(a, b)


def test_bidirectional_requests_are_bounded_and_correlated():
    target, a, b = setup_pair()
    def a_reply(req):
        if req.metadata["strategy"] == "manager_collaboration":
            return {"content": "backend contract"}
        result = req.context.get("collaboration_result")
        return request(b) if result is None else subtask("A uses " + result["response"]["content"])
    def b_reply(req):
        if req.metadata["strategy"] == "manager_collaboration":
            return {"content": "frontend requirement"}
        result = req.context.get("collaboration_result")
        return request(a, subject="UI requirements") if result is None else subtask(
            "B uses " + result["response"]["content"])
    bind_manager(target, a, a_reply); bind_manager(target, b, b_reply)
    target.allow_manager_communication(a, b)
    target.allow_manager_communication(b, a)
    result = target.run(Task(objective="Work"))
    assert result.success and "frontend requirement" in result.final_output
    assert "backend contract" in result.final_output
    assert result.metadata["collaboration_metrics"]["threads"] == 2
    assert result.metadata["collaboration_metrics"]["messages_sent"] == 4


def test_disabled_default_preserves_single_provider_call_per_manager():
    target, a, b = setup_pair()
    ap = bind_manager(target, a, lambda req: subtask("A work"))
    bp = bind_manager(target, b, lambda req: subtask("B work"))
    result = target.run(Task(objective="Work"))
    assert result.success
    assert len(ap.requests) == len(bp.requests) == 1
    assert "collaboration_metrics" not in result.metadata
    assert not any("collaboration" in event.event_type for event in result.trace.events)


def test_global_budget_and_thread_round_limit_allow_recovery():
    target, a, b = setup_pair(config=AgentTreeConfig(
        max_collaboration_messages_total=2, max_collaboration_rounds_per_thread=1,
        max_collaboration_turns_per_decision=3))
    def a_reply(req):
        result = req.context.get("collaboration_result")
        if result is None:
            return request(b)
        if result["success"]:
            return request(b, thread_id=result["thread_id"])
        return subtask("A recovered from " + result["error_type"])
    bind_manager(target, a, a_reply)
    bind_manager(target, b, lambda req: {"content": "reply"} if req.metadata["strategy"] == "manager_collaboration" else subtask("B"))
    target.allow_manager_communication(a, b)
    result = target.run(Task(objective="Work"))
    assert result.success and "ManagerMessageBudgetExceeded" in result.final_output
    assert result.metadata["collaboration_metrics"]["messages_sent"] == 2
    assert "manager.collaboration.budget_exhausted" in [event.event_type for event in result.trace.events]


def test_timeout_is_safe_and_later_run_is_isolated():
    target, a, b = setup_pair(config=AgentTreeConfig(collaboration_timeout=0.01))
    def a_reply(req):
        result = req.context.get("collaboration_result")
        return request(b) if result is None else subtask("A continues after " + result["error_type"])
    def b_reply(req):
        if req.metadata["strategy"] == "manager_collaboration":
            sleep(0.08)
            return {"content": "late response"}
        return subtask("B")
    bind_manager(target, a, a_reply); bind_manager(target, b, b_reply)
    target.allow_manager_communication(a, b)
    started = monotonic()
    first = target.run(Task(objective="Work"))
    assert monotonic() - started < 0.07
    assert first.success and "ManagerCollaborationTimeout" in first.final_output
    assert first.metadata["collaboration_metrics"]["timeouts"] == 1
    assert "manager.collaboration.timeout" in [event.event_type for event in first.trace.events]
    second = target.run(Task(objective="Work again"))
    assert second.task_id != first.task_id
    assert second.metadata["collaboration_metrics"]["timeouts"] == 1
    sleep(0.1)
    assert first.metadata["collaboration_metrics"]["messages_sent"] == 1


def test_payload_limit_and_unknown_manager_do_not_route():
    target, a, b = setup_pair(config=AgentTreeConfig(max_collaboration_payload_bytes=100))
    responses = iter([request("unknown"), request(b, content="x" * 200), subtask("A recovered")])
    bind_manager(target, a, lambda req: next(responses))
    peer = bind_manager(target, b, lambda req: subtask("B"))
    target.allow_manager_communication(a, b)
    result = target.run(Task(objective="Work"))
    assert result.success
    assert not any(req.metadata["strategy"] == "manager_collaboration" for req in peer.requests)
    assert result.metadata["collaboration_metrics"]["failures"] == 2
    assert "manager.collaboration.denied" in [event.event_type for event in result.trace.events]


def test_review_request_can_inform_own_manager_revision():
    target, a, b = setup_pair(
        decomposer=StaticTaskDecomposer((SubtaskTemplate("Initial", ("work",)),)),
        reviewer=ProviderManagerReviewer(CallbackProvider("fallback-review", lambda req: {"decision": "pass", "feedback": "ok"})))
    def a_reply(req):
        if req.metadata["strategy"] == "manager_review":
            result = req.context.get("collaboration_result")
            if result is None:
                return request(b, kind="review_request", subject="API fit", content="Review contract")
            revision = req.context["subtask"]["metadata"].get("revision")
            return {"decision": "revise", "feedback": result["response"]["content"]} if not revision else {
                "decision": "pass", "feedback": "accepted"}
        return {"content": "A response"}
    def b_reply(req):
        return {"content": "Use POST /api/login"} if req.metadata["strategy"] == "manager_collaboration" else {
            "decision": "pass", "feedback": "ok"}
    bind_manager(target, a, a_reply); bind_manager(target, b, b_reply)
    target.allow_manager_communication(a, b)
    result = target.run(Task(objective="Work"))
    assert result.success and any(
        outcome.revision_count >= 1
        for manager in result.manager_results for outcome in manager.subtask_outcomes)
    assert result.metadata["collaboration_metrics"]["responses"] >= 1
    assert any(event.metadata.get("message_type") == "review_response"
               for event in result.trace.events if event.event_type == "manager.collaboration.responded")
    assert "Use POST /api/login" in repr(result.manager_results)


def test_thread_round_limit_and_per_manager_budget():
    a = ManagerAgent(name="A")
    b = ManagerAgent(name="B")
    responder = CallbackProvider("responder", lambda req: {"content": "ok"})
    session = ManagerCollaborationSession(execution_id="task", task_objective="Work",
        managers=(a, b), permissions={a.id: (b.id,)},
        provider_bindings={b.id: (responder, None)},
        config=AgentTreeConfig(max_collaboration_rounds_per_thread=1,
                               max_collaboration_messages_per_manager=1))
    session.activate((a.id, b.id))
    first = session.route(a.id, request(b)["collaboration_request"])
    assert first.success
    second = session.route(a.id, request(b, thread_id=first.request.thread_id)["collaboration_request"])
    assert not second.success and second.error_type == "ManagerMessageBudgetExceeded"
    assert len(responder.requests) == 1
    assert len(session.history) == 2
    # A separate session isolates the thread-round rule from the sender budget.
    second_session = ManagerCollaborationSession(execution_id="task-2", task_objective="Work",
        managers=(a, b), permissions={a.id: (b.id,)},
        provider_bindings={b.id: (responder, None)},
        config=AgentTreeConfig(max_collaboration_rounds_per_thread=1))
    second_session.activate((a.id, b.id))
    accepted = second_session.route(a.id, request(b)["collaboration_request"])
    denied = second_session.route(a.id, request(b, thread_id=accepted.request.thread_id)["collaboration_request"])
    assert denied.error_type == "ManagerThreadRoundLimitExceeded"
    assert second_session.events[-1].event_type == "manager.collaboration.round_limit"


def test_peer_response_is_non_reentrant_and_recovers():
    target, a, b = setup_pair()
    def a_reply(req):
        outcome = req.context.get("collaboration_result")
        return request(b) if outcome is None else subtask("A after " + outcome["error_type"])
    def b_reply(req):
        if req.metadata["strategy"] == "manager_collaboration":
            return request(a)
        return subtask("B work")
    bind_manager(target, a, a_reply); bind_manager(target, b, b_reply)
    target.allow_manager_communication(a, b)
    target.allow_manager_communication(b, a)
    result = target.run(Task(objective="Work"))
    assert result.success and "ManagerCollaborationExecutionError" in result.final_output
    assert result.metadata["collaboration_metrics"]["failures"] == 1
    assert result.metadata["collaboration_metrics"]["messages_sent"] == 1


def test_response_worker_cannot_reenter_runtime_even_with_reverse_permission():
    a = ManagerAgent(name="A")
    b = ManagerAgent(name="B")
    attempted = []
    def callback(req):
        session = _active_collaboration_session.get()
        attempted.append(session.route(b.id, request(a)["collaboration_request"]).error_type)
        return {"content": "B work product"}
    responder = CallbackProvider("responder", callback)
    session = ManagerCollaborationSession(execution_id="task", task_objective="Work",
        managers=(a, b), permissions={a.id: (b.id,), b.id: (a.id,)},
        provider_bindings={b.id: (responder, None)}, config=AgentTreeConfig())
    session.activate((a.id, b.id))
    token = _active_collaboration_session.set(session)
    try:
        result = session.route(a.id, request(b)["collaboration_request"])
    finally:
        _active_collaboration_session.reset(token)
    assert result.success
    assert attempted == ["ManagerCollaborationDisabled"]
    assert len(session.history) == 2


def test_responding_manager_uses_own_tool_without_peer_permission_escalation():
    target, a, b = setup_pair()
    calls = []
    def schema():
        calls.append(a.id)
        return {"path": "/api/login"}
    tool = FunctionTool(name="schema", function=schema)
    target.register_tool(tool)
    target.bind_tool(a, tool)
    def a_reply(req):
        if req.metadata["strategy"] == "manager_collaboration":
            if not req.tool_history:
                return ProviderResponse(content="", provider="manager-a", tool_calls=(
                    {"id": "schema-call", "function": {"name": "schema", "arguments": "{}"}},))
            value = req.tool_history[0]["results"][0]["output"]
            return {"content": "API path " + value["path"]}
        return subtask("A work")
    def b_reply(req):
        if req.metadata["strategy"] == "decomposition":
            outcome = req.context.get("collaboration_result")
            if outcome is None and not req.tool_history:
                return ProviderResponse(content="", provider="manager-b", tool_calls=(
                    {"id": "unauthorized", "function": {"name": "schema", "arguments": "{}"}},))
            return request(a, content="Provide API path") if outcome is None else subtask(
                "B uses " + outcome["response"]["content"])
        return {"content": "B answer"}
    ap = bind_manager(target, a, a_reply)
    bp = bind_manager(target, b, b_reply)
    target.allow_manager_communication(b, a)
    result = target.run(Task(objective="Work"))
    assert result.success and "/api/login" in result.final_output
    assert calls == [a.id]
    assert ap.requests[1].tools[0]["name"] == "schema"
    assert all(not req.tools for req in bp.requests)
    tool_events = [event for event in result.trace.events if event.event_type == "manager.tool.completed"]
    assert len(tool_events) == 1 and tool_events[0].actor_id == a.id
    assert any(event.event_type == "manager.tool.denied" and event.actor_id == b.id
               for event in result.trace.events)
    assert result.metadata["tool_metrics"]["completed"] == 1


def test_execution_history_does_not_cross_runs():
    target, a, b = setup_pair()
    def a_reply(req):
        outcome = req.context.get("collaboration_result")
        return request(b) if outcome is None else subtask("A uses " + outcome["response"]["content"])
    peer = bind_manager(target, b, lambda req: {"content": "reply"} if req.metadata["strategy"] == "manager_collaboration" else subtask("B"))
    bind_manager(target, a, a_reply)
    target.allow_manager_communication(a, b)
    first = target.run(Task(objective="First"))
    second = target.run(Task(objective="Second"))
    first_message = peer.requests[0].context["message"]
    second_message = peer.requests[2].context["message"]
    assert first_message["message_id"] != second_message["message_id"]
    assert first_message["thread_id"] != second_message["thread_id"]
    assert peer.requests[2].context["thread_history"][0]["execution_id"] == second.task_id
    assert first.task_id != second.task_id


def test_mixed_native_and_compatible_provider_collaboration_with_tools():
    class GeminiModels:
        def __init__(self): self.calls = []
        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return {"text": '{"delegate":true}' if len(self.calls) == 1 else "Full stack complete",
                    "model_version": "fixture",
                    "usage_metadata": {"prompt_token_count": 2,
                                       "candidates_token_count": 3,
                                       "total_token_count": 5}}
    class Client:
        def __init__(self): self.models = GeminiModels()
    root_client = Client()
    root_provider = GeminiProvider(ProviderConfig("gemini", model="fixture"), client=root_client)
    backend = ManagerAgent(name="Backend", capabilities=("manage",))
    frontend = ManagerAgent(name="Frontend", capabilities=("manage",))
    database = ManagerAgent(name="Database", capabilities=("manage",))
    groq = GroqProvider(ProviderConfig("groq", model="fixture"), api_key="placeholder")
    openrouter = OpenRouterProvider(ProviderConfig("openrouter", model="fixture"), api_key="placeholder")
    cerebras = CerebrasProvider(ProviderConfig("cerebras", model="fixture"), api_key="placeholder")
    target = AgentTree(root_agent=RootAgent(name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        decomposer=ProviderTaskDecomposer(groq),
        manager_reviewer=ProviderManagerReviewer(groq),
        final_reviewer=StaticFinalReviewer())
    specialists = []
    for manager, capability in ((backend, "backend"), (frontend, "frontend"), (database, "database")):
        specialist = SpecialistAgent(name=f"{capability} worker", capabilities=(capability,))
        target.register_manager(manager)
        target.register_specialist(manager, specialist)
        specialists.append(specialist)
    for provider in (root_provider, groq, openrouter, cerebras):
        target.register_provider(provider)
    for agent, provider in ((target.root_agent, root_provider), (backend, groq),
                            (frontend, openrouter), (database, cerebras)):
        target.bind_provider(agent, provider)
    for specialist in specialists:
        provider = CallbackProvider(f"worker-{specialist.id}", lambda req: req.prompt)
        target.register_provider(provider)
        target.bind_provider(specialist, provider)
        tool = FunctionTool(name=f"tool_{specialist.name.split()[0]}", function=lambda: "safe")
        target.register_tool(tool)
        target.bind_tool(specialist, tool)
    schema_tool = FunctionTool(name="api_schema", function=lambda: {"path": "/api/login"})
    target.register_tool(schema_tool)
    target.bind_tool(backend, schema_tool)
    target.allow_manager_communication(backend, database)
    target.allow_manager_communication(frontend, backend)

    def routed_subtask(objective, capability):
        return {"subtasks": [{"objective": objective,
                              "required_capabilities": [capability]}]}

    groq_answers = [
        {"content": None, "tool_calls": [{"id": "tool-1", "type": "function",
            "function": {"name": "api_schema", "arguments": "{}"}}]},
        {"content": json.dumps(request(database, subject="Schema", content="Provide user schema"))},
        {"content": json.dumps(routed_subtask("Backend uses users table", "backend"))},
        {"content": json.dumps({"content": "POST /api/login"})},
        {"content": '{"decision":"pass","feedback":"ok"}'},
    ]
    openrouter_answers = [
        {"content": json.dumps(request(backend, subject="API", content="Provide endpoint"))},
        {"content": json.dumps(routed_subtask("Frontend uses POST /api/login", "frontend"))},
        {"content": '{"decision":"pass","feedback":"ok"}'},
    ]
    cerebras_answers = [
        {"content": json.dumps({"content": "users table"})},
        {"content": json.dumps(routed_subtask("Database work", "database"))},
        {"content": '{"decision":"pass","feedback":"ok"}'},
    ]
    payloads = {}
    for provider, answers in ((groq, groq_answers),
                              (openrouter, openrouter_answers),
                              (cerebras, cerebras_answers)):
        sent = []
        def fake_request(path, payload=None, *, responses=answers, sent=sent, **kwargs):
            sent.append(payload)
            return {"model": "fixture", "choices": [{"message": responses.pop(0)}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}
        provider._json_request = fake_request
        payloads[provider.name] = sent
    result = target.run(Task(objective="Build full-stack login"))
    assert result.success and result.final_output == "Full stack complete"
    assert result.metadata["collaboration_metrics"]["messages_sent"] == 4
    assert result.metadata["tool_metrics"]["completed"] == 1
    assert payloads["groq"][0]["tools"][0]["function"]["name"] == "api_schema"
    assert payloads["groq"][1]["messages"][-1]["tool_call_id"] == "tool-1"
    assert payloads["openrouter"][1]["messages"][0]["role"] in ("system", "user")
    assert len(payloads["cerebras"]) == 3
    assert len(root_client.models.calls) == 2
    assert result.usage["total_tokens"] > 0
    assert result.trace.events[-1].event_type == "execution.completed"
    assert "placeholder" not in repr(result)


def test_context_message_is_delivered_without_peer_provider_call():
    a = ManagerAgent(name="A")
    b = ManagerAgent(name="B")
    responder = CallbackProvider("responder", lambda req: {"content": "unused"})
    session = ManagerCollaborationSession(execution_id="task", task_objective="Work",
        managers=(a, b), permissions={a.id: (b.id,)},
        provider_bindings={b.id: (responder, None)}, config=AgentTreeConfig())
    session.activate((a.id, b.id))
    outcome = session.route(a.id, request(b, kind="context", content="API uses JSON")["collaboration_request"])
    assert outcome.success and outcome.response is None
    assert not responder.requests
    assert session.inbox_for(b.id)[0]["content"] == "API uses JSON"
    assert session.metrics["messages_sent"] == 1


def test_specialist_output_cannot_create_peer_message():
    target, a, b = setup_pair(decomposer=StaticTaskDecomposer((
        SubtaskTemplate("Work", ("work",)),)))
    target.allow_manager_communication(a, b)
    specialist = a.specialists[0]
    malicious = CallbackProvider("malicious-worker", lambda req: json.dumps(request(b)))
    target.register_provider(malicious)
    target.bind_provider(specialist, malicious)
    result = target.run(Task(objective="Work"))
    assert result.success
    assert "collaboration_metrics" not in result.metadata
    assert not any("collaboration" in event.event_type for event in result.trace.events)
    with pytest.raises(TypeError):
        target.allow_manager_communication(specialist, b)


def test_redacted_payload_and_unknown_ids_never_enter_trace():
    a = ManagerAgent(name="A")
    b = ManagerAgent(name="B")
    peer = CallbackProvider("peer", lambda req: {"content": "password=HIDDEN"})
    session = ManagerCollaborationSession(execution_id="task", task_objective="Work",
        managers=(a, b), permissions={a.id: (b.id,)},
        provider_bindings={b.id: (peer, None)}, config=AgentTreeConfig())
    session.activate((a.id, b.id))
    result = session.route(a.id, request(b, content="Authorization: Bearer SECRET")["collaboration_request"])
    assert result.success
    assert "SECRET" not in json.dumps(result.to_context())
    assert "HIDDEN" not in json.dumps(result.to_context())
    bad = session.route(a.id, request("api_key=LEAK")["collaboration_request"])
    assert bad.error_type == "ManagerNotFound"
    assert "LEAK" not in repr(session.events)
    invalid = session.route(a.id, request(b, thread_id="password=LEAK")["collaboration_request"])
    assert invalid.error_type == "ManagerMessageValidationError"
    assert "LEAK" not in repr(session.events)


def test_missing_peer_provider_is_structured_recoverable_error():
    a = ManagerAgent(name="A")
    b = ManagerAgent(name="B")
    session = ManagerCollaborationSession(execution_id="task", task_objective="Work",
        managers=(a, b), permissions={a.id: (b.id,)},
        provider_bindings={}, config=AgentTreeConfig())
    session.activate((a.id, b.id))
    outcome = session.route(a.id, request(b)["collaboration_request"])
    assert not outcome.success and outcome.error_type == "ManagerCollaborationExecutionError"
    assert session.events[-1].event_type == "manager.collaboration.failed"


def test_invalid_peer_response_still_counts_provider_usage():
    target, a, b = setup_pair()
    def a_reply(req):
        result = req.context.get("collaboration_result")
        return request(b) if result is None else subtask("A recovered from " + result["error_type"])
    bind_manager(target, a, a_reply)
    bind_manager(target, b, lambda req: {"unexpected": "shape"} if
                 req.metadata["strategy"] == "manager_collaboration" else subtask("B work"))
    target.allow_manager_communication(a, b)
    result = target.run(Task(objective="Work"))
    assert result.success
    assert "ManagerCollaborationExecutionError" in result.final_output
    assert len([call for call in result.usage["calls"]
                if call["stage"] == "manager_collaboration"]) == 1


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None,
                    reason="Optional LangGraph is not installed")
def test_real_langgraph_backend_uses_same_collaboration_session():
    target, a, b = setup_pair(backend=LangGraphOrchestrationBackend())
    def a_reply(req):
        result = req.context.get("collaboration_result")
        return request(b) if result is None else subtask("A uses " + result["response"]["content"])
    bind_manager(target, a, a_reply)
    bind_manager(target, b, lambda req: {"content": "peer contract"} if
                 req.metadata["strategy"] == "manager_collaboration" else subtask("B work"))
    target.allow_manager_communication(a, b)
    result = target.run(Task(objective="Work"))
    assert result.success and "peer contract" in result.final_output
    assert result.metadata["collaboration_metrics"]["messages_sent"] == 2


def test_global_budget_across_multiple_manager_threads():
    a, b, c = (ManagerAgent(name=name) for name in ("A", "B", "C"))
    responder = CallbackProvider("peer", lambda req: {"content": "ok"})
    session = ManagerCollaborationSession(execution_id="task", task_objective="Work",
        managers=(a, b, c), permissions={a.id: (b.id,), c.id: (b.id,)},
        provider_bindings={b.id: (responder, None)},
        config=AgentTreeConfig(max_collaboration_messages_total=2))
    session.activate((a.id, b.id, c.id))
    assert session.route(a.id, request(b)["collaboration_request"]).success
    denied = session.route(c.id, request(b)["collaboration_request"])
    assert denied.error_type == "ManagerMessageBudgetExceeded"
    assert len(responder.requests) == 1
    assert session.metrics["messages_sent"] == 2


def test_repeated_provider_collaboration_requests_end_at_decision_limit():
    target, a, b = setup_pair(config=AgentTreeConfig(max_collaboration_turns_per_decision=1))
    bind_manager(target, a, lambda req: request(b))
    bind_manager(target, b, lambda req: {"content": "reply"} if
                 req.metadata["strategy"] == "manager_collaboration" else subtask("B"))
    target.allow_manager_communication(a, b)
    with pytest.raises(Exception, match="collaboration decision turn limit"):
        target.run(Task(objective="Work"))
    assert any(event.event_type == "manager.collaboration.budget_exhausted"
               for event in target.last_state.trace.events)
