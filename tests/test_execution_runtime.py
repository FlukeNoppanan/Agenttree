"""Durable lifecycle and real process-restart tests for the execution runtime."""

import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
from time import sleep

import pytest

from agenttree import AgentTree, AgentTreeConfig, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    ExecutionCapacityError, ExecutionCancelled, ExecutionConflict,
    ExecutionRecoveryBlocked, ExecutionRuntime, ExecutionState,
    InMemoryExecutionStore, ProviderTaskDecomposer, RuleBasedTaskTriage,
    SQLiteExecutionStore, StaticFinalReviewer, StaticManagerReviewer,
    ProviderRootPlanner, ProviderRootSynthesizer,
)
from agenttree.core.execution_codec import ExecutionDataError, dumps, loads
from agenttree.core.execution_store import ExecutionCheckpoint, ExecutionRecord, utc_now
from agenttree.models import ExecutionEvent, ReviewDecision
from agenttree.providers import (
    BaseProvider, MockProvider, ProviderCapabilities, ProviderConfig,
    ProviderResponse, ProviderUsage,
)
from agenttree.tools import FunctionTool, ToolRecoveryPolicy


class ScriptProvider(BaseProvider):
    def __init__(self, name, responder):
        super().__init__(ProviderConfig(name, model="fixture"))
        self.responder = responder
        self.requests = []
        self.credential = "SUPER_SECRET_PHASE5_TEST_VALUE"

    @property
    def capabilities(self):
        return ProviderCapabilities(tool_calling=True)

    def generate(self, request):
        self.requests.append(request)
        value = self.responder(request)
        if isinstance(value, ProviderResponse):
            return value
        return ProviderResponse(content=json.dumps(value) if isinstance(value, dict) else value,
                                provider=self.name, model="fixture",
                                usage=ProviderUsage(1, 1, 2))


def build_tree(log_path=None, *, different=False, slow=None, revise=False,
               slow_tool=None, slow_collaboration=None,
               tool_policy=ToolRecoveryPolicy.UNKNOWN, exit_after_tool_effect=False,
               exit_after_specialist_effect=None, with_root=False,
               exit_after_collaboration_effect=False):
    log_path = Path(log_path) if log_path is not None else None
    def mark(label):
        if log_path is not None:
            with log_path.open("a") as output:
                output.write(label + "\n")
    fallback = ScriptProvider("fallback", lambda request: {"subtasks": []})
    reviewer = (StaticManagerReviewer(outcomes=((ReviewDecision.REVISE, "Improve"),
                                                (ReviewDecision.PASS, "Accepted")))
                if revise else StaticManagerReviewer())
    tree = AgentTree(root_agent=RootAgent(id="root", name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        decomposer=ProviderTaskDecomposer(fallback),
        manager_reviewer=reviewer, final_reviewer=StaticFinalReviewer(),
        config=AgentTreeConfig(max_manager_revisions=1) if revise else None)
    a = ManagerAgent(id="manager-a", name="A", capabilities=("manage",))
    b = ManagerAgent(id="manager-b", name="B" if not different else "Changed", capabilities=("manage",))
    sa = SpecialistAgent(id="specialist-a", name="A worker", capabilities=("work",))
    sb = SpecialistAgent(id="specialist-b", name="B worker", capabilities=("work",))
    tree.register_manager(a); tree.register_specialist(a, sa)
    tree.register_manager(b)
    if with_root:
        tree.register_specialist(a, sb)
        sf = SpecialistAgent(id="specialist-frontend", name="Frontend worker",
                             capabilities=("work",))
        tree.register_specialist(b, sf)
    else:
        tree.register_specialist(b, sb)
        sf = None
    def schema():
        if slow_tool is not None:
            slow_tool[0].set(); slow_tool[1].wait(3)
        mark("tool")
        if exit_after_tool_effect:
            os._exit(0)
        return {"path": "/api/login"}
    tool = FunctionTool(name="schema", function=schema, tool_id="schema-tool",
                        metadata={"api_key": "SUPER_SECRET_PHASE5_TEST_VALUE"},
                        recovery_policy=tool_policy)
    tree.register_tool(tool); tree.bind_tool(a, tool)
    def a_response(request):
        if request.metadata["strategy"] == "manager_collaboration":
            return {"content": "backend contract"}
        if not request.tool_history and "collaboration_result" not in request.context:
            return ProviderResponse(content="", provider="manager-a", tool_calls=(
                {"id": "schema-call", "function": {"name": "schema", "arguments": "{}"}},))
        if "collaboration_result" not in request.context:
            return {"collaboration_request": {"target_manager_id": b.id, "type": "request",
                      "subject": "UI", "content": "Provide interface requirements"}}
        mark("decompose-a")
        return {"subtasks": [{"objective": "API uses " +
                 request.context["collaboration_result"]["response"]["content"],
                 "required_capabilities": ["work"]}]}
    def b_response(request):
        if request.metadata["strategy"] == "manager_collaboration":
            if slow_collaboration is not None:
                slow_collaboration[0].set(); slow_collaboration[1].wait(3)
            mark("collaboration-response")
            if exit_after_collaboration_effect:
                os._exit(0)
            return {"content": "POST /api/login"}
        mark("decompose-b")
        return {"subtasks": [{"objective": "Login UI", "required_capabilities": ["work"]}]}
    manager_providers = ((a, "mock-groq" if with_root else "manager-a", a_response),
                         (b, "mock-openrouter" if with_root else "manager-b", b_response))
    for agent, name, callback in manager_providers:
        provider = ScriptProvider(name, callback)
        tree.register_provider(provider); tree.bind_provider(agent, provider)
    for specialist in ((sa, sb, sf) if with_root else (sa, sb)):
        def worker(request, specialist=specialist):
            if slow is not None:
                slow[0].set(); slow[1].wait(3)
            mark("specialist-" + specialist.id)
            if exit_after_specialist_effect == specialist.id:
                os._exit(0)
            return request.prompt
        provider_name = ({sa.id: "mock-cerebras", sb.id: "mock-gemini-worker",
                          sf.id: "mock-openrouter-worker"}[specialist.id]
                         if with_root else "provider-" + specialist.id)
        provider = ScriptProvider(provider_name, worker)
        tree.register_provider(provider); tree.bind_provider(specialist, provider)
    tree.allow_manager_communication(a, b)
    if with_root:
        def plan(_request):
            mark("root-plan")
            return {"delegate": True}
        def synth(_request):
            mark("root-synthesis")
            return "Completed login design"
        tree._root_planner = ProviderRootPlanner(ScriptProvider("mock-gemini", plan))
        tree._root_synthesizer = ProviderRootSynthesizer(
            ScriptProvider("mock-gemini-synthesis", synth))
    return tree


class ExitAfterDelegation(SQLiteExecutionStore):
    def checkpoint(self, execution_id, expected_version, checkpoint):
        result = super().checkpoint(execution_id, expected_version, checkpoint)
        if checkpoint.phase == "delegation":
            os._exit(0)
        return result


class ExitAfterManagerReview(SQLiteExecutionStore):
    def checkpoint(self, execution_id, expected_version, checkpoint):
        result = super().checkpoint(execution_id, expected_version, checkpoint)
        if checkpoint.phase == "manager_review":
            os._exit(0)
        return result


def child_checkpoint(db_path, log_path, phase="delegation", revise=False):
    tree = build_tree(log_path, revise=revise)
    store_type = ExitAfterManagerReview if phase == "manager_review" else ExitAfterDelegation
    runtime = ExecutionRuntime(store_type(db_path), max_workers=1)
    runtime.submit(tree, Task(id="durable-task", objective="Build login"))
    Event().wait(10)


def test_background_durable_result_survives_new_runtime(tmp_path):
    path = tmp_path / "runs.db"
    tree = build_tree()
    runtime = ExecutionRuntime(SQLiteExecutionStore(path))
    handle = runtime.submit(tree, Task(objective="Build login"))
    result = handle.result(timeout=5)
    assert handle.status().state is ExecutionState.COMPLETED
    assert not hasattr(handle.status(), "input_json")
    assert handle.status().metrics["checkpoint_count"] == 5
    assert result.success and "/api/login" in result.final_output
    assert result.trace.events[-1].event_type == "execution.completed"
    assert handle.status().checkpoint_sequence == 5
    runtime.shutdown()
    reopened = ExecutionRuntime(SQLiteExecutionStore(path))
    assert reopened.handle(handle.execution_id).result().final_output == result.final_output
    assert reopened.store.list_recoverable() == ()
    assert b"SUPER_SECRET_PHASE5_TEST_VALUE" not in path.read_bytes()
    reopened.shutdown()


def test_event_stream_independent_cursors_and_restart(tmp_path):
    path = tmp_path / "event-stream.db"
    runtime = ExecutionRuntime(SQLiteExecutionStore(path))
    handle = runtime.submit(build_tree(), Task(objective="Build login"))
    handle.result(timeout=5)
    first = tuple(handle.stream_events(page_size=3))
    second = tuple(handle.stream_events(page_size=7))
    assert first == second
    assert first and first[-1].event.event_type == "execution.completed"
    assert tuple(item.sequence for item in first) == tuple(range(1, len(first) + 1))
    runtime.shutdown()
    reopened = ExecutionRuntime(SQLiteExecutionStore(path))
    tail = tuple(reopened.handle(handle.execution_id).stream_events(
        after_sequence=first[-3].sequence))
    assert tail == first[-2:]
    assert tuple(reopened.handle(handle.execution_id).stream_events(
        after_sequence=first[-1].sequence, timeout=0)) == ()
    reopened.shutdown()


def test_tree_start_uses_same_orchestration_with_in_memory_runtime():
    tree = build_tree()
    handle = tree.start(Task(objective="Build login"))
    result = handle.result(timeout=5)
    assert result.success and "/api/login" in result.final_output
    assert handle.status().state is ExecutionState.COMPLETED
    tree.execution_runtime.shutdown()


def test_multiple_background_trees_keep_sessions_isolated():
    runtime = ExecutionRuntime(max_workers=2, max_pending=1)
    first = runtime.submit(build_tree(), Task(objective="First login"))
    second = runtime.submit(build_tree(), Task(objective="Second login"))
    outputs = (first.result(timeout=5), second.result(timeout=5))
    assert first.execution_id != second.execution_id
    assert all(result.success for result in outputs)
    for handle, result in zip((first, second), outputs):
        assert result.task_id == handle.execution_id
        assert all(event.task_id == handle.execution_id for event in result.trace.events)
        assert result.metadata["collaboration_metrics"]["messages_sent"] == 2
    runtime.shutdown()


def test_real_process_restart_resumes_after_tool_and_collaboration(tmp_path):
    path = tmp_path / "restart.db"
    log = tmp_path / "calls.log"
    script = ("import sys; sys.path.insert(0, 'tests'); "
              "from test_execution_runtime import child_checkpoint; "
              "child_checkpoint(sys.argv[1], sys.argv[2])")
    child = subprocess.run([sys.executable, "-c", script, str(path), str(log)],
                           cwd=Path(__file__).resolve().parents[1], timeout=5, check=True)
    del child
    before = log.read_text().splitlines()
    assert before.count("tool") == 1
    assert before.count("collaboration-response") == 1
    store = SQLiteExecutionStore(path)
    old = store.get("durable-task")
    assert old.state is ExecutionState.RUNNING and old.checkpoint_sequence == 2
    runtime = ExecutionRuntime(store)
    handle = runtime.recover("durable-task", build_tree(log))
    result = handle.result(timeout=5)
    after = log.read_text().splitlines()
    assert after.count("tool") == 1
    assert after.count("collaboration-response") == 1
    assert after.count("decompose-a") == 1
    assert result.success and "/api/login" in result.final_output
    assert result.metadata["tool_metrics"]["completed"] == 1
    assert result.metadata["collaboration_metrics"]["messages_sent"] == 2
    assert len([call for call in result.usage["calls"] if call["stage"] == "manager_collaboration"]) == 1
    assert result.usage["input_tokens"] == 6
    checkpoint = store.load_checkpoint("durable-task")
    collaboration = loads(checkpoint.collaboration_json)
    messages = collaboration["messages"]
    assert len(messages) == 2 and messages[1].reply_to == messages[0].message_id
    assert collaboration["sent"]["manager-a"] == 1
    assert collaboration["sent"]["manager-b"] == 1
    assert result.trace.events[-1].event_type == "execution.completed"
    assert store.get("durable-task").checkpoint_sequence == 5
    assert handle.status().metrics["recovery_count"] == 1
    runtime.shutdown()


def test_recovery_rejects_changed_tree_and_unknown_in_flight(tmp_path):
    path = tmp_path / "restart.db"
    script = ("import sys; sys.path.insert(0, 'tests'); "
              "from test_execution_runtime import child_checkpoint; "
              "child_checkpoint(sys.argv[1], sys.argv[2])")
    subprocess.run([sys.executable, "-c", script, str(path), str(tmp_path / "log")],
                   cwd=Path(__file__).resolve().parents[1], timeout=5, check=True)
    store = SQLiteExecutionStore(path)
    runtime = ExecutionRuntime(store)
    with pytest.raises(ExecutionRecoveryBlocked, match="fingerprint"):
        runtime.recover("durable-task", build_tree(different=True))
    record = store.get("durable-task")
    store.update(record.execution_id, record.version, in_flight="tool:unknown")
    with pytest.raises(ExecutionRecoveryBlocked, match="In-flight"):
        runtime.recover("durable-task", build_tree())
    runtime.shutdown()


def test_root_synthesis_recovers_without_repeating_specialists(tmp_path):
    path = tmp_path / "root.db"
    log = tmp_path / "calls.log"
    script = ("import sys; sys.path.insert(0, 'tests'); "
              "from test_execution_runtime import child_checkpoint; "
              "child_checkpoint(sys.argv[1], sys.argv[2], 'manager_review')")
    subprocess.run([sys.executable, "-c", script, str(path), str(log)],
                   cwd=Path(__file__).resolve().parents[1], timeout=5, check=True)
    before = log.read_text().splitlines()
    assert before.count("specialist-specialist-a") == 1
    assert before.count("specialist-specialist-b") == 1
    runtime = ExecutionRuntime(SQLiteExecutionStore(path))
    result = runtime.recover("durable-task", build_tree(log)).result(timeout=5)
    assert result.success and result.final_output
    assert log.read_text().splitlines() == before
    assert result.trace.events[-1].event_type == "execution.completed"
    runtime.shutdown()


def test_manager_revision_budget_survives_review_checkpoint(tmp_path):
    path = tmp_path / "revision.db"
    log = tmp_path / "calls.log"
    script = ("import sys; sys.path.insert(0, 'tests'); "
              "from test_execution_runtime import child_checkpoint; "
              "child_checkpoint(sys.argv[1], sys.argv[2], 'manager_review', True)")
    subprocess.run([sys.executable, "-c", script, str(path), str(log)],
                   cwd=Path(__file__).resolve().parents[1], timeout=5, check=True)
    before = log.read_text().splitlines()
    assert before.count("specialist-specialist-a") == 2
    assert before.count("specialist-specialist-b") == 2
    store = SQLiteExecutionStore(path)
    state = loads(store.load_checkpoint("durable-task").state_json)
    assert all(outcome.revision_count == 1 for manager in state.manager_review_result.manager_results
               for outcome in manager.subtask_outcomes)
    runtime = ExecutionRuntime(store)
    result = runtime.recover("durable-task", build_tree(log, revise=True)).result(timeout=5)
    assert result.success and log.read_text().splitlines() == before
    runtime.shutdown()


def test_cancellation_discards_late_provider_result_and_deadline(tmp_path):
    started, release = Event(), Event()
    tree = build_tree(slow=(started, release))
    runtime = ExecutionRuntime(SQLiteExecutionStore(tmp_path / "cancel.db"), max_workers=1)
    handle = runtime.submit(tree, Task(objective="Build login"))
    assert started.wait(3)
    assert handle.cancel().state in (ExecutionState.CANCELLATION_REQUESTED, ExecutionState.CANCELLED)
    with pytest.raises(ExecutionCancelled):
        handle.result(timeout=2)
    release.set(); sleep(0.1)
    assert handle.status().state is ExecutionState.CANCELLED
    runtime.shutdown()
    reopened = ExecutionRuntime(SQLiteExecutionStore(tmp_path / "cancel.db"))
    assert reopened.handle(handle.execution_id).status().state is ExecutionState.CANCELLED
    reopened.shutdown()

    started2, release2 = Event(), Event()
    second = build_tree(slow=(started2, release2))
    runtime = ExecutionRuntime(max_workers=1)
    expiring = runtime.submit(second, Task(objective="Build login"), timeout=0.1)
    with pytest.raises(ExecutionCancelled):
        expiring.result(timeout=2)
    release2.set(); runtime.shutdown()


@pytest.mark.parametrize("operation", ["tool", "collaboration"])
def test_cancellation_stops_late_tool_or_peer_response(operation):
    started, release = Event(), Event()
    options = {"slow_tool" if operation == "tool" else "slow_collaboration":
               (started, release)}
    tree = build_tree(**options)
    runtime = ExecutionRuntime(max_workers=1)
    handle = runtime.submit(tree, Task(objective="Build login"))
    assert started.wait(3)
    handle.cancel()
    with pytest.raises(ExecutionCancelled):
        handle.result(timeout=2)
    release.set(); runtime.shutdown()
    assert handle.status().state is ExecutionState.CANCELLED
    assert all(message.type.value != "response" for message in tree.last_collaboration_messages)


def test_queue_capacity_cancel_before_start_and_store_transitions():
    started, release = Event(), Event()
    runtime = ExecutionRuntime(max_workers=1, max_pending=1)
    first = runtime.submit(build_tree(slow=(started, release)), Task(objective="first"))
    assert started.wait(3)
    second_tree = build_tree()
    second = runtime.submit(second_tree, Task(objective="second"))
    with pytest.raises(ExecutionCapacityError):
        runtime.submit(build_tree(), Task(objective="third"))
    assert second.cancel().state is ExecutionState.CANCELLED
    release.set(); first.result(timeout=5)
    assert second_tree.last_state is None
    with pytest.raises(ExecutionConflict):
        runtime.store.transition(first.execution_id, first.status().version, ExecutionState.RUNNING)
    runtime.shutdown()


def test_store_version_conflict_and_malformed_checkpoint(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "version.db")
    now = utc_now()
    record = ExecutionRecord("x", ExecutionState.QUEUED, "fingerprint", dumps(Task(objective="x")), now, now)
    store.create(record)
    running = store.transition("x", 0, ExecutionState.RUNNING)
    with pytest.raises(ExecutionConflict):
        store.update("x", 0, in_flight="stale")
    with pytest.raises(ExecutionConflict):
        store.update("x", running.version, state=ExecutionState.COMPLETED)
    with pytest.raises(ExecutionConflict):
        store.transition("x", running.version, ExecutionState.COMPLETED)
        store.transition("x", running.version, ExecutionState.RUNNING)
    with pytest.raises(ExecutionDataError):
        loads('{"$class":"untrusted.module.Code","fields":{}}')


@pytest.mark.parametrize("store_kind", ("memory", "sqlite"))
def test_terminal_transition_commits_durable_event_atomically(tmp_path, store_kind):
    store = (InMemoryExecutionStore() if store_kind == "memory" else
             SQLiteExecutionStore(tmp_path / "atomic.db"))
    now = utc_now()
    store.create(ExecutionRecord("x", ExecutionState.QUEUED, "fingerprint",
                                 dumps(Task(objective="x")), now, now))
    running = store.transition("x", 0, ExecutionState.RUNNING)
    completed = ExecutionEvent(task_id="x", event_type="execution.completed")
    with pytest.raises(ExecutionDataError):
        store.transition("x", running.version, ExecutionState.COMPLETED,
                         durable_events=(completed, ExecutionEvent(
                             task_id="other", event_type="execution.completed")))
    assert store.get("x").state is ExecutionState.RUNNING
    assert store.read_events("x") == ()
    store.transition("x", running.version, ExecutionState.COMPLETED,
                     durable_events=(completed,))
    assert store.get("x").state is ExecutionState.COMPLETED
    assert [item.event.event_type for item in store.read_events("x")] == [
        "execution.completed"]


def test_durable_failure_and_live_owner_recovery_rejection(tmp_path):
    path = tmp_path / "failed.db"
    tree = build_tree()
    def fail(request):
        raise RuntimeError("api_key=SUPER_SECRET_PHASE5_TEST_VALUE")
    tree._providers.get("manager-a").responder = fail
    runtime = ExecutionRuntime(SQLiteExecutionStore(path), max_workers=1)
    handle = runtime.submit(tree, Task(objective="Build login"))
    with pytest.raises(Exception, match="RuntimeError"):
        handle.result(timeout=5)
    assert handle.status().state is ExecutionState.FAILED
    runtime.shutdown()
    reopened = ExecutionRuntime(SQLiteExecutionStore(path))
    assert reopened.handle(handle.execution_id).status().state is ExecutionState.FAILED
    assert b"SUPER_SECRET_PHASE5_TEST_VALUE" not in path.read_bytes()
    reopened.shutdown()

    started, release = Event(), Event()
    running_tree = build_tree(slow=(started, release))
    runtime_a = ExecutionRuntime(SQLiteExecutionStore(tmp_path / "owned.db"), max_workers=1)
    active = runtime_a.submit(running_tree, Task(objective="Build login"))
    assert started.wait(3)
    runtime_b = ExecutionRuntime(SQLiteExecutionStore(tmp_path / "owned.db"))
    with pytest.raises(ExecutionRecoveryBlocked, match="live local process"):
        runtime_b.recover(active.execution_id, build_tree())
    runtime_b.cancel(active.execution_id)
    with pytest.raises(ExecutionCancelled):
        active.result(timeout=2)
    release.set(); runtime_a.shutdown(); runtime_b.shutdown()


def test_unknown_schema_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "corrupt.db"
    store = SQLiteExecutionStore(path)
    now = utc_now()
    store.create(ExecutionRecord("x", ExecutionState.QUEUED, "hash",
                                 dumps(Task(objective="x")), now, now))
    with store._connection() as db:
        db.execute("INSERT INTO checkpoints VALUES (?,?,?)", ("x", 1, '{"bad":true}'))
    with pytest.raises(ExecutionDataError):
        store.load_checkpoint("x")
    valid = ExecutionCheckpoint(1, "x", 1, now, "planning", "{}", "{}", "{}", "{}")
    encoded = json.loads(dumps(valid))
    encoded["fields"]["schema_version"] = 99
    with store._connection() as db:
        db.execute("UPDATE checkpoints SET data=? WHERE execution_id=?",
                   (json.dumps(encoded), "x"))
    with pytest.raises(ExecutionDataError, match="Malformed execution data"):
        store.load_checkpoint("x")


@pytest.mark.parametrize("durable", [False, True])
def test_execution_transition_matrix_rejects_terminal_restarts(tmp_path, durable):
    store = SQLiteExecutionStore(tmp_path / "matrix.db") if durable else InMemoryExecutionStore()
    now = utc_now()
    for name in ("complete", "fail", "cancel", "queued-cancel"):
        store.create(ExecutionRecord(name, ExecutionState.QUEUED, "hash", "{}", now, now))
    first = store.transition("complete", 0, ExecutionState.RUNNING)
    complete = store.transition("complete", first.version, ExecutionState.COMPLETED)
    second = store.transition("fail", 0, ExecutionState.RUNNING)
    failed = store.transition("fail", second.version, ExecutionState.FAILED)
    third = store.transition("cancel", 0, ExecutionState.RUNNING)
    asked = store.transition("cancel", third.version, ExecutionState.CANCELLATION_REQUESTED)
    cancelled = store.transition("cancel", asked.version, ExecutionState.CANCELLED)
    queued_cancelled = store.transition("queued-cancel", 0, ExecutionState.CANCELLED)
    for record in (complete, failed, cancelled, queued_cancelled):
        with pytest.raises(ExecutionConflict):
            store.transition(record.execution_id, record.version, ExecutionState.RUNNING)
    with pytest.raises(ExecutionConflict):
        store.transition("fail", failed.version, ExecutionState.COMPLETED)
