"""Operation CAS, replay, and true process-loss recovery."""

import os
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from threading import Event

import pytest

from agenttree import Task, AgentTree, RootAgent, SpecialistAgent
from agenttree.core import (
    ExecutionConflict, ExecutionRecoveryBlocked, ExecutionRuntime,
    InMemoryExecutionStore, OperationState, SQLiteExecutionStore, ExecutionState,
    ProviderRootPlanner, ProviderRootSynthesizer, RuleBasedTaskTriage,
    StaticTaskDecomposer, StaticManagerReviewer, StaticFinalReviewer,
)
from agenttree.core.execution_store import ExecutionRecord, OperationRecord, DurableEvent, utc_now
from agenttree.core.execution_codec import dumps, ExecutionDataError
from agenttree.core.operation_journal import OperationJournal, _active_operation_journal
from agenttree.models import ExecutionEvent
from agenttree.models import SubtaskTemplate
from agenttree.tools import (FunctionTool, ToolBindingRegistry, ToolCall,
                             ToolRecoveryPolicy, ToolRegistry, ToolResult,
                             ToolReconciliation, ToolReconciliationStatus)
from agenttree.tools.runtime import ToolSession
from agenttree import AgentTreeConfig, ManagerAgent

from tests.test_execution_runtime import build_tree, ScriptProvider


class ExitAfterOperation(SQLiteExecutionStore):
    target = ""

    def transition_operation(self, execution_id, operation_key, expected_version, state,
                             **kwargs):
        result = super().transition_operation(execution_id, operation_key,
                                              expected_version, state, **kwargs)
        if state is OperationState.COMMITTED and self.target in operation_key:
            os._exit(0)
        return result


def child_operation(db_path, log_path, target, revise=False):
    ExitAfterOperation.target = target
    runtime = ExecutionRuntime(ExitAfterOperation(db_path), max_workers=1)
    runtime.submit(build_tree(log_path, revise=revise),
                   Task(id="durable-task", objective="Build login"))
    Event().wait(10)


def build_root_tree(log_path):
    log_path = Path(log_path)
    def mark(label):
        with log_path.open("a") as out:
            out.write(label + "\n")
    def plan(_request):
        mark("root-plan")
        return {"delegate": True}
    def work(_request):
        mark("specialist")
        return "work"
    def synth(_request):
        mark("root-synthesis")
        return "Final answer"
    tree = AgentTree(root_agent=RootAgent(id="root", name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        decomposer=StaticTaskDecomposer((SubtaskTemplate("Work", ("work",)),)),
        manager_reviewer=StaticManagerReviewer(),
        final_reviewer=StaticFinalReviewer(),
        root_planner=ProviderRootPlanner(ScriptProvider("planner", plan)),
        root_synthesizer=ProviderRootSynthesizer(ScriptProvider("synth", synth)))
    manager = ManagerAgent(id="manager", name="Manager", capabilities=("manage",))
    specialist = SpecialistAgent(id="specialist", name="Specialist", capabilities=("work",))
    tree.register_manager(manager); tree.register_specialist(manager, specialist)
    provider = ScriptProvider("worker", work)
    tree.register_provider(provider); tree.bind_provider(specialist, provider)
    return tree


def child_root_operation(db_path, log_path, target):
    ExitAfterOperation.target = target
    runtime = ExecutionRuntime(ExitAfterOperation(db_path), max_workers=1)
    runtime.submit(build_root_tree(log_path), Task(id="durable-task", objective="Build login"))
    Event().wait(10)


def child_uncertain_tool(db_path, log_path):
    runtime = ExecutionRuntime(SQLiteExecutionStore(db_path), max_workers=1)
    runtime.submit(build_tree(log_path, tool_policy=ToolRecoveryPolicy.NON_IDEMPOTENT,
                              exit_after_tool_effect=True),
                   Task(id="durable-task", objective="Build login"))
    Event().wait(10)


def child_uncertain_provider(db_path, log_path):
    runtime = ExecutionRuntime(SQLiteExecutionStore(db_path), max_workers=1)
    runtime.submit(build_tree(log_path, exit_after_specialist_effect="specialist-a"),
                   Task(id="durable-task", objective="Build login"))
    Event().wait(10)


def child_mixed_crash(db_path, log_path):
    runtime = ExecutionRuntime(SQLiteExecutionStore(db_path), max_workers=1)
    runtime.submit(build_tree(log_path, with_root=True, revise=True,
                              exit_after_specialist_effect="specialist-b"),
                   Task(id="durable-task", objective="Build login"))
    Event().wait(10)


def child_uncertain_collaboration(db_path, log_path):
    runtime = ExecutionRuntime(SQLiteExecutionStore(db_path), max_workers=1)
    runtime.submit(build_tree(log_path, exit_after_collaboration_effect=True),
                   Task(id="durable-task", objective="Build login"))
    Event().wait(10)


def _crash(tmp_path, target, revise=False):
    db_path = tmp_path / "operations.db"
    log_path = tmp_path / "calls.log"
    script = ("import sys; "
              "from tests.test_operation_journal import child_operation; "
              "child_operation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == '1')")
    subprocess.run([sys.executable, "-c", script, str(db_path), str(log_path), target,
                    "1" if revise else "0"],
                   cwd=Path(__file__).resolve().parents[1], timeout=5, check=True)
    return db_path, log_path


@pytest.mark.parametrize("target", ["root:plan", "root:synthesis"])
def test_root_operations_are_reused_after_process_loss(tmp_path, target):
    db_path = tmp_path / "root.db"; log = tmp_path / "root.log"
    script = ("import sys; from tests.test_operation_journal import child_root_operation; "
              "child_root_operation(sys.argv[1], sys.argv[2], sys.argv[3])")
    subprocess.run([sys.executable, "-c", script, str(db_path), str(log), target],
                   cwd=Path(__file__).resolve().parents[1], timeout=5, check=True)
    before = log.read_text().splitlines()
    assert before.count("root-plan") == 1
    if target == "root:synthesis":
        assert before.count("root-synthesis") == 1
    runtime = ExecutionRuntime(SQLiteExecutionStore(db_path))
    result = runtime.recover("durable-task", build_root_tree(log)).result(timeout=5)
    assert result.success and result.final_output == "Final answer"
    after = log.read_text().splitlines()
    assert after.count("root-plan") == 1
    assert after.count("root-synthesis") == 1
    runtime.shutdown()


def test_specialist_committed_inside_phase_is_reused_after_process_loss(tmp_path):
    db_path, log = _crash(tmp_path, "specialist:specialist-a")
    before = log.read_text().splitlines()
    assert before.count("specialist-specialist-a") == 1
    assert "specialist-specialist-b" not in before
    store = SQLiteExecutionStore(db_path)
    runtime = ExecutionRuntime(store)
    result = runtime.recover("durable-task", build_tree(log)).result(timeout=5)
    after = log.read_text().splitlines()
    assert result.success
    assert after.count("specialist-specialist-a") == 1
    assert after.count("specialist-specialist-b") == 1
    operations = store.list_operations("durable-task")
    assert any(op.state is OperationState.COMMITTED and
               "specialist:specialist-a" in op.operation_key for op in operations)
    runtime.shutdown()


def test_tool_commit_is_not_replayed_inside_incomplete_delegation(tmp_path):
    db_path, log = _crash(tmp_path, "tool:schema-call")
    assert log.read_text().splitlines().count("tool") == 1
    store = SQLiteExecutionStore(db_path)
    runtime = ExecutionRuntime(store)
    result = runtime.recover("durable-task", build_tree(log)).result(timeout=5)
    assert result.success
    assert log.read_text().splitlines().count("tool") == 1
    assert result.metadata["tool_metrics"]["completed"] == 1
    runtime.shutdown()


def test_non_idempotent_tool_side_effect_blocks_recovery(tmp_path):
    db_path = tmp_path / "uncertain.db"; log = tmp_path / "effects.log"
    script = ("import sys; from tests.test_operation_journal import child_uncertain_tool; "
              "child_uncertain_tool(sys.argv[1], sys.argv[2])")
    subprocess.run([sys.executable, "-c", script, str(db_path), str(log)],
                   cwd=Path(__file__).resolve().parents[1], timeout=5, check=True)
    assert log.read_text().splitlines().count("tool") == 1
    store = SQLiteExecutionStore(db_path)
    runtime = ExecutionRuntime(store)
    with pytest.raises(ExecutionRecoveryBlocked) as caught:
        runtime.recover("durable-task", build_tree(log,
            tool_policy=ToolRecoveryPolicy.NON_IDEMPOTENT))
    assert caught.value.code == "UNCERTAIN_TOOL_OUTCOME"
    assert "tool:schema-call" in caught.value.operation_key
    assert log.read_text().splitlines().count("tool") == 1
    assert store.get("durable-task").state is ExecutionState.RUNNING
    assert any(item.event.event_type == "execution.recovery.blocked" and
               item.event.metadata["code"] == "UNCERTAIN_TOOL_OUTCOME"
               for item in store.read_events("durable-task", limit=1000))
    runtime.shutdown()


def test_uncertain_provider_gets_new_physical_attempt(tmp_path):
    db_path = tmp_path / "provider.db"; log = tmp_path / "calls.log"
    script = ("import sys; from tests.test_operation_journal import child_uncertain_provider; "
              "child_uncertain_provider(sys.argv[1], sys.argv[2])")
    subprocess.run([sys.executable, "-c", script, str(db_path), str(log)],
                   cwd=Path(__file__).resolve().parents[1], timeout=5, check=True)
    assert log.read_text().splitlines().count("specialist-specialist-a") == 1
    store = SQLiteExecutionStore(db_path)
    runtime = ExecutionRuntime(store)
    result = runtime.recover("durable-task", build_tree(log)).result(timeout=5)
    assert result.success
    assert log.read_text().splitlines().count("specialist-specialist-a") == 2
    provider = next(op for op in store.list_operations("durable-task")
                    if op.operation_type == "provider.generate" and
                    "specialist:specialist-a" in op.operation_key)
    assert [attempt.outcome for attempt in store.list_attempts(
        "durable-task", provider.operation_key)] == ["uncertain", "committed"]
    assert result.usage["total_tokens"] == 12  # Lost provider attempt has no invented usage.
    runtime.shutdown()


def test_mixed_execution_recovers_at_operation_boundary(tmp_path):
    db_path = tmp_path / "mixed.db"; log = tmp_path / "mixed.log"
    script = ("import sys; from tests.test_operation_journal import child_mixed_crash; "
              "child_mixed_crash(sys.argv[1], sys.argv[2])")
    subprocess.run([sys.executable, "-c", script, str(db_path), str(log)],
                   cwd=Path(__file__).resolve().parents[1], timeout=5, check=True)
    before = log.read_text().splitlines()
    for label in ("root-plan", "tool", "collaboration-response", "decompose-a",
                  "decompose-b", "specialist-specialist-a", "specialist-specialist-b"):
        assert before.count(label) == 1
    store = SQLiteExecutionStore(db_path)
    before_events = store.read_events("durable-task", limit=1000)
    runtime = ExecutionRuntime(store)
    handle = runtime.recover("durable-task", build_tree(log, with_root=True, revise=True))
    result = handle.result(timeout=5)
    after = log.read_text().splitlines()
    assert result.success and result.final_output == "Completed login design"
    for label in ("root-plan", "tool", "collaboration-response", "decompose-a", "decompose-b"):
        assert after.count(label) == 1
    assert after.count("specialist-specialist-a") == 2  # One initial, one revision.
    assert after.count("specialist-specialist-b") == 3  # Lost attempt, retry, revision.
    assert after.count("specialist-specialist-frontend") == 2
    assert after.count("root-synthesis") == 1
    assert result.metadata["tool_metrics"]["completed"] == 1
    assert result.metadata["collaboration_metrics"]["messages_sent"] == 2
    assert all(outcome.revision_count == 1 for manager in result.manager_results
               for outcome in manager.subtask_outcomes)
    events = handle.events(limit=1000)
    assert events[len(before_events)].sequence == before_events[-1].sequence + 1
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    runtime.shutdown()


@pytest.mark.parametrize("target,revise,expected_calls", [
    ("review:1", False, 1),
    ("revision:1:specialist:specialist-a", True, 2),
])
def test_review_and_revision_commits_are_reused(tmp_path, target, revise, expected_calls):
    db_path, log = _crash(tmp_path, target, revise=revise)
    before = log.read_text().splitlines()
    assert before.count("specialist-specialist-a") == expected_calls
    store = SQLiteExecutionStore(db_path)
    operation = next(op for op in store.list_operations("durable-task")
                     if target in op.operation_key and op.state is OperationState.COMMITTED)
    runtime = ExecutionRuntime(store)
    result = runtime.recover("durable-task", build_tree(log, revise=revise)).result(timeout=5)
    assert result.success
    assert log.read_text().splitlines().count("specialist-specialist-a") == expected_calls
    assert len(store.list_attempts("durable-task", operation.operation_key)) == 1
    runtime.shutdown()


def test_collaboration_commit_restores_messages_without_peer_replay(tmp_path):
    db_path, log = _crash(tmp_path, "collaboration:1")
    assert log.read_text().splitlines().count("collaboration-response") == 1
    store = SQLiteExecutionStore(db_path)
    runtime = ExecutionRuntime(store)
    result = runtime.recover("durable-task", build_tree(log)).result(timeout=5)
    assert result.success
    assert log.read_text().splitlines().count("collaboration-response") == 1
    assert result.metadata["collaboration_metrics"]["messages_sent"] == 2
    runtime.shutdown()


def test_uncertain_peer_response_retries_without_double_budget(tmp_path):
    db_path = tmp_path / "peer.db"; log = tmp_path / "peer.log"
    script = ("import sys; from tests.test_operation_journal import child_uncertain_collaboration; "
              "child_uncertain_collaboration(sys.argv[1], sys.argv[2])")
    subprocess.run([sys.executable, "-c", script, str(db_path), str(log)],
                   cwd=Path(__file__).resolve().parents[1], timeout=5, check=True)
    assert log.read_text().splitlines().count("collaboration-response") == 1
    store = SQLiteExecutionStore(db_path)
    runtime = ExecutionRuntime(store)
    result = runtime.recover("durable-task", build_tree(log)).result(timeout=5)
    assert result.success
    assert log.read_text().splitlines().count("collaboration-response") == 2
    assert result.metadata["collaboration_metrics"]["messages_sent"] == 2
    provider = next(op for op in store.list_operations("durable-task")
                    if op.operation_type == "provider.generate" and "collaboration:1" in op.operation_key)
    assert [attempt.outcome for attempt in store.list_attempts(
        "durable-task", provider.operation_key)] == ["uncertain", "committed"]
    runtime.shutdown()


def test_durable_event_order_has_root_commit_before_completion(tmp_path):
    runtime = ExecutionRuntime(SQLiteExecutionStore(tmp_path / "events.db"))
    handle = runtime.submit(build_tree(), Task(objective="Build login"))
    assert handle.result(timeout=5).success
    events = handle.events(limit=1000)
    assert [item.sequence for item in events] == list(range(1, len(events) + 1))
    root = next(item.sequence for item in events if item.event.event_type == "operation.committed"
                and item.event.metadata["operation_type"] == "root.synthesis")
    completed = next(item.sequence for item in events
                     if item.event.event_type == "execution.completed")
    assert root < completed
    runtime.shutdown()


def test_operation_and_event_journal_do_not_persist_secret_sentinel(tmp_path):
    path = tmp_path / "safe.db"
    tree = build_tree()
    tool = tree.tools[0]
    tool._metadata["api_key"] = "SUPER_SECRET_PHASE6_OPERATION_VALUE"
    tree.providers[0].credential = "SUPER_SECRET_PHASE6_OPERATION_VALUE"
    runtime = ExecutionRuntime(SQLiteExecutionStore(path))
    assert runtime.submit(tree, Task(objective="Build login")).result(timeout=5).success
    assert b"SUPER_SECRET_PHASE6_OPERATION_VALUE" not in path.read_bytes()
    runtime.shutdown()


def test_uncertain_tool_policy_blocks_or_replays_with_stable_identity():
    for policy in (ToolRecoveryPolicy.UNKNOWN, ToolRecoveryPolicy.PURE,
                   ToolRecoveryPolicy.IDEMPOTENT):
        store = InMemoryExecutionStore()
        now = utc_now()
        store.create(ExecutionRecord("job", ExecutionState.QUEUED,
                                     "fingerprint", "{}", now, now))
        seen = []
        from agenttree.core import current_operation_idempotency_key

        def action():
            seen.append(current_operation_idempotency_key())
            return "done"

        tool = FunctionTool(name="work", function=action, tool_id="work-id",
                            recovery_policy=policy)
        registry = ToolRegistry(); registry.register(tool)
        bindings = ToolBindingRegistry(); bindings.assign("manager", tool.id)
        session = ToolSession(registry, bindings,
                              (ManagerAgent(id="manager", name="Manager"),),
                              AgentTreeConfig(), "job")
        call = ToolCall("call-1", "work", "manager", "manager", {}, "job")
        journal = OperationJournal(store, "job"); journal.start_phase("execution")
        key = "execution/tool:call-1"
        input_value = {"agent_id": "manager", "tool_name": "work",
                       "call_id": "call-1", "arguments": {}}
        op = OperationRecord("job", key, "tool.call", "execution",
                             OperationState.PREPARED,
                             sha256(dumps(input_value).encode()).hexdigest(),
                             policy.value, now, now,
                             idempotency_key=sha256(f"job:{key}".encode()).hexdigest())
        store.create_operation(op)
        store.transition_operation("job", key, 0, OperationState.IN_FLIGHT)
        token = _active_operation_journal.set(journal)
        try:
            if policy is ToolRecoveryPolicy.UNKNOWN:
                with pytest.raises(ExecutionRecoveryBlocked, match="Uncertain"):
                    session.invoke(call)
                assert seen == []
                assert store.get_operation("job", key).state is OperationState.UNCERTAIN
            else:
                assert session.invoke(call).output == "done"
                assert len(seen) == 1 and seen[0] == store.get_operation("job", key).idempotency_key
                assert [item.outcome for item in store.list_attempts("job", key)] == [
                    "uncertain", "committed"]
        finally:
            _active_operation_journal.reset(token)


@pytest.mark.parametrize("status,expected_calls", [
    (ToolReconciliationStatus.COMPLETED, 0),
    (ToolReconciliationStatus.NOT_COMPLETED, 1),
    (ToolReconciliationStatus.UNKNOWN, 0),
])
def test_tool_reconciliation_controls_retry(status, expected_calls):
    store = InMemoryExecutionStore()
    now = utc_now()
    store.create(ExecutionRecord("job", ExecutionState.QUEUED,
                                 "fingerprint", "{}", now, now))
    called = []

    class ReconcilingTool(FunctionTool):
        def reconcile(self, operation):
            assert operation.operation_key == "execution/tool:call-1"
            return ToolReconciliation(status,
                ToolResult(self.id, True, output="confirmed")
                if status is ToolReconciliationStatus.COMPLETED else None)

    def action():
        called.append(1)
        return "retried"

    tool = ReconcilingTool(name="work", function=action, tool_id="work-id",
                           recovery_policy=ToolRecoveryPolicy.RECONCILABLE)
    registry = ToolRegistry(); registry.register(tool)
    bindings = ToolBindingRegistry(); bindings.assign("manager", tool.id)
    session = ToolSession(registry, bindings, (ManagerAgent(id="manager", name="Manager"),),
                          AgentTreeConfig(), "job")
    call = ToolCall("call-1", "work", "manager", "manager", {}, "job")
    key = "execution/tool:call-1"
    digest = sha256(dumps({"agent_id": "manager", "tool_name": "work",
                           "call_id": "call-1", "arguments": {}}).encode()).hexdigest()
    op = OperationRecord("job", key, "tool.call", "execution", OperationState.PREPARED,
                         digest, "reconcilable", now, now)
    store.create_operation(op)
    store.transition_operation("job", key, 0, OperationState.IN_FLIGHT)
    journal = OperationJournal(store, "job"); journal.start_phase("execution")
    token = _active_operation_journal.set(journal)
    try:
        if status is ToolReconciliationStatus.UNKNOWN:
            with pytest.raises(ExecutionRecoveryBlocked):
                session.invoke(call)
        else:
            result = session.invoke(call)
            assert result.output == ("confirmed" if expected_calls == 0 else "retried")
        assert len(called) == expected_calls
    finally:
        _active_operation_journal.reset(token)


@pytest.mark.parametrize("store_type", [InMemoryExecutionStore, SQLiteExecutionStore])
def test_operation_transitions_and_event_cursor(store_type, tmp_path):
    store = store_type() if store_type is InMemoryExecutionStore else store_type(tmp_path / "store.db")
    now = utc_now()
    store.create(ExecutionRecord("job", ExecutionState.QUEUED,
                                 "fingerprint", "{}", now, now))
    operation = OperationRecord("job", "phase/op", "provider.generate", "phase",
                                OperationState.PREPARED, "a" * 64, "retry", now, now)
    store.create_operation(operation)
    with pytest.raises(ExecutionConflict):
        store.create_operation(operation)
    started = store.transition_operation("job", "phase/op", 0, OperationState.IN_FLIGHT)
    with pytest.raises(ExecutionConflict):
        store.transition_operation("job", "phase/op", 0, OperationState.COMMITTED,
                                   result_json='"done"')
    uncertain = store.transition_operation("job", "phase/op", started.version,
                                           OperationState.UNCERTAIN)
    restarted = store.transition_operation("job", "phase/op", uncertain.version,
                                           OperationState.IN_FLIGHT)
    store.transition_operation("job", "phase/op", restarted.version,
                               OperationState.COMMITTED, result_json='"done"')
    assert [attempt.outcome for attempt in store.list_attempts("job", "phase/op")] == [
        "uncertain", "committed"]
    assert store.get_operation("job", "phase/op").result_json == '"done"'
    first = store.append_event("job", ExecutionEvent(task_id="job", event_type="operation.started"))
    second = store.append_event("job", ExecutionEvent(task_id="job", event_type="operation.committed"))
    assert (first.sequence, second.sequence) == (6, 7)
    assert store.read_events("job", after=6)[0].sequence == 7


def test_operation_input_mismatch_and_malformed_event_fail_safely():
    store = InMemoryExecutionStore()
    now = utc_now()
    store.create(ExecutionRecord("job", ExecutionState.QUEUED,
                                 "fingerprint", "{}", now, now))
    journal = OperationJournal(store, "job"); journal.start_phase("planning")
    assert journal.run("planning/work", "internal", {"value": 1}, lambda: "first") == "first"
    with pytest.raises(ExecutionRecoveryBlocked) as mismatch:
        journal.run("planning/work", "internal", {"value": 2}, lambda: "second")
    assert mismatch.value.code == "OPERATION_INPUT_MISMATCH"
    with pytest.raises(ExecutionDataError):
        DurableEvent("job", 1, {"event_type": "invalid"})
    with pytest.raises(ExecutionDataError):
        store.append_event("job", ExecutionEvent(task_id="another", event_type="wrong"))
