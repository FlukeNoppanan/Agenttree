"""Host-visible text arrives before completion and remains bounded."""

from threading import Event

from agenttree import AgentTree, AgentTreeConfig, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    ExecutionCancelled,
    ExecutionRuntime, RuleBasedTaskTriage, StaticFinalReviewer,
    StaticManagerReviewer, StaticTaskDecomposer, create_artifact_tool,
)
from agenttree.models import SubtaskTemplate
from agenttree.providers import (
    BaseProvider, ProviderCapabilities, ProviderConfig, ProviderResponse,
    ProviderStreamChunk,
)
from agenttree.core.live_output import LiveOutputHub


class WaitingProvider(BaseProvider):
    def __init__(self, started, release):
        super().__init__(ProviderConfig("streamer", model="fixture"))
        self.started = started
        self.release = release

    @property
    def capabilities(self):
        return ProviderCapabilities(streaming=True, tool_calling=True)

    def generate(self, request):
        raise AssertionError("streaming should be selected")

    def generate_stream(self, request):
        yield ProviderStreamChunk(delta_text="Implement")
        self.started.set()
        self.release.wait(3)
        yield ProviderStreamChunk(delta_text="ing health endpoint")
        yield ProviderStreamChunk(response=ProviderResponse("Implementing health endpoint", self.name))


def build_tree(started, release, provider=None):
    tree = AgentTree(root_agent=RootAgent(id="root", name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        decomposer=StaticTaskDecomposer((SubtaskTemplate("Health endpoint", ("work",)),)),
        manager_reviewer=StaticManagerReviewer(), final_reviewer=StaticFinalReviewer(),
        config=AgentTreeConfig(provider_streaming=True))
    manager = ManagerAgent(id="manager", name="Manager", capabilities=("manage",))
    specialist = SpecialistAgent(id="specialist", name="Specialist", capabilities=("work",))
    tree.register_manager(manager)
    tree.register_specialist(manager, specialist)
    provider = provider or WaitingProvider(started, release)
    tree.register_provider(provider)
    tree.bind_provider(specialist, provider)
    return tree


class ProcessLossProvider(WaitingProvider):
    def __init__(self, marker):
        super().__init__(Event(), Event())
        self.marker = marker

    def generate_stream(self, request):
        from os import _exit
        from time import sleep, monotonic
        yield ProviderStreamChunk(delta_text="partial before crash")
        deadline = monotonic() + 5
        while not self.marker.exists() and monotonic() < deadline:
            sleep(0.01)
        _exit(0)


def child_process_loss(db, marker):
    from pathlib import Path
    from agenttree.core import SQLiteExecutionStore
    runtime = ExecutionRuntime(SQLiteExecutionStore(db))
    handle = runtime.submit(build_tree(Event(), Event(), ProcessLossProvider(Path(marker))),
                            Task(id="stream-loss", objective="Add health"))
    item = next(handle.stream_output(timeout=5))
    Path(marker).write_text(item.delta)
    Event().wait(5)


def test_host_receives_attributed_delta_while_running():
    started, release = Event(), Event()
    runtime = ExecutionRuntime()
    handle = runtime.submit(build_tree(started, release), Task(objective="Add health endpoint"))
    output = handle.stream_output(timeout=5)
    try:
        first = next(output)
        assert started.is_set()
        assert first.delta == "Implement"
        assert first.role == "specialist" and first.agent_id == "specialist"
        assert first.operation_id and first.attempt == 1
        assert handle.status().state.value == "running"
    finally:
        release.set()
    result = handle.result(timeout=5)
    assert result.success
    assert "Implementing health endpoint" in result.final_output
    assert any(item.event.event_type == "output.final.available" for item in handle.events(limit=1000))
    output.close()
    runtime.shutdown()


def test_slow_consumers_are_independent_and_bounded():
    hub = LiveOutputHub("run", capacity=2)
    a, b = hub.subscribe(), hub.subscribe()
    for word in ("one", "two", "three"):
        hub.publish("root", "r", word, "op", 1)
    hub.close()
    first = list(hub.stream(a, 0))
    second = list(hub.stream(b, 0))
    assert [item.delta for item in first] == ["two", "three"]
    assert first == second
    assert first[0].dropped_before == 1


def test_live_output_subscriber_count_is_bounded():
    import pytest
    hub = LiveOutputHub("run", max_subscribers=1)
    subscriber = hub.subscribe()
    with pytest.raises(RuntimeError, match="subscriber limit"):
        hub.subscribe()
    hub.close()
    assert list(hub.stream(subscriber, 0)) == []


def test_cancel_during_stream_cannot_commit_late_response():
    import pytest
    started, release = Event(), Event()
    runtime = ExecutionRuntime()
    handle = runtime.submit(build_tree(started, release), Task(objective="Cancel health"))
    output = handle.stream_output(timeout=5)
    assert next(output).delta == "Implement"
    handle.cancel()
    release.set()
    with pytest.raises(ExecutionCancelled):
        handle.result(timeout=5)
    assert handle.status().state.value == "cancelled"
    assert not any(item.event.event_type == "output.final.available" for item in handle.events(limit=1000))
    output.close()
    runtime.shutdown()


def test_deadline_during_stream_cannot_commit_late_response():
    import pytest
    started, release = Event(), Event()
    runtime = ExecutionRuntime()
    handle = runtime.submit(build_tree(started, release), Task(objective="Deadline health"),
                            timeout=0.1)
    output = handle.stream_output(timeout=5)
    assert next(output).delta == "Implement"
    with pytest.raises(ExecutionCancelled):
        handle.result(timeout=5)
    release.set()
    assert handle.status().state.value == "cancelled"
    output.close()
    runtime.shutdown()


def test_process_loss_after_live_delta_retries_uncertain_provider(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    from agenttree.core import SQLiteExecutionStore, OperationState
    db = tmp_path / "stream-loss.db"
    marker = tmp_path / "observed.txt"
    script = ("import sys; from tests.test_live_output import child_process_loss; "
              "child_process_loss(sys.argv[1], sys.argv[2])")
    subprocess.run([sys.executable, "-c", script, str(db), str(marker)],
                   cwd=Path(__file__).resolve().parents[1], timeout=8, check=True)
    assert marker.read_text() == "partial before crash"
    store = SQLiteExecutionStore(db)
    operations = store.list_operations("stream-loss")
    assert any(item.operation_type == "provider.generate" and
               item.state is OperationState.IN_FLIGHT for item in operations)
    release = Event(); release.set()
    runtime = ExecutionRuntime(store)
    handle = runtime.recover("stream-loss", build_tree(Event(), release))
    result = handle.result(timeout=5)
    assert result.success
    provider_ops = [item for item in handle.operations()
                    if item.operation_type == "provider.generate"]
    assert len(provider_ops) == 1 and provider_ops[0].attempt == 2
    assert result.usage["total_tokens"] is None
    assert tuple(runtime.stream_output("stream-loss", timeout=0)) == ()
    runtime.shutdown()


class ToolArtifactProvider(WaitingProvider):
    def __init__(self):
        super().__init__(Event(), Event())
        self.rounds = 0

    def generate_stream(self, request):
        self.rounds += 1
        if not request.tool_history:
            yield ProviderStreamChunk(delta_text="Preparing patch")
            yield ProviderStreamChunk(response=ProviderResponse(
                "", self.name, tool_calls=({"id": "artifact-call", "function": {
                    "name": "create_artifact", "arguments": (
                        '{"name":"health.py","type":"code","content":"def health(): pass",'
                        '"path":"health.py","operation":"create"}')
                }},)))
        else:
            yield ProviderStreamChunk(delta_text="Patch ready")
            yield ProviderStreamChunk(response=ProviderResponse("Health endpoint ready", self.name))


def test_streamed_tool_creates_artifact_after_complete_call(tmp_path):
    from agenttree.core import SQLiteExecutionStore
    provider = ToolArtifactProvider()
    tree = build_tree(Event(), Event(), provider)
    tool = create_artifact_tool()
    tree.register_tool(tool)
    specialist = next(agent for agent in tree.specialists if agent.id == "specialist")
    tree.bind_tool(specialist, tool)
    runtime = ExecutionRuntime(SQLiteExecutionStore(tmp_path / "tool-artifact.db"))
    handle = runtime.submit(tree, Task(objective="Add health"))
    output = handle.stream_output(timeout=5)
    result = handle.result(timeout=5)
    assert result.success and provider.rounds == 2
    assert len(result.artifacts) == 1
    assert result.artifacts[0].path == "health.py"
    assert handle.artifact(result.artifacts[0].artifact_id) == b"def health(): pass"
    assert not (tmp_path / "health.py").exists()
    assert any(item.event.event_type == "artifact.committed" for item in handle.events(limit=1000))
    assert [item.delta for item in output] == ["Preparing patch", "Patch ready"]
    runtime.shutdown()


def test_streamed_unauthorized_artifact_tool_is_denied():
    provider = ToolArtifactProvider()
    tree = build_tree(Event(), Event(), provider)
    tree.register_tool(create_artifact_tool())
    runtime = ExecutionRuntime()
    handle = runtime.submit(tree, Task(objective="Add health"))
    result = handle.result(timeout=5)
    assert result.artifacts == ()
    assert handle.artifacts() == ()
    assert any("denied" in item.event_type for item in result.trace.events)
    runtime.shutdown()


class RootSynthesisProvider(WaitingProvider):
    def generate_stream(self, request):
        yield ProviderStreamChunk(delta_text="Root final ")
        yield ProviderStreamChunk(delta_text="answer")
        yield ProviderStreamChunk(response=ProviderResponse("Root final answer", self.name))


def test_root_synthesis_stream_is_attributed_and_committed():
    from agenttree.core import ProviderRootSynthesizer
    release = Event(); release.set()
    tree = build_tree(Event(), release)
    tree._root_synthesizer = ProviderRootSynthesizer(RootSynthesisProvider(Event(), Event()))
    runtime = ExecutionRuntime()
    handle = runtime.submit(tree, Task(objective="Add health"))
    output = handle.stream_output(timeout=5)
    result = handle.result(timeout=5)
    root_deltas = [item for item in output if item.role == "root"]
    assert [item.delta for item in root_deltas] == ["Root final ", "answer"]
    assert all(item.agent_id == "root" and item.operation_id for item in root_deltas)
    assert result.final_output == "Root final answer"
    runtime.shutdown()


def test_streaming_and_artifact_metadata_do_not_persist_provider_secret(tmp_path):
    from agenttree.core import SQLiteExecutionStore
    secret = "SUPER_SECRET_PHASE7_ARTIFACT_VALUE"
    provider = ToolArtifactProvider()
    provider.credential = secret
    tree = build_tree(Event(), Event(), provider)
    tool = create_artifact_tool()
    tool._metadata["credential"] = secret
    tree.register_tool(tool)
    tree.bind_tool(tree.specialists[0], tool)
    db = tmp_path / "secrets.db"
    runtime = ExecutionRuntime(SQLiteExecutionStore(db))
    handle = runtime.submit(tree, Task(objective="Add health"))
    result = handle.result(timeout=5)
    assert result.artifacts
    assert secret.encode() not in db.read_bytes()
    for path in (tmp_path / "secrets.db.artifacts").rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    assert secret not in repr(handle.events(limit=1000))
    runtime.shutdown()


class NamedStreamingProvider(BaseProvider):
    def __init__(self, name):
        super().__init__(ProviderConfig(name, model="fixture"))

    @property
    def capabilities(self):
        return ProviderCapabilities(streaming=True, tool_calling=True)

    def generate(self, request):
        raise AssertionError("streaming expected")

    def generate_stream(self, request):
        yield ProviderStreamChunk(delta_text=self.name + "-1")
        yield ProviderStreamChunk(delta_text=self.name + "-2")
        yield ProviderStreamChunk(response=ProviderResponse(self.name + "-1" +
                                                            self.name + "-2", self.name))


def test_two_specialist_branches_keep_output_identity_and_local_order():
    tree = AgentTree(root_agent=RootAgent(id="root", name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        decomposer=StaticTaskDecomposer((SubtaskTemplate("Work", ("work",)),)),
        manager_reviewer=StaticManagerReviewer(), final_reviewer=StaticFinalReviewer(),
        config=AgentTreeConfig(provider_streaming=True))
    for label in ("backend", "frontend"):
        manager = ManagerAgent(id=label + "-manager", name=label,
                               capabilities=("manage",))
        specialist = SpecialistAgent(id=label + "-specialist", name=label,
                                     capabilities=("work",))
        provider = NamedStreamingProvider(label)
        tree.register_manager(manager)
        tree.register_specialist(manager, specialist)
        tree.register_provider(provider)
        tree.bind_provider(specialist, provider)
    runtime = ExecutionRuntime()
    handle = runtime.submit(tree, Task(objective="Build backend and frontend"))
    output = handle.stream_output(timeout=5)
    result = handle.result(timeout=5)
    assert result.success
    deltas = list(output)
    by_agent = {agent: [item.delta for item in deltas if item.agent_id == agent]
                for agent in ("backend-specialist", "frontend-specialist")}
    assert by_agent == {"backend-specialist": ["backend-1", "backend-2"],
                        "frontend-specialist": ["frontend-1", "frontend-2"]}
    assert all(item.role == "specialist" and item.operation_id for item in deltas)
    runtime.shutdown()
