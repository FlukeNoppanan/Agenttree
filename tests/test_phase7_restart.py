"""Composite Phase 7 process-loss proof across prior durable work."""

import os
from pathlib import Path
import subprocess
import sys
from threading import Event
from time import monotonic, sleep

from agenttree import AgentTreeConfig, Task
from agenttree.core import (
    ExecutionRuntime, SQLiteExecutionStore, current_artifact_session,
)
from agenttree.models import ArtifactType, FileIntent
from agenttree.providers import (
    BaseProvider, ProviderCapabilities, ProviderConfig, ProviderResponse,
    ProviderStreamChunk,
)
from tests.test_execution_runtime import build_tree


class ArtifactAProvider(BaseProvider):
    def __init__(self, log):
        super().__init__(ProviderConfig("provider-specialist-a", model="fixture"))
        self.log = Path(log)

    def generate(self, request):
        with self.log.open("a") as output:
            output.write("artifact-a\n")
        current_artifact_session().create(
            type=ArtifactType.CODE, name="health.py", path="health.py",
            operation=FileIntent.CREATE, content="def health(): return 200")
        return ProviderResponse("Health implementation", self.name)


class SpecialistBStream(BaseProvider):
    def __init__(self, marker, *, crash):
        super().__init__(ProviderConfig("provider-specialist-b", model="fixture"))
        self.marker = Path(marker)
        self.crash = crash

    @property
    def capabilities(self):
        return ProviderCapabilities(streaming=True, tool_calling=True)

    def generate(self, request):
        raise AssertionError("stream expected")

    def generate_stream(self, request):
        yield ProviderStreamChunk(delta_text="Tests are starting")
        if self.crash:
            deadline = monotonic() + 5
            while not self.marker.exists() and monotonic() < deadline:
                sleep(0.01)
            os._exit(0)
        yield ProviderStreamChunk(response=ProviderResponse("Tests passed", self.name))


def composite_tree(log, marker, *, crash):
    tree = build_tree(log)
    tree._config = AgentTreeConfig(provider_streaming=True)
    by_id = {specialist.id: specialist for specialist in tree.specialists}
    for name, provider, agent_id in (
        ("provider-specialist-a", ArtifactAProvider(log), "specialist-a"),
        ("provider-specialist-b", SpecialistBStream(marker, crash=crash), "specialist-b"),
    ):
        tree._providers.unregister(name)
        tree.register_provider(provider)
        tree.bind_provider(by_id[agent_id], provider)
    return tree


def child_composite(db, log, marker):
    runtime = ExecutionRuntime(SQLiteExecutionStore(db))
    handle = runtime.submit(composite_tree(log, marker, crash=True),
                            Task(id="phase7-restart", objective="Build health endpoint"))
    for delta in handle.stream_output(timeout=6):
        if delta.agent_id == "specialist-b":
            Path(marker).write_text(delta.delta)
            break
    Event().wait(6)


def test_composite_restart_reuses_tool_collaboration_artifact_and_provider(tmp_path):
    db = tmp_path / "composite.db"
    log = tmp_path / "effects.log"
    marker = tmp_path / "observed.txt"
    script = ("import sys; from tests.test_phase7_restart import child_composite; "
              "child_composite(*sys.argv[1:])")
    subprocess.run([sys.executable, "-c", script, str(db), str(log), str(marker)],
                   cwd=Path(__file__).resolve().parents[1], timeout=10, check=True)
    assert marker.read_text() == "Tests are starting"
    store = SQLiteExecutionStore(db)
    runtime = ExecutionRuntime(store)
    before = runtime.artifacts("phase7-restart")
    assert len(before) == 1
    original = before[0]
    handle = runtime.recover("phase7-restart", composite_tree(log, marker, crash=False))
    result = handle.result(timeout=5)
    assert result.success and result.final_output
    assert len(result.artifacts) == 1
    assert result.artifacts[0].artifact_id == original.artifact_id
    assert result.artifacts[0].sha256 == original.sha256
    effects = log.read_text().splitlines()
    assert effects.count("tool") == 1
    assert effects.count("collaboration-response") == 1
    assert effects.count("artifact-a") == 1
    provider_ops = [item for item in handle.operations()
                    if item.operation_type == "provider.generate" and
                    item.agent_id == "specialist-b"]
    assert len(provider_ops) == 1 and provider_ops[0].attempt == 2
    events = handle.events(limit=1000)
    assert [item.sequence for item in events] == list(range(1, len(events) + 1))
    assert any(item.event.event_type == "artifact.committed" for item in events)
    assert events[-1].event.event_type == "execution.completed"
    runtime.shutdown()
