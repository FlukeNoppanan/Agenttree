"""Structured artifacts, local store integrity, and runtime integration."""

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
from threading import Event

import pytest

from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    ArtifactSession, ArtifactStoreError, ArtifactValidationError,
    BaseSpecialistExecutor, ExecutionRuntime, FileArtifactStore,
    InMemoryArtifactStore, RuleBasedTaskTriage, SQLiteExecutionStore,
    StaticFinalReviewer, StaticManagerReviewer, StaticTaskDecomposer,
    current_artifact_session,
    create_artifact_tool,
)
from agenttree.models import AgentResult, ArtifactType, FileIntent, SubtaskTemplate, ReviewDecision


def _session(tmp_path, *, file=False, **limits):
    store = FileArtifactStore(tmp_path / "artifacts") if file else InMemoryArtifactStore()
    return ArtifactSession("task", store, **limits)


@pytest.mark.parametrize("path", ["../secret", "a/../b", "/etc/passwd",
                                  "C:\\Windows\\system.ini", "a\\b", "a//b",
                                  "a/./b", "a\x00b", "a\nb"])
def test_artifact_rejects_unsafe_logical_paths(tmp_path, path):
    session = _session(tmp_path)
    with pytest.raises(ArtifactValidationError):
        session.create(type=ArtifactType.CODE, name="code", path=path,
                       operation=FileIntent.CREATE, content="print(1)")


def test_artifact_content_hash_types_limits_and_immutability(tmp_path):
    session = _session(tmp_path, max_artifacts=3, max_artifact_bytes=32,
                       max_total_bytes=64)
    code = session.create(type=ArtifactType.CODE, name="source", path="src/a.py",
                          operation=FileIntent.CREATE, content="print(1)",
                          media_type="text/x-python")
    assert code.size_bytes == len(b"print(1)")
    assert code.sha256 == sha256(b"print(1)").hexdigest()
    assert session.read(code.artifact_id) == b"print(1)"
    data = session.create(type=ArtifactType.JSON, name="data", content='{"b":2,"a":1}')
    assert session.read(data.artifact_id) == b'{"a":1,"b":2}'
    deletion = session.create(type=ArtifactType.FILE, name="remove", path="old.txt",
                              operation=FileIntent.DELETE)
    assert session.read(deletion.artifact_id) == b""
    with pytest.raises(ArtifactValidationError, match="limit"):
        session.create(type=ArtifactType.TEXT, name="extra", content="x")
    with pytest.raises(ArtifactValidationError, match="Patch"):
        _session(tmp_path).create(type=ArtifactType.PATCH, name="bad", content="print(1)")


def test_opt_in_artifact_tool_declares_intent_without_writing_workspace(tmp_path):
    from agenttree.core.artifacts import _active_artifact_session
    session = _session(tmp_path)
    token = _active_artifact_session.set(session)
    try:
        tool = create_artifact_tool()
        result = tool.invoke({"name": "module", "type": "code", "content": "print(1)",
                              "path": "module.py", "operation": "create"})
        assert result.success
        assert result.output["artifact_id"] == session.refs()[0].artifact_id
        assert not (tmp_path / "module.py").exists()
        assert session.read(result.output["artifact_id"]) == b"print(1)"
    finally:
        _active_artifact_session.reset(token)


def test_file_store_rejects_symlink_and_corruption(tmp_path):
    store = FileArtifactStore(tmp_path / "safe")
    session = ArtifactSession("task", store)
    ref = session.create(type=ArtifactType.FILE, name="binary", content=b"\x00\xff",
                         media_type="application/octet-stream")
    assert session.read(ref.artifact_id) == b"\x00\xff"
    folder = tmp_path / "safe" / sha256(b"task").hexdigest()
    target = folder / ref.artifact_id
    target.unlink()
    target.symlink_to(tmp_path / "outside")
    with pytest.raises(ArtifactStoreError):
        session.read(ref.artifact_id)
    target.unlink()
    target.write_bytes(b"wrong")
    with pytest.raises(ArtifactStoreError):
        session.read(ref.artifact_id)
    ref_file = folder / (ref.artifact_id + ".json")
    ref_file.unlink()
    ref_file.symlink_to(tmp_path / "outside")
    with pytest.raises(ArtifactStoreError):
        store.get_metadata("task", ref.artifact_id)


def test_artifact_ids_cannot_cross_execution_boundary(tmp_path):
    store = FileArtifactStore(tmp_path / "owned")
    ref = ArtifactSession("execution-a", store).create(
        type=ArtifactType.TEXT, name="private", content="hello")
    assert ArtifactSession("execution-b", store).refs() == ()
    with pytest.raises(KeyError):
        store.get("execution-b", ref.artifact_id)
    with pytest.raises(ArtifactValidationError, match="this execution"):
        ArtifactSession("execution-b", store).create(
            type=ArtifactType.TEXT, name="copy", content="hello",
            supersedes_artifact_id=ref.artifact_id)


def test_specialist_cannot_read_unrelated_branch_artifact(tmp_path):
    from agenttree.core.artifacts import _active_artifact_producer
    session = ArtifactSession("run", InMemoryArtifactStore(), agent_manager_ids={
        "worker-a": "manager-a", "worker-b": "manager-b",
        "manager-a": "manager-a", "manager-b": "manager-b"})
    token = _active_artifact_producer.set(("specialist", "worker-a"))
    try:
        ref = session.create(type=ArtifactType.TEXT, name="contract", content="secret")
    finally:
        _active_artifact_producer.reset(token)
    token = _active_artifact_producer.set(("specialist", "worker-b"))
    try:
        assert session.refs() == ()
        with pytest.raises(ArtifactStoreError):
            session.read(ref.artifact_id)
    finally:
        _active_artifact_producer.reset(token)
    token = _active_artifact_producer.set(("manager", "manager-a"))
    try:
        assert session.read(ref.artifact_id) == b"secret"
    finally:
        _active_artifact_producer.reset(token)


class ArtifactExecutor(BaseSpecialistExecutor):
    crash = False

    def execute(self, task, subtask, specialist):
        session = current_artifact_session()
        assert session is not None
        session.create(type=ArtifactType.PATCH, name="auth patch",
                       path="src/auth.py", operation=FileIntent.MODIFY,
                       content="--- a/src/auth.py\n+++ b/src/auth.py\n@@ -1 +1 @@\n-old\n+new\n")
        if self.crash:
            os._exit(0)
        return AgentResult(agent_id=specialist.id, success=True, output="Prepared a patch")


def build_tree(*, crash=False):
    ArtifactExecutor.crash = crash
    tree = AgentTree(root_agent=RootAgent(id="root", name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        decomposer=StaticTaskDecomposer((SubtaskTemplate("Create patch", ("work",)),)),
        manager_reviewer=StaticManagerReviewer(), final_reviewer=StaticFinalReviewer(),
        executor=ArtifactExecutor())
    manager = ManagerAgent(id="manager", name="Manager", capabilities=("manage",))
    specialist = SpecialistAgent(id="specialist", name="Specialist", capabilities=("work",))
    tree.register_manager(manager); tree.register_specialist(manager, specialist)
    return tree


class RevisionArtifactExecutor(BaseSpecialistExecutor):
    def execute(self, task, subtask, specialist):
        session = current_artifact_session()
        previous = session.refs()
        round_number = subtask.metadata.get("revision", {}).get("revision_number", 0)
        content = f"--- a/health.py\n+++ b/health.py\n@@ -1 +1 @@\n-old\n+v{round_number}\n"
        session.create(type=ArtifactType.PATCH, name=f"patch-v{round_number}",
                       path="health.py", operation=FileIntent.MODIFY, content=content,
                       supersedes_artifact_id=previous[-1].artifact_id if previous else None)
        return AgentResult(agent_id=specialist.id, success=True, output=f"v{round_number}")


def test_revision_selects_only_accepted_artifact_and_preserves_history(tmp_path):
    tree = build_tree()
    tree._executor = RevisionArtifactExecutor()
    tree._manager_reviewer = StaticManagerReviewer(outcomes=(
        (ReviewDecision.REVISE, "Improve"), (ReviewDecision.PASS, "Accepted")))
    runtime = ExecutionRuntime(SQLiteExecutionStore(tmp_path / "revision.db"))
    handle = runtime.submit(tree, Task(objective="Improve health"))
    result = handle.result(timeout=5)
    history = handle.artifacts()
    assert len(history) == 2
    assert history[0].artifact_id != history[1].artifact_id
    assert history[1].metadata["supersedes_artifact_id"] == history[0].artifact_id
    assert len(result.artifacts) == 1
    assert result.artifacts[0].artifact_id == history[1].artifact_id
    assert handle.artifact(history[0].artifact_id).endswith(b"+v0\n")
    runtime.shutdown()


def test_failed_manager_review_keeps_artifact_historical_not_final():
    tree = build_tree()
    tree._manager_reviewer = StaticManagerReviewer(decision=ReviewDecision.FAIL,
                                                    feedback="Rejected")
    result = tree.run(Task(objective="Build health"))
    assert not result.success
    assert result.artifacts == ()
    assert len(tree.artifact_store.list_for_execution(result.task_id)) == 1


class CodingArtifactExecutor(BaseSpecialistExecutor):
    def execute(self, task, subtask, specialist):
        name = "health.py" if "source" in subtask.objective else "test_health.py"
        current_artifact_session().create(type=ArtifactType.CODE, name=name,
            path=name, operation=FileIntent.CREATE,
            content="def health(): return 200" if name == "health.py" else
                    "def test_health(): assert health() == 200")
        return AgentResult(agent_id=specialist.id, success=True,
                           output=f"Prepared {name}")


def test_coding_consumer_gets_two_structured_files_without_markdown_scraping(tmp_path):
    tree = AgentTree(root_agent=RootAgent(id="root", name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate("Build source", ("work",)),
            SubtaskTemplate("Build tests", ("work",)))),
        manager_reviewer=StaticManagerReviewer(), final_reviewer=StaticFinalReviewer(),
        executor=CodingArtifactExecutor())
    manager = ManagerAgent(id="manager", name="Manager", capabilities=("manage",))
    specialist = SpecialistAgent(id="worker", name="Worker", capabilities=("work",))
    tree.register_manager(manager); tree.register_specialist(manager, specialist)
    result = tree.run(Task(objective="Add a /health endpoint and tests"))
    assert result.success
    assert "Prepared health.py" in result.final_output
    assert "Prepared test_health.py" in result.final_output
    assert {ref.path for ref in result.artifacts} == {"health.py", "test_health.py"}
    assert all(ref.sha256 and ref.size_bytes > 0 and ref.operation is FileIntent.CREATE
               for ref in result.artifacts)
    assert not (tmp_path / "health.py").exists()
    assert not (tmp_path / "test_health.py").exists()


def child_artifact(db_path):
    runtime = ExecutionRuntime(SQLiteExecutionStore(db_path), max_workers=1)
    runtime.submit(build_tree(crash=True), Task(id="job", objective="Patch auth"))
    Event().wait(10)


def test_runtime_result_and_recovery_reference_committed_artifact(tmp_path):
    db = tmp_path / "runs.db"
    script = ("import sys; from tests.test_artifacts import child_artifact; "
              "child_artifact(sys.argv[1])")
    subprocess.run([sys.executable, "-c", script, str(db)],
                   cwd=Path(__file__).resolve().parents[1], timeout=5, check=True)
    store = SQLiteExecutionStore(db)
    runtime = ExecutionRuntime(store)
    assert runtime.artifacts("job") == ()  # The creating operation has not committed.
    result = runtime.recover("job", build_tree()).result(timeout=5)
    assert result.success and result.final_output
    assert len(result.artifacts) == 1
    ref = result.artifacts[0]
    assert ref.path == "src/auth.py" and ref.operation is FileIntent.MODIFY
    assert runtime.artifact("job", ref.artifact_id).startswith(b"--- a/src/auth.py")
    assert not (tmp_path / "src" / "auth.py").exists()
    assert len(runtime.artifacts("job")) == 1
    runtime.shutdown()
    reopened = ExecutionRuntime(SQLiteExecutionStore(db))
    assert reopened.handle("job").result().artifacts == (ref,)
    assert reopened.handle("job").artifact(ref.artifact_id).startswith(b"--- a/")
    with pytest.raises(KeyError):
        reopened.artifact("other", ref.artifact_id)
    reopened.shutdown()
