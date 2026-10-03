"""Execution-scoped validation and creation of immutable structured outputs."""

from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from hashlib import sha256
import json
import re
from typing import Any, Mapping
from uuid import uuid4

from agenttree.core.artifact_store import ArtifactStore, ArtifactStoreError
from agenttree.core.execution_codec import dumps, loads
from agenttree.core.execution_control import check_execution
from agenttree.core.execution_store import ExecutionStore, OperationState, utc_now
from agenttree.models import ArtifactRef, ArtifactType, ExecutionEvent, FileIntent


class ArtifactValidationError(ValueError):
    """An artifact declaration violates the output contract or limits."""


_active_artifact_session: ContextVar[ArtifactSession | None] = ContextVar(
    "agenttree_artifact_session", default=None)
_active_artifact_producer: ContextVar[tuple[str, str | None] | None] = ContextVar(
    "agenttree_artifact_producer", default=None)
_active_artifact_operation: ContextVar[str | None] = ContextVar(
    "agenttree_artifact_operation", default=None)


def current_artifact_session() -> ArtifactSession | None:
    """Return the current execution's artifact helper, if one is active."""
    return _active_artifact_session.get()


def _logical_path(value: str | None, maximum: int) -> str | None:
    if value is None:
        return None
    if (not isinstance(value, str) or not value or len(value) > maximum or
            value.startswith("/") or "\\" in value or "\x00" in value or
            re.match(r"^[A-Za-z]:", value) or
            any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ArtifactValidationError("Artifact path must be a safe relative POSIX path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ArtifactValidationError("Artifact path contains an invalid segment")
    return value


class ArtifactSession:
    def __init__(self, execution_id: str, store: ArtifactStore, *,
                 execution_store: ExecutionStore | None = None,
                 agent_manager_ids: Mapping[str, str] | None = None,
                 max_artifacts: int = 32, max_artifact_bytes: int = 1_000_000,
                 max_total_bytes: int = 8_000_000, max_metadata_bytes: int = 16_384,
                 max_path_length: int = 512, max_name_length: int = 128) -> None:
        self.execution_id = execution_id
        self.store = store
        self.execution_store = execution_store
        self.agent_manager_ids = dict(agent_manager_ids or {})
        self.max_artifacts = max_artifacts
        self.max_artifact_bytes = max_artifact_bytes
        self.max_total_bytes = max_total_bytes
        self.max_metadata_bytes = max_metadata_bytes
        self.max_path_length = max_path_length
        self.max_name_length = max_name_length
        self.events: list[ExecutionEvent] = []
        self._counts: dict[str, int] = {}
        self._recorded_events = 0
        self._committed_ids: set[str] = set()
        if execution_store is not None:
            cursor = 0
            while True:
                page = execution_store.read_events(execution_id, after=cursor, limit=1000)
                for item in page:
                    if item.event.event_type == "artifact.committed":
                        artifact_id = item.event.metadata.get("artifact_id")
                        if isinstance(artifact_id, str):
                            self._committed_ids.add(artifact_id)
                if len(page) < 1000:
                    break
                cursor = page[-1].sequence

    def create(self, *, type: ArtifactType, content: Any = None, name: str,
               path: str | None = None, operation: FileIntent = FileIntent.NONE,
               media_type: str | None = None,
               metadata: Mapping[str, Any] | None = None,
               supersedes_artifact_id: str | None = None) -> ArtifactRef:
        check_execution()
        if not isinstance(type, ArtifactType) or not isinstance(operation, FileIntent):
            raise ArtifactValidationError("Invalid artifact type or file intent")
        if (not isinstance(name, str) or not name or len(name) > self.max_name_length or
                any(ord(char) < 32 or ord(char) == 127 for char in name)):
            raise ArtifactValidationError("Invalid artifact name")
        path = _logical_path(path, self.max_path_length)
        if operation is not FileIntent.NONE and path is None:
            raise ArtifactValidationError("File intent requires a logical path")
        if operation is FileIntent.DELETE and content is not None:
            raise ArtifactValidationError("Delete intent cannot carry content")
        if operation is not FileIntent.DELETE and content is None:
            raise ArtifactValidationError("Artifact content is required")
        defaults = {ArtifactType.TEXT: "text/plain", ArtifactType.CODE: "text/plain",
                    ArtifactType.JSON: "application/json", ArtifactType.FILE: "application/octet-stream",
                    ArtifactType.PATCH: "text/x-diff", ArtifactType.REFERENCE: "text/uri-list"}
        media_type = media_type or defaults[type]
        if (not isinstance(media_type, str) or len(media_type) > 128 or
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]*/[A-Za-z0-9][A-Za-z0-9.+-]*", media_type) is None):
            raise ArtifactValidationError("Invalid artifact media type")
        try:
            if operation is FileIntent.DELETE:
                encoded = b""
                encoding = None
            elif type is ArtifactType.JSON:
                value = json.loads(content) if isinstance(content, str) else content
                encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")
                encoding = "utf-8"
            elif type is ArtifactType.FILE and isinstance(content, bytes):
                encoded = content
                encoding = None
            elif isinstance(content, str):
                if type is ArtifactType.PATCH and not (
                        content.startswith("diff --git ") or
                        (content.startswith("--- ") and "\n+++ " in content)):
                    raise ArtifactValidationError("Patch must have a diff header")
                encoded = content.encode("utf-8")
                encoding = "utf-8"
            else:
                raise ArtifactValidationError("Artifact content has invalid type")
        except (TypeError, ValueError, UnicodeError) as error:
            if isinstance(error, ArtifactValidationError):
                raise
            raise ArtifactValidationError("Artifact content is invalid") from None
        if len(encoded) > self.max_artifact_bytes:
            raise ArtifactValidationError("Artifact exceeds size limit")
        try:
            safe_metadata = loads(dumps(dict(metadata or {}), max_bytes=self.max_metadata_bytes))
        except (TypeError, ValueError):
            raise ArtifactValidationError("Artifact metadata is invalid or too large") from None
        if "owner_manager_id" in safe_metadata or "supersedes_artifact_id" in safe_metadata:
            raise ArtifactValidationError("Reserved artifact metadata is runtime-controlled")
        existing = self.store.list_for_execution(self.execution_id)
        if supersedes_artifact_id is not None:
            if (not isinstance(supersedes_artifact_id, str) or
                    not any(item.artifact_id == supersedes_artifact_id and self._visible(item)
                            for item in existing)):
                raise ArtifactValidationError("Superseded artifact must belong to this execution")
            safe_metadata["supersedes_artifact_id"] = supersedes_artifact_id
        from agenttree.core.operation_journal import current_journal, _active_operation_key
        journal = current_journal()
        key = _active_operation_key.get() if journal is not None else _active_artifact_operation.get()
        count_key = key or "unscoped"
        position = self._counts.get(count_key, 0) + 1
        self._counts[count_key] = position
        artifact_id = (sha256(f"{self.execution_id}:{key}:{position}".encode()).hexdigest()
                       if key is not None else uuid4().hex * 2)
        prior = next((item for item in existing if item.artifact_id == artifact_id), None)
        if prior is None and (len(existing) >= self.max_artifacts or
                              sum(item.size_bytes for item in existing) + len(encoded) > self.max_total_bytes):
            self._event("limit_exceeded", artifact_id)
            raise ArtifactValidationError("Execution artifact limit exceeded")
        producer = _active_artifact_producer.get()
        if producer is None and journal is not None and key is not None:
            record = journal.store.get_operation(self.execution_id, key)
            producer = ("host", record.agent_id) if record is not None else None
        role, agent_id = producer if producer is not None else ("host", None)
        owner_manager_id = self.agent_manager_ids.get(agent_id) if agent_id else None
        if role == "manager":
            owner_manager_id = agent_id
        if owner_manager_id is not None:
            safe_metadata["owner_manager_id"] = owner_manager_id
        ref = ArtifactRef(artifact_id, self.execution_id, type, name, path, operation,
                          media_type, encoding, len(encoded), sha256(encoded).hexdigest(),
                          prior.created_at if prior is not None else utc_now(), role,
                          agent_id, key, safe_metadata)
        try:
            committed = self.store.put(ref, encoded)
        except (OSError, ArtifactStoreError):
            self._event("rejected", artifact_id)
            raise ArtifactStoreError("Artifact could not be committed") from None
        if prior is None:
            self._event("staged" if journal is not None and key is not None else "committed",
                        committed.artifact_id)
            if journal is None or key is None:
                self._committed_ids.add(committed.artifact_id)
        return committed

    def _event(self, status: str, artifact_id: str) -> None:
        metadata = {"artifact_id": artifact_id, "status": status}
        if status == "committed":
            ref = self.store.get_metadata(self.execution_id, artifact_id)
            metadata.update({"type": ref.type.value, "name": ref.name,
                             "path": ref.path, "operation": ref.operation.value,
                             "media_type": ref.media_type, "sha256": ref.sha256,
                             "size_bytes": ref.size_bytes,
                             "producer_role": ref.producer_role,
                             "producer_agent_id": ref.producer_agent_id,
                             "producer_operation_key": ref.producer_operation_key})
        event = ExecutionEvent(task_id=self.execution_id,
                               event_type=f"artifact.{status}",
                               metadata=metadata)
        self.events.append(event)
        if self.execution_store is not None:
            self.execution_store.append_event(self.execution_id, event)

    def refs(self) -> tuple[ArtifactRef, ...]:
        refs = self.store.list_for_execution(self.execution_id)
        if self.execution_store is None:
            return tuple(ref for ref in refs if self._visible(ref))
        result = []
        for ref in refs:
            if ref.producer_operation_key is None:
                if self._visible(ref):
                    result.append(ref)
                continue
            operation = self.execution_store.get_operation(self.execution_id,
                                                           ref.producer_operation_key)
            if operation is not None and operation.state is OperationState.COMMITTED:
                if self._visible(ref):
                    result.append(ref)
                if ref.artifact_id not in self._committed_ids:
                    self._event("committed", ref.artifact_id)
                    self._committed_ids.add(ref.artifact_id)
        return tuple(result)

    def _visible(self, ref: ArtifactRef) -> bool:
        producer = _active_artifact_producer.get()
        if producer is None or producer[0] in ("host", "root"):
            return True
        role, agent_id = producer
        owner = self.agent_manager_ids.get(agent_id) if agent_id else None
        if role == "manager":
            owner = agent_id
        if owner is None or ref.metadata.get("owner_manager_id") != owner:
            return False
        if role == "specialist" and agent_id != ref.producer_agent_id:
            return False
        if role == "tool" and agent_id != owner and agent_id != ref.producer_agent_id:
            return False
        return True

    def final_refs(self, manager_results, *, success: bool) -> tuple[ArtifactRef, ...]:
        """Select accepted latest-round artifacts; keep all drafts in history."""
        from agenttree.models import ReviewDecision
        accepted = {outcome.subtask_id: (outcome.manager_id, outcome.revision_count,
                    {execution.specialist_id for execution in
                     (outcome.revisions[-1].executions if outcome.revisions else outcome.executions)
                     if execution.agent_result is not None and execution.agent_result.success})
                    for manager in manager_results for outcome in manager.subtask_outcomes
                    if outcome.decision is ReviewDecision.PASS}
        result = []
        for ref in self.refs():
            key = ref.producer_operation_key or ""
            subtask = re.search(r"(?:^|[/:])subtask:([^:/]+)", key)
            manager = re.search(r"(?:^|[/:])manager:([^:/]+)", key)
            if subtask is not None:
                revision = re.search(r"(?:^|[/:])revision:(\d+)", key)
                number = int(revision.group(1)) if revision is not None else 0
                choice = accepted.get(subtask.group(1))
                if (choice is not None and choice[1] == number and
                        (manager is None or manager.group(1) == choice[0]) and
                        ref.producer_agent_id in choice[2]):
                    result.append(ref)
            elif success and ref.producer_role == "root":
                result.append(ref)
        return tuple(result)

    def unrecorded_events(self) -> tuple[ExecutionEvent, ...]:
        events = tuple(self.events[self._recorded_events:])
        self._recorded_events = len(self.events)
        return events

    def read(self, artifact_id: str) -> bytes:
        ref = self.store.get_metadata(self.execution_id, artifact_id)
        if ref not in self.refs():
            raise ArtifactStoreError("Artifact is not committed")
        return self.store.get(self.execution_id, artifact_id)


from agenttree.tools.function import FunctionTool


class ArtifactOutputTool(FunctionTool):
    """Trusted built-in output Tool with a separately bounded content field.

    Identity is by implementation, never by name or user-editable metadata.
    ArtifactSession remains responsible for storage, intent and aggregate quotas.
    """


def create_artifact_tool():
    """Return an opt-in model callable for structured text artifacts.

    Hosts must explicitly bind this tool to allowed agents. The artifact is a
    logical output declaration; file intents never mutate the workspace.
    """
    from agenttree.tools import FunctionTool, ToolRecoveryPolicy

    def create_artifact(name: str, type: str, content: str | None = None,
                        path: str | None = None, operation: str = "none",
                        media_type: str | None = None,
                        supersedes_artifact_id: str | None = None) -> dict[str, str]:
        session = current_artifact_session()
        if session is None:
            raise ArtifactValidationError("No active execution artifact session")
        try:
            artifact_type = ArtifactType(type)
            file_intent = FileIntent(operation)
        except ValueError:
            raise ArtifactValidationError("Invalid artifact type or file intent") from None
        ref = session.create(type=artifact_type, name=name, content=content,
                             path=path, operation=file_intent, media_type=media_type,
                             supersedes_artifact_id=supersedes_artifact_id)
        return {"artifact_id": ref.artifact_id, "sha256": ref.sha256}

    return ArtifactOutputTool(
        name="create_artifact", function=create_artifact,
        description="Store one output artifact. Put only the report/file body in content, "
                    "not conversation history, Agent state or Execution Trace. "
                    "For JSON, content must be a JSON-encoded string; for text use type=text. "
                    "Use operation=none unless declaring a file intent. "
                    "A file path and operation describe intent only; no file is changed.",
        recovery_policy=ToolRecoveryPolicy.IDEMPOTENT,
    )
