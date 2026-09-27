"""Immutable execution artifact storage with safe local file layout."""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from agenttree.core.execution_codec import dumps, loads
from agenttree.models import ArtifactRef


class ArtifactStoreError(RuntimeError):
    """An artifact is missing, corrupt, conflicting, or inaccessible."""


def _verify(ref: ArtifactRef, content: bytes) -> None:
    if not isinstance(ref, ArtifactRef) or not isinstance(content, bytes):
        raise TypeError("Artifact store requires an ArtifactRef and bytes")
    if len(content) != ref.size_bytes or sha256(content).hexdigest() != ref.sha256:
        raise ArtifactStoreError("Artifact content does not match committed metadata")


class ArtifactStore(ABC):
    @abstractmethod
    def put(self, ref: ArtifactRef, content: bytes) -> ArtifactRef: ...

    @abstractmethod
    def get(self, execution_id: str, artifact_id: str) -> bytes: ...

    @abstractmethod
    def get_metadata(self, execution_id: str, artifact_id: str) -> ArtifactRef: ...

    @abstractmethod
    def list_for_execution(self, execution_id: str) -> tuple[ArtifactRef, ...]: ...

    def exists(self, execution_id: str, artifact_id: str) -> bool:
        try:
            self.get_metadata(execution_id, artifact_id)
        except KeyError:
            return False
        return True


class InMemoryArtifactStore(ArtifactStore):
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], tuple[ArtifactRef, bytes]] = {}
        self._lock = RLock()

    def put(self, ref: ArtifactRef, content: bytes) -> ArtifactRef:
        _verify(ref, content)
        safe_ref = loads(dumps(ref, max_bytes=65_536))
        key = ref.execution_id, ref.artifact_id
        with self._lock:
            prior = self._items.get(key)
            if prior is not None:
                if prior != (safe_ref, content):
                    raise ArtifactStoreError("Artifact identity already has different content")
                return prior[0]
            self._items[key] = safe_ref, content
            return safe_ref

    def get(self, execution_id: str, artifact_id: str) -> bytes:
        with self._lock:
            ref, content = self._items[execution_id, artifact_id]
            _verify(ref, content)
            return content

    def get_metadata(self, execution_id: str, artifact_id: str) -> ArtifactRef:
        with self._lock:
            return loads(dumps(self._items[execution_id, artifact_id][0]))

    def list_for_execution(self, execution_id: str) -> tuple[ArtifactRef, ...]:
        with self._lock:
            return tuple(loads(dumps(ref)) for (owner, _), (ref, _) in self._items.items()
                         if owner == execution_id)


class FileArtifactStore(ArtifactStore):
    """Content and metadata under a dedicated root; logical paths are never used."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ArtifactStoreError("Artifact root must be a real directory")
        self._lock = RLock()

    @staticmethod
    def _id(value: str) -> str:
        if not isinstance(value, str) or len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value):
            raise ValueError("Invalid artifact ID")
        return value

    def _directory(self, execution_id: str, *, create: bool) -> int:
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("Invalid execution ID")
        folder = sha256(execution_id.encode()).hexdigest()
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        root_fd = os.open(self.root, flags)
        try:
            if create:
                try:
                    os.mkdir(folder, mode=0o700, dir_fd=root_fd)
                except FileExistsError:
                    pass
            return os.open(folder, flags, dir_fd=root_fd)
        finally:
            os.close(root_fd)

    @staticmethod
    def _read(fd: int, name: str, limit: int) -> bytes:
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(file_fd, "rb") as source:
            return source.read(limit + 1)

    @staticmethod
    def _write_atomic(fd: int, name: str, content: bytes) -> None:
        temp = f".{uuid4().hex}.tmp"
        handle = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=fd)
        try:
            with os.fdopen(handle, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, name, src_dir_fd=fd, dst_dir_fd=fd)
            os.fsync(fd)
        finally:
            try:
                os.unlink(temp, dir_fd=fd)
            except FileNotFoundError:
                pass

    def put(self, ref: ArtifactRef, content: bytes) -> ArtifactRef:
        _verify(ref, content)
        artifact_id = self._id(ref.artifact_id)
        raw_meta = dumps(ref, max_bytes=65_536).encode("utf-8")
        with self._lock:
            fd = self._directory(ref.execution_id, create=True)
            try:
                try:
                    prior = self._metadata(fd, ref.execution_id, artifact_id)
                except KeyError:
                    prior = None
                if prior is not None:
                    if prior != ref:
                        raise ArtifactStoreError("Artifact identity already has different metadata")
                    _verify(prior, self._read(fd, artifact_id, prior.size_bytes))
                    return prior
                self._write_atomic(fd, artifact_id, content)
                self._write_atomic(fd, artifact_id + ".json", raw_meta)
                return ref
            except OSError as error:
                raise ArtifactStoreError("Artifact storage write failed") from error
            finally:
                os.close(fd)

    def _metadata(self, fd: int, execution_id: str, artifact_id: str) -> ArtifactRef:
        try:
            raw = self._read(fd, artifact_id + ".json", 65_536)
        except FileNotFoundError:
            raise KeyError(artifact_id) from None
        try:
            ref = loads(raw.decode("utf-8"), max_bytes=65_536)
        except Exception as error:
            raise ArtifactStoreError("Artifact metadata is corrupt") from error
        if (not isinstance(ref, ArtifactRef) or ref.execution_id != execution_id or
                ref.artifact_id != artifact_id):
            raise ArtifactStoreError("Artifact metadata ownership is invalid")
        return ref

    def get_metadata(self, execution_id: str, artifact_id: str) -> ArtifactRef:
        artifact_id = self._id(artifact_id)
        try:
            fd = self._directory(execution_id, create=False)
        except FileNotFoundError:
            raise KeyError(artifact_id) from None
        try:
            return self._metadata(fd, execution_id, artifact_id)
        except OSError as error:
            raise ArtifactStoreError("Artifact metadata cannot be read") from error
        finally:
            os.close(fd)

    def get(self, execution_id: str, artifact_id: str) -> bytes:
        ref = self.get_metadata(execution_id, artifact_id)
        fd = self._directory(execution_id, create=False)
        try:
            try:
                content = self._read(fd, artifact_id, ref.size_bytes)
            except FileNotFoundError:
                raise ArtifactStoreError("Committed artifact content is missing") from None
            _verify(ref, content)
            return content
        except OSError as error:
            raise ArtifactStoreError("Artifact content cannot be read") from error
        finally:
            os.close(fd)

    def list_for_execution(self, execution_id: str) -> tuple[ArtifactRef, ...]:
        try:
            fd = self._directory(execution_id, create=False)
        except FileNotFoundError:
            return ()
        try:
            refs = [self._metadata(fd, execution_id, name[:-5])
                    for name in os.listdir(fd)
                    if name.endswith(".json") and len(name) == 69 and
                    all(char in "0123456789abcdef" for char in name[:-5])]
            return tuple(sorted(refs, key=lambda item: (item.created_at, item.artifact_id)))
        finally:
            os.close(fd)
