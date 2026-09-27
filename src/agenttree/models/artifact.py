"""Immutable references to structured execution outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ArtifactType(str, Enum):
    TEXT = "text"
    CODE = "code"
    JSON = "json"
    FILE = "file"
    PATCH = "patch"
    REFERENCE = "reference"


class FileIntent(str, Enum):
    NONE = "none"
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    execution_id: str
    type: ArtifactType
    name: str
    path: str | None
    operation: FileIntent
    media_type: str
    encoding: str | None
    size_bytes: int
    sha256: str
    created_at: datetime
    producer_role: str
    producer_agent_id: str | None
    producer_operation_key: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (not isinstance(self.artifact_id, str) or len(self.artifact_id) != 64 or
                any(char not in "0123456789abcdef" for char in self.artifact_id) or
                not isinstance(self.execution_id, str) or not self.execution_id or
                not isinstance(self.type, ArtifactType) or
                not isinstance(self.operation, FileIntent) or
                not isinstance(self.name, str) or not self.name or
                not isinstance(self.media_type, str) or not self.media_type or
                not isinstance(self.size_bytes, int) or self.size_bytes < 0 or
                not isinstance(self.sha256, str) or len(self.sha256) != 64 or
                any(char not in "0123456789abcdef" for char in self.sha256) or
                not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None or
                self.producer_role not in ("root", "manager", "specialist", "tool", "host") or
                not isinstance(self.metadata, dict)):
            raise ValueError("Invalid artifact reference")
