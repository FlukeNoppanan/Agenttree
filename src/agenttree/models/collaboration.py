"""Provider-neutral, execution-scoped Manager coordination contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any
from uuid import UUID, uuid4


class ManagerMessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    CONTEXT = "context"
    REVIEW_REQUEST = "review_request"
    REVIEW_RESPONSE = "review_response"


class ManagerMessageStatus(str, Enum):
    DELIVERED = "delivered"
    RESPONDED = "responded"
    FAILED = "failed"


_SECRET = re.compile(r"(?i)(authorization\s*:\s*bearer\s+\S+|bearer\s+\S+|(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+)")
_PRIVATE_KEYS = ("password", "secret", "token", "credential", "authorization", "api_key")


def safe_message_text(value: str) -> str:
    """Remove common credential forms from message data and repr snapshots."""
    return _SECRET.sub("[REDACTED]", value)


def _safe_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if any(part in key.casefold() for part in _PRIVATE_KEYS)
                else _safe_metadata(item) for key, item in value.items() if isinstance(key, str)}
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item) for item in value]
    if isinstance(value, str):
        return safe_message_text(value)
    return deepcopy(value)


@dataclass(frozen=True)
class ManagerMessage:
    """One validated coordination operation, never a Tool invocation."""

    execution_id: str
    from_manager_id: str
    to_manager_id: str
    type: ManagerMessageType
    subject: str
    content: str = field(repr=False)
    thread_id: str = field(default_factory=lambda: str(uuid4()))
    message_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: str | None = None
    reply_to: str | None = None
    status: ManagerMessageStatus = ManagerMessageStatus.DELIVERED
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in ("execution_id", "from_manager_id", "to_manager_id", "thread_id", "message_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty text")
        if not isinstance(self.type, ManagerMessageType):
            raise TypeError("type must be a ManagerMessageType")
        if not isinstance(self.status, ManagerMessageStatus):
            raise TypeError("status must be a ManagerMessageStatus")
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("subject must be nonempty text")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be nonempty text")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        for name in ("correlation_id", "reply_to"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be nonempty text or None")
        if self.type in (ManagerMessageType.RESPONSE, ManagerMessageType.REVIEW_RESPONSE):
            if self.reply_to is None or self.correlation_id is None:
                raise ValueError("responses require reply_to and correlation_id")
        elif self.reply_to is not None:
            raise ValueError("only responses may have reply_to")
        object.__setattr__(self, "subject", safe_message_text(self.subject.strip()))
        object.__setattr__(self, "content", safe_message_text(self.content.strip()))
        safe_metadata = _safe_metadata(self.metadata)
        try:
            json.dumps(safe_metadata, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            raise ValueError("metadata must contain JSON-compatible values") from None
        object.__setattr__(self, "metadata", safe_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {"execution_id": self.execution_id, "from_manager_id": self.from_manager_id,
                "to_manager_id": self.to_manager_id, "type": self.type.value,
                "subject": self.subject, "content": self.content,
                "thread_id": self.thread_id, "message_id": self.message_id,
                "correlation_id": self.correlation_id, "reply_to": self.reply_to,
                "status": self.status.value, "metadata": deepcopy(self.metadata)}


@dataclass(frozen=True)
class ManagerCollaborationRequest:
    """Validated model request before runtime policy and routing."""

    target_manager_id: str
    type: ManagerMessageType
    subject: str
    content: str
    thread_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, ManagerMessageType) or self.type not in (
            ManagerMessageType.REQUEST, ManagerMessageType.CONTEXT,
            ManagerMessageType.REVIEW_REQUEST,
        ):
            raise ValueError("Invalid collaboration request type")
        for name in ("target_manager_id", "subject", "content"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Collaboration fields must be nonempty text")
        if self.thread_id is not None:
            try:
                UUID(self.thread_id)
            except (TypeError, ValueError):
                raise ValueError("thread_id must be a valid UUID") from None

    @classmethod
    def from_data(cls, data: Any) -> ManagerCollaborationRequest:
        if not isinstance(data, dict) or set(data) - {"target_manager_id", "type", "subject", "content", "thread_id"}:
            raise ValueError("Collaboration request must contain only known fields")
        try:
            kind = ManagerMessageType(data["type"])
        except (KeyError, ValueError, TypeError):
            raise ValueError("Unknown collaboration message type") from None
        if kind not in (ManagerMessageType.REQUEST, ManagerMessageType.CONTEXT,
                        ManagerMessageType.REVIEW_REQUEST):
            raise ValueError("Managers may request, share context, or ask for review")
        for name in ("target_manager_id", "subject", "content"):
            if not isinstance(data.get(name), str) or not data[name].strip():
                raise ValueError("Collaboration fields must be nonempty text")
        thread_id = data.get("thread_id")
        if thread_id is not None and (not isinstance(thread_id, str) or not thread_id.strip()):
            raise ValueError("thread_id must be nonempty text")
        if thread_id is not None:
            try:
                UUID(thread_id)
            except ValueError:
                raise ValueError("thread_id must be a valid UUID") from None
        return cls(data["target_manager_id"].strip(), kind,
                   safe_message_text(data["subject"].strip()),
                   safe_message_text(data["content"].strip()), thread_id)


@dataclass(frozen=True)
class ManagerCollaborationOutcome:
    """Small safe result returned to the requesting Manager's next decision turn."""

    success: bool
    request: ManagerMessage | None = None
    response: ManagerMessage | None = None
    error_type: str | None = None

    def to_context(self) -> dict[str, Any]:
        result = {"success": self.success, "error_type": self.error_type}
        if self.request is not None:
            result.update({"message_id": self.request.message_id,
                           "thread_id": self.request.thread_id,
                           "target_manager_id": self.request.to_manager_id,
                           "type": self.request.type.value})
        if self.response is not None:
            result["response"] = {"message_id": self.response.message_id,
                                  "reply_to": self.response.reply_to,
                                  "content": self.response.content,
                                  "metadata": deepcopy(self.response.metadata)}
        return result
