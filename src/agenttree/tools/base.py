"""Common interface for explicitly registered executable tools."""

from abc import ABC, abstractmethod
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Mapping
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4

from agenttree.tools.models import ToolInputSpec, ToolResult


class ToolRecoveryPolicy(str, Enum):
    PURE = "pure"
    IDEMPOTENT = "idempotent"
    RECONCILABLE = "reconcilable"
    NON_IDEMPOTENT = "non_idempotent"
    UNKNOWN = "unknown"


class ToolReconciliationStatus(str, Enum):
    COMPLETED = "completed"
    NOT_COMPLETED = "not_completed"
    UNKNOWN = "unknown"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolReconciliation:
    status: ToolReconciliationStatus
    result: ToolResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ToolReconciliationStatus):
            raise TypeError("Invalid reconciliation status")
        if self.status is ToolReconciliationStatus.COMPLETED and not isinstance(self.result, ToolResult):
            raise ValueError("Completed reconciliation requires a ToolResult")
        if self.status is not ToolReconciliationStatus.COMPLETED and self.result is not None:
            raise ValueError("Only completed reconciliation can carry a result")


class BaseTool(ABC):
    """Base interface for local and future externally backed tools."""

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        tool_id: str | None = None,
        input_spec: ToolInputSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
        enabled: bool = True,
        recovery_policy: ToolRecoveryPolicy = ToolRecoveryPolicy.UNKNOWN,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(description, str):
            raise TypeError("description must be a string")
        prepared_id = str(uuid4()) if tool_id is None else tool_id
        if not isinstance(prepared_id, str) or not prepared_id.strip():
            raise ValueError("tool_id must be a non-empty string")
        if input_spec is not None and not isinstance(input_spec, ToolInputSpec):
            raise TypeError("input_spec must be a ToolInputSpec")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        self._id = prepared_id.strip()
        self._name = name.strip()
        self._description = description
        self._input_spec = input_spec or ToolInputSpec()
        self._metadata = deepcopy(dict(metadata or {}))
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        self._enabled = enabled
        if not isinstance(recovery_policy, ToolRecoveryPolicy):
            raise TypeError("recovery_policy must be a ToolRecoveryPolicy")
        self._recovery_policy = recovery_policy

    @property
    def id(self) -> str:
        """Return the stable tool identifier."""
        return self._id

    @property
    def name(self) -> str:
        """Return the display name used for normalized registry lookup."""
        return self._name

    @property
    def description(self) -> str:
        """Return the developer-provided tool description."""
        return self._description

    @property
    def input_spec(self) -> ToolInputSpec:
        """Return the immutable structured input specification."""
        return self._input_spec

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return a read-only, isolated metadata snapshot."""
        return MappingProxyType(deepcopy(self._metadata))

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def recovery_policy(self) -> ToolRecoveryPolicy:
        return self._recovery_policy

    def reconcile(self, operation: Any) -> ToolReconciliation | ToolResult | None:
        """Return a confirmed result or None if external outcome remains unknown."""
        return None

    @abstractmethod
    def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Invoke this tool explicitly and return a normalized result."""
        raise NotImplementedError
