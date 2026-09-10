"""Domain-independent subtask and template contracts."""

from dataclasses import dataclass, field
from typing import Any, Iterable
from uuid import uuid4


def _prepare_capabilities(capabilities: Iterable[str]) -> tuple[str, ...]:
    if isinstance(capabilities, str):
        raise TypeError("required_capabilities must be an iterable of strings")
    prepared: list[str] = []
    seen: set[str] = set()
    for capability in capabilities:
        if not isinstance(capability, str):
            raise TypeError("Capability names must be strings")
        display_name = capability.strip()
        if not display_name:
            raise ValueError("Capability names cannot be empty")
        normalized = display_name.casefold()
        if normalized not in seen:
            seen.add(normalized)
            prepared.append(display_name)
    return tuple(prepared)


@dataclass(frozen=True)
class SubtaskTemplate:
    """Reusable static configuration for creating a manager-owned subtask."""

    objective: str
    required_capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        objective = self.objective.strip()
        if not objective:
            raise ValueError("Subtask template objective cannot be empty")
        object.__setattr__(self, "objective", objective)
        object.__setattr__(
            self,
            "required_capabilities",
            _prepare_capabilities(self.required_capabilities),
        )


@dataclass(frozen=True)
class Subtask:
    """A stable unit of manager-decomposed work that has not been executed."""

    parent_task_id: str
    manager_id: str
    objective: str
    id: str = field(default_factory=lambda: str(uuid4()))
    required_capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        objective = self.objective.strip()
        if not objective:
            raise ValueError("Subtask objective cannot be empty")
        object.__setattr__(self, "objective", objective)
        object.__setattr__(
            self,
            "required_capabilities",
            _prepare_capabilities(self.required_capabilities),
        )
