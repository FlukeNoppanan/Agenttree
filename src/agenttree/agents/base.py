"""Shared agent identity and configuration."""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class BaseAgent:
    """Common configuration without execution behavior.

    Fields are fixed after creation so IDs remain stable in registrations.
    Metadata may contain mutable structured values. Capabilities are labels,
    not executable functions or registry entries.
    """

    name: str
    id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
