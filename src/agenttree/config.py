"""Configuration for the synchronous developer-facing framework."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTreeConfig:
    """Immutable controls for existing matching and bounded revision behavior."""

    manager_match_all: bool = False
    specialist_match_all: bool = False
    max_manager_revisions: int = 2
    max_final_revisions: int = 1

    def __post_init__(self) -> None:
        for name in ("manager_match_all", "specialist_match_all"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in ("max_manager_revisions", "max_final_revisions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
