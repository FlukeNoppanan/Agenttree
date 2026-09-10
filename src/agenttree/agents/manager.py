"""Manager configuration and specialist membership."""

from dataclasses import dataclass, field

from agenttree.agents.base import BaseAgent
from agenttree.agents.specialist import SpecialistAgent


@dataclass(frozen=True)
class ManagerAgent(BaseAgent):
    """Manage specialist membership without routing or coordinating tasks."""

    _specialists: dict[str, SpecialistAgent] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

    @property
    def specialists(self) -> tuple[SpecialistAgent, ...]:
        """Return a snapshot of specialists in registration order."""
        return tuple(self._specialists.values())

    def register_specialist(self, specialist: SpecialistAgent) -> None:
        """Register a specialist; reject other types and duplicate IDs.

        Raises TypeError for non-specialists and ValueError for an existing ID.
        """
        if not isinstance(specialist, SpecialistAgent):
            raise TypeError("Only SpecialistAgent instances can be registered")
        if specialist.id in self._specialists:
            raise ValueError(f"Specialist ID already registered: {specialist.id}")
        self._specialists[specialist.id] = specialist

    def remove_specialist(self, specialist_id: str) -> SpecialistAgent:
        """Remove and return a specialist by ID; raise KeyError if absent."""
        return self._specialists.pop(specialist_id)
