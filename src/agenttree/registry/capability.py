"""Capability-based agent registration and discovery."""

from types import MappingProxyType
from typing import Iterable, Mapping, TypeAlias

from agenttree.agents import BaseAgent

AgentClassFilter: TypeAlias = type[BaseAgent] | tuple[type[BaseAgent], ...]


def _normalize_capability(capability: str) -> str:
    """Normalize a capability using surrounding-whitespace trim and casefold."""
    if not isinstance(capability, str):
        raise TypeError("Capabilities must be strings")
    normalized = capability.strip().casefold()
    if not normalized:
        raise ValueError("Capability names cannot be empty")
    return normalized


class CapabilityRegistry:
    """Store agents and discover them by normalized capability labels.

    Registration order determines the order of agent snapshots and search
    results. Capability matching trims surrounding whitespace and uses
    ``str.casefold`` for case-insensitive comparison. No scoring or routing is
    performed.
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._capability_index: dict[str, dict[str, BaseAgent]] = {}
        self._agent_capabilities: dict[str, tuple[str, ...]] = {}

    @property
    def agents(self) -> tuple[BaseAgent, ...]:
        """Return registered agents in registration order as a snapshot."""
        return tuple(self._agents.values())

    @property
    def capabilities(self) -> frozenset[str]:
        """Return the known normalized capability names."""
        return frozenset(self._capability_index)

    @property
    def capability_index(self) -> Mapping[str, tuple[BaseAgent, ...]]:
        """Return a read-only snapshot of normalized capability associations."""
        snapshot = {
            capability: tuple(agents.values())
            for capability, agents in self._capability_index.items()
        }
        return MappingProxyType(snapshot)

    def register(self, agent: BaseAgent) -> None:
        """Register an agent and index its capabilities.

        Raises ``TypeError`` for non-agents and ``ValueError`` for a duplicate
        agent ID or an invalid capability. Registration is atomic if capability
        validation fails.
        """
        if not isinstance(agent, BaseAgent):
            raise TypeError("Only BaseAgent instances can be registered")
        if agent.id in self._agents:
            raise ValueError(f"Agent ID already registered: {agent.id}")

        normalized_capabilities = tuple(
            dict.fromkeys(_normalize_capability(item) for item in agent.capabilities)
        )
        self._agents[agent.id] = agent
        self._agent_capabilities[agent.id] = normalized_capabilities
        for capability in normalized_capabilities:
            self._capability_index.setdefault(capability, {})[agent.id] = agent

    def unregister(self, agent_id: str) -> BaseAgent:
        """Remove and return an agent by ID, including all index references."""
        agent = self._agents.pop(agent_id)
        for capability in self._agent_capabilities.pop(agent_id):
            bucket = self._capability_index[capability]
            del bucket[agent_id]
            if not bucket:
                del self._capability_index[capability]
        return agent

    def get(self, agent_id: str) -> BaseAgent:
        """Return an agent by ID, raising ``KeyError`` when it is absent."""
        return self._agents[agent_id]

    def find_by_capability(
        self,
        capability: str,
        agent_type: AgentClassFilter | None = None,
    ) -> tuple[BaseAgent, ...]:
        """Find agents advertising one capability in registration order."""
        return self.find_by_capabilities(
            (capability,), match_all=True, agent_type=agent_type,
        )

    def find_by_capabilities(
        self,
        capabilities: Iterable[str],
        match_all: bool = False,
        agent_type: AgentClassFilter | None = None,
    ) -> tuple[BaseAgent, ...]:
        """Find agents matching any or all requested capabilities.

        An empty request returns an empty tuple for both matching modes.
        ``agent_type`` uses normal ``isinstance`` semantics, so subclasses are
        included and a tuple of classes may be supplied.
        """
        if isinstance(capabilities, str):
            raise TypeError("capabilities must be an iterable of strings")
        requested = frozenset(_normalize_capability(item) for item in capabilities)
        if not requested:
            return ()
        if agent_type is not None:
            self._validate_agent_type(agent_type)

        matches: list[BaseAgent] = []
        for agent_id, agent in self._agents.items():
            declared = set(self._agent_capabilities[agent_id])
            capability_match = (
                requested.issubset(declared) if match_all else bool(requested & declared)
            )
            if capability_match and (
                agent_type is None or isinstance(agent, agent_type)
            ):
                matches.append(agent)
        return tuple(matches)

    @staticmethod
    def _validate_agent_type(agent_type: AgentClassFilter) -> None:
        classes = agent_type if isinstance(agent_type, tuple) else (agent_type,)
        if not classes or any(
            not isinstance(item, type) or not issubclass(item, BaseAgent)
            for item in classes
        ):
            raise TypeError("agent_type must contain BaseAgent classes")
