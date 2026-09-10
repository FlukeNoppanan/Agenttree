"""Registration and discovery tests for the capability registry."""

from types import MappingProxyType

import pytest

from agenttree.agents import BaseAgent, ManagerAgent, RootAgent, SpecialistAgent
from agenttree.registry import CapabilityRegistry


def test_registers_all_framework_agent_roles() -> None:
    registry = CapabilityRegistry()
    agents = (
        RootAgent(name="Root"),
        ManagerAgent(name="Manager"),
        SpecialistAgent(name="Specialist"),
    )
    for agent in agents:
        registry.register(agent)
    assert registry.agents == agents


def test_rejects_duplicate_ids_without_changing_registry() -> None:
    registry = CapabilityRegistry()
    original = RootAgent(name="Original", id="shared", capabilities=("first",))
    registry.register(original)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ManagerAgent(name="Duplicate", id="shared", capabilities=("second",)))
    assert registry.agents == (original,)
    assert registry.capabilities == frozenset({"first"})


@pytest.mark.parametrize("invalid", [object(), "agent", None])
def test_rejects_non_agents(invalid: object) -> None:
    registry = CapabilityRegistry()
    with pytest.raises(TypeError, match="BaseAgent"):
        registry.register(invalid)  # type: ignore[arg-type]
    assert registry.agents == ()


def test_get_and_unregister_by_id() -> None:
    registry = CapabilityRegistry()
    agent = SpecialistAgent(name="Worker")
    registry.register(agent)
    assert registry.get(agent.id) is agent
    assert registry.unregister(agent.id) is agent
    assert registry.agents == ()
    with pytest.raises(KeyError):
        registry.get(agent.id)
    with pytest.raises(KeyError):
        registry.unregister(agent.id)


def test_finds_one_or_multiple_agents_by_capability() -> None:
    registry = CapabilityRegistry()
    first = SpecialistAgent(name="First", capabilities=("summarize",))
    second = ManagerAgent(name="Second", capabilities=("summarize", "review"))
    third = RootAgent(name="Third", capabilities=("plan",))
    for agent in (first, second, third):
        registry.register(agent)
    assert registry.find_by_capability("summarize") == (first, second)
    assert registry.find_by_capability("missing") == ()


def test_normalizes_capabilities_with_trim_and_casefold() -> None:
    registry = CapabilityRegistry()
    agent = SpecialistAgent(
        name="Worker", capabilities=("  Summarize  ", "STRASSE", "strasse"),
    )
    registry.register(agent)
    assert registry.capabilities == frozenset({"summarize", "strasse"})
    assert registry.find_by_capability(" SUMMARIZE ") == (agent,)
    assert registry.find_by_capability("Strasse") == (agent,)
    assert registry.capability_index["strasse"] == (agent,)


@pytest.mark.parametrize("invalid", ["", "   "])
def test_rejects_empty_capability_atomically(invalid: str) -> None:
    registry = CapabilityRegistry()
    agent = SpecialistAgent(name="Worker", capabilities=("valid", invalid))
    with pytest.raises(ValueError, match="empty"):
        registry.register(agent)
    assert registry.agents == ()
    assert registry.capabilities == frozenset()


def test_any_and_all_matching() -> None:
    registry = CapabilityRegistry()
    first = SpecialistAgent(name="First", capabilities=("alpha",))
    both = SpecialistAgent(name="Both", capabilities=("alpha", "beta"))
    second = SpecialistAgent(name="Second", capabilities=("beta",))
    for agent in (first, both, second):
        registry.register(agent)
    assert registry.find_by_capabilities(("alpha", "beta")) == (first, both, second)
    assert registry.find_by_capabilities(("alpha", "beta"), match_all=True) == (both,)
    assert registry.find_by_capabilities(()) == ()


def test_filters_by_agent_class_and_preserves_order() -> None:
    registry = CapabilityRegistry()
    specialist = SpecialistAgent(name="Specialist", capabilities=("shared",))
    root = RootAgent(name="Root", capabilities=("shared",))
    manager = ManagerAgent(name="Manager", capabilities=("shared",))
    for agent in (specialist, root, manager):
        registry.register(agent)
    assert registry.find_by_capability("shared", ManagerAgent) == (manager,)
    assert registry.find_by_capability("shared", SpecialistAgent) == (specialist,)
    assert registry.find_by_capability("shared", (RootAgent, ManagerAgent)) == (root, manager)


def test_filter_includes_subclasses_and_rejects_invalid_filters() -> None:
    class CustomSpecialist(SpecialistAgent):
        """Application-owned specialist type."""

    registry = CapabilityRegistry()
    agent = CustomSpecialist(name="Custom", capabilities=("shared",))
    registry.register(agent)
    assert registry.find_by_capability("shared", SpecialistAgent) == (agent,)
    for invalid in (str, (), (SpecialistAgent, str)):
        with pytest.raises(TypeError, match="agent_type"):
            registry.find_by_capability("shared", invalid)  # type: ignore[arg-type]


def test_unregister_cleans_capability_index() -> None:
    registry = CapabilityRegistry()
    first = SpecialistAgent(name="First", capabilities=("shared", "only-first"))
    second = SpecialistAgent(name="Second", capabilities=("shared",))
    registry.register(first)
    registry.register(second)
    registry.unregister(first.id)
    assert registry.capabilities == frozenset({"shared"})
    assert registry.find_by_capability("shared") == (second,)
    assert "only-first" not in registry.capability_index
    registry.unregister(second.id)
    assert registry.capability_index == {}


def test_snapshots_do_not_expose_internal_collections() -> None:
    registry = CapabilityRegistry()
    first = BaseAgent(name="First", capabilities=("shared",))
    registry.register(first)
    agents_snapshot = registry.agents
    capabilities_snapshot = registry.capabilities
    index_snapshot = registry.capability_index
    assert isinstance(index_snapshot, MappingProxyType)
    with pytest.raises(TypeError):
        index_snapshot["new"] = ()  # type: ignore[index]

    second = BaseAgent(name="Second", capabilities=("shared", "new"))
    registry.register(second)
    assert agents_snapshot == (first,)
    assert capabilities_snapshot == frozenset({"shared"})
    assert index_snapshot == {"shared": (first,)}
    assert registry.agents == (first, second)
    assert registry.capability_index["shared"] == (first, second)


def test_cleanup_uses_registration_snapshot_if_capabilities_are_mutable() -> None:
    capabilities = ["first"]
    registry = CapabilityRegistry()
    agent = SpecialistAgent(name="Worker", capabilities=capabilities)  # type: ignore[arg-type]
    registry.register(agent)
    capabilities[:] = ["second"]
    registry.unregister(agent.id)
    assert registry.capability_index == {}


def test_capability_search_rejects_plain_string_collection() -> None:
    registry = CapabilityRegistry()
    with pytest.raises(TypeError, match="iterable"):
        registry.find_by_capabilities("one")
