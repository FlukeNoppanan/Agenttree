"""Identity and membership behavior for agent abstractions."""

from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from agenttree.agents import BaseAgent, ManagerAgent, RootAgent, SpecialistAgent


@pytest.mark.parametrize("agent_class", [BaseAgent, RootAgent, ManagerAgent, SpecialistAgent])
def test_agent_configuration(agent_class: type[BaseAgent]) -> None:
    agent = agent_class(
        name="Example", id="agent-1", description="Example configuration",
        capabilities=("summarize", "compare"),
        metadata={"options": {"labels": ["sample"], "enabled": True}},
    )
    assert isinstance(agent, BaseAgent)
    assert agent.id == "agent-1"
    assert agent.name == "Example"
    assert agent.description == "Example configuration"
    assert agent.capabilities == ("summarize", "compare")
    assert agent.metadata == {"options": {"labels": ["sample"], "enabled": True}}


def test_default_ids_are_unique() -> None:
    agents = [BaseAgent(name="Same"), RootAgent(name="Same"),
              ManagerAgent(name="Same"), SpecialistAgent(name="Same")]
    assert len({agent.id for agent in agents}) == len(agents)
    assert all(UUID(agent.id).version == 4 for agent in agents)


def test_defaults_are_independent() -> None:
    first = ManagerAgent(name="First")
    second = ManagerAgent(name="Second")
    first.metadata["labels"] = ["sample"]
    first.register_specialist(SpecialistAgent(name="Worker"))
    assert second.metadata == {}
    assert second.specialists == ()
    assert second.capabilities == ()
    assert second.description == ""


def test_agent_id_cannot_change_after_registration() -> None:
    manager = ManagerAgent(name="Manager")
    specialist = SpecialistAgent(name="Worker")
    manager.register_specialist(specialist)
    with pytest.raises(FrozenInstanceError):
        setattr(specialist, "id", "replacement")
    assert manager.remove_specialist(specialist.id) is specialist


def test_register_specialist() -> None:
    manager = ManagerAgent(name="Manager")
    specialist = SpecialistAgent(name="Worker")
    manager.register_specialist(specialist)
    assert manager.specialists == (specialist,)


def test_duplicate_registration_by_id_is_rejected() -> None:
    manager = ManagerAgent(name="Manager")
    specialist = SpecialistAgent(name="First", id="shared")
    manager.register_specialist(specialist)
    for duplicate in (specialist, SpecialistAgent(name="Second", id="shared")):
        with pytest.raises(ValueError, match="already registered"):
            manager.register_specialist(duplicate)
    assert manager.specialists == (specialist,)


def test_remove_specialist() -> None:
    manager = ManagerAgent(name="Manager")
    specialist = SpecialistAgent(name="Worker")
    manager.register_specialist(specialist)
    assert manager.remove_specialist(specialist.id) is specialist
    assert manager.specialists == ()
    with pytest.raises(KeyError):
        manager.remove_specialist(specialist.id)


def test_registration_order_and_snapshots() -> None:
    manager = ManagerAgent(name="Manager")
    first = SpecialistAgent(name="First", id="z")
    second = SpecialistAgent(name="Second", id="a")
    manager.register_specialist(first)
    snapshot = manager.specialists
    manager.register_specialist(second)
    assert manager.specialists == (first, second)
    assert snapshot == (first,)
    manager.remove_specialist(first.id)
    manager.register_specialist(first)
    assert manager.specialists == (second, first)


@pytest.mark.parametrize(
    "invalid", [BaseAgent(name="Base"), RootAgent(name="Root"),
                ManagerAgent(name="Other"), "worker", None],
)
def test_manager_rejects_non_specialists(invalid: object) -> None:
    manager = ManagerAgent(name="Manager")
    with pytest.raises(TypeError, match="SpecialistAgent"):
        manager.register_specialist(invalid)  # type: ignore[arg-type]
    assert manager.specialists == ()


def test_specialist_subclasses_can_be_registered() -> None:
    class CustomSpecialist(SpecialistAgent):
        """Application-defined specialist configuration."""

    manager = ManagerAgent(name="Manager")
    specialist = CustomSpecialist(name="Custom")
    manager.register_specialist(specialist)
    assert manager.specialists == (specialist,)
