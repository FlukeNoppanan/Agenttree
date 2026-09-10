"""Contract and deterministic behavior tests for task triage."""

from dataclasses import fields

import pytest

from agenttree.agents import ManagerAgent, SpecialistAgent
from agenttree.core import BaseTaskTriage, RuleBasedTaskTriage
from agenttree.models import Task, TaskStatus, TriageResult
from agenttree.registry import CapabilityRegistry


def test_triage_result_creation_and_metadata_defaults() -> None:
    result = TriageResult(
        task_id="task-1",
        objective="Normalized objective",
        required_capabilities=("summarize", "compare"),
        category="compound",
        confidence=0.75,
        notes="Two rule matches",
        metadata={"matched_rules": 2},
    )
    assert result.task_id == "task-1"
    assert result.required_capabilities == ("summarize", "compare")
    assert result.category == "compound"
    assert result.confidence == 0.75
    assert result.metadata == {"matched_rules": 2}
    assert TriageResult(task_id="task-2", objective="Other").metadata == {}


def test_base_task_triage_is_an_abstract_contract() -> None:
    with pytest.raises(TypeError):
        BaseTaskTriage()  # type: ignore[abstract]

    class IncompleteTriage(BaseTaskTriage):
        pass

    with pytest.raises(TypeError):
        IncompleteTriage()


def test_custom_triage_can_implement_contract() -> None:
    class CustomTriage(BaseTaskTriage):
        def triage(self, task: Task) -> TriageResult:
            return TriageResult(task_id=task.id, objective=task.objective)

    task = Task(objective="Objective")
    assert CustomTriage().triage(task).task_id == task.id


def test_rule_based_triage_returns_input_identity_and_normalized_objective() -> None:
    task = Task(objective="  Create a concise summary  ")
    result = RuleBasedTaskTriage({"summary": ("summarize",)}).triage(task)
    assert result.task_id == task.id
    assert result.objective == "Create a concise summary"
    assert result.required_capabilities == ("summarize",)


def test_keyword_matching_trims_and_uses_casefold() -> None:
    triage = RuleBasedTaskTriage({"  STRASSE  ": ("Transform",)})
    result = triage.triage(Task(objective="Process Straße records"))
    assert result.required_capabilities == ("Transform",)


def test_multiple_rules_preserve_order_and_remove_duplicate_capabilities() -> None:
    triage = RuleBasedTaskTriage(
        {
            "prepare": ("Analyze", "Summarize"),
            "result": (" summarize ", "Format"),
        },
    )
    result = triage.triage(Task(objective="Prepare the result"))
    assert result.required_capabilities == ("Analyze", "Summarize", "Format")


def test_rule_capability_duplicates_are_normalized_during_configuration() -> None:
    triage = RuleBasedTaskTriage(
        {"work": ("  Compose  ", "compose", "COMPOSE", "Validate")},
    )
    assert triage.triage(Task(objective="Work item")).required_capabilities == (
        "Compose", "Validate",
    )


def test_no_match_without_fallback_returns_empty_capabilities() -> None:
    result = RuleBasedTaskTriage({"match": ("one",)}).triage(
        Task(objective="No configured keyword"),
    )
    assert result.required_capabilities == ()


def test_fallback_is_used_only_when_rules_contribute_nothing() -> None:
    triage = RuleBasedTaskTriage(
        {"match": ("Matched",)},
        fallback_capabilities=("General", " general ", "Prepare"),
    )
    assert triage.triage(Task(objective="No rule")).required_capabilities == (
        "General", "Prepare",
    )
    assert triage.triage(Task(objective="A match")).required_capabilities == (
        "Matched",
    )


def test_result_configuration_and_metadata_are_isolated_per_call() -> None:
    triage = RuleBasedTaskTriage(
        {}, category="generic", confidence=1.0, notes="Deterministic",
        metadata={"source": "rules"},
    )
    first = triage.triage(Task(objective="First"))
    second = triage.triage(Task(objective="Second"))
    first.metadata["changed"] = True
    assert second.metadata == {"source": "rules"}
    assert (second.category, second.confidence, second.notes) == (
        "generic", 1.0, "Deterministic",
    )


@pytest.mark.parametrize(
    ("rules", "fallback", "error"),
    [
        ({"": ("capability",)}, (), ValueError),
        ({"keyword": (" ",)}, (), ValueError),
        ({"keyword": "capability"}, (), TypeError),
        ({"keyword": (1,)}, (), TypeError),
        ({}, "fallback", TypeError),
    ],
)
def test_invalid_rule_configuration_is_rejected(
    rules: object, fallback: object, error: type[Exception],
) -> None:
    with pytest.raises(error):
        RuleBasedTaskTriage(  # type: ignore[arg-type]
            rules, fallback_capabilities=fallback,
        )


def test_triage_rejects_non_task_inputs() -> None:
    with pytest.raises(TypeError, match="Task"):
        RuleBasedTaskTriage({}).triage("objective")  # type: ignore[arg-type]


def test_triage_does_not_mutate_task() -> None:
    task = Task(
        objective="  Prepare result  ", status=TaskStatus.PENDING,
        metadata={"source": {"id": 1}},
    )
    original = (
        task.id, task.objective, task.context, task.metadata.copy(),
        task.status, task.created_at,
    )
    RuleBasedTaskTriage({"prepare": ("prepare",)}).triage(task)
    assert (
        task.id, task.objective, task.context, task.metadata,
        task.status, task.created_at,
    ) == original


def test_triage_result_contains_no_selected_agents() -> None:
    result_fields = {item.name for item in fields(TriageResult)}
    assert "agent" not in result_fields
    assert "manager" not in result_fields
    assert "specialist" not in result_fields


def test_triage_does_not_modify_capability_registry() -> None:
    registry = CapabilityRegistry()
    agent = SpecialistAgent(name="Worker", capabilities=("summarize",))
    registry.register(agent)
    before = (registry.agents, registry.capabilities, registry.capability_index)
    RuleBasedTaskTriage({"summary": ("summarize",)}).triage(
        Task(objective="Create a summary"),
    )
    assert (registry.agents, registry.capabilities, registry.capability_index) == before


def test_triage_capabilities_are_compatible_with_registry_lookup() -> None:
    registry = CapabilityRegistry()
    specialist = SpecialistAgent(name="Worker", capabilities=("summarize",))
    manager = ManagerAgent(name="Manager", capabilities=("coordinate",))
    registry.register(specialist)
    registry.register(manager)
    result = RuleBasedTaskTriage(
        {"summary": ("  SUMMARIZE  ",)},
    ).triage(Task(objective="Create a summary"))
    assert result.required_capabilities == ("SUMMARIZE",)
    assert registry.find_by_capabilities(result.required_capabilities) == (specialist,)
