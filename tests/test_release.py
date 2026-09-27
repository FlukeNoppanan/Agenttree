"""Release-facing smoke checks using supported public imports."""

from importlib.metadata import version
from importlib.resources import files

import pytest

import agenttree
from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    RuleBasedTaskTriage,
    StaticFinalReviewer,
    StaticManagerReviewer,
    StaticTaskDecomposer,
)
from agenttree.models import SubtaskTemplate
from agenttree.orchestration import FinalResult, FinalStatus
from agenttree.providers import MockProvider


def _release_framework() -> tuple[AgentTree, SpecialistAgent, MockProvider]:
    framework = AgentTree(
        root_agent=RootAgent(name="Release Root", id="release-root"),
        triage=RuleBasedTaskTriage(
            {}, fallback_capabilities=("release_analysis",),
        ),
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate("Run offline smoke work", ("release_work",)),
        )),
        manager_reviewer=StaticManagerReviewer(),
        final_reviewer=StaticFinalReviewer(),
    )
    manager = ManagerAgent(
        name="Release Manager",
        id="release-manager",
        capabilities=("release_analysis",),
    )
    specialist = SpecialistAgent(
        name="Release Specialist",
        id="release-specialist",
        capabilities=("release_work",),
    )
    provider = MockProvider(response_content="offline release result")
    framework.register_manager(manager)
    framework.register_specialist(manager, specialist)
    framework.register_provider(provider)
    framework.bind_provider(specialist, provider)
    return framework, specialist, provider


def test_release_smoke_uses_public_sdk_and_retains_trace() -> None:
    framework, _, _ = _release_framework()

    result = framework.run(Task(id="release-task", objective="Run release smoke test"))

    assert isinstance(result, FinalResult)
    assert result.status is FinalStatus.COMPLETED
    assert result.success
    assert result.task_id == "release-task"
    assert result.trace.task_id == result.task_id
    assert result.trace.event_count > 0
    assert framework.last_state is not None
    assert framework.last_state.trace.events == result.trace.events


def test_root_exports_are_curated_and_backward_compatible() -> None:
    assert agenttree.__all__ == [
        "AgentTree",
        "AgentTreeConfig",
        "RootAgent",
        "ManagerAgent",
        "SpecialistAgent",
        "Task",
    ]


def test_distribution_version_and_typed_marker() -> None:
    assert version("agenttree") == "0.2.2"
    assert agenttree.__version__ == version("agenttree")
    assert files("agenttree").joinpath("py.typed").is_file()


def test_binding_error_identifies_unknown_provider_and_fix() -> None:
    framework, specialist, _ = _release_framework()

    with pytest.raises(KeyError, match=r"Provider 'unknown'.*register_provider"):
        framework.bind_provider(specialist, "unknown")
