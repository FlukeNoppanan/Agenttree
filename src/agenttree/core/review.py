"""Manager review contract and deterministic static implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from agenttree.agents import ManagerAgent
from agenttree.models import ReviewDecision, ReviewResult, Subtask, Task
from agenttree.core.structured_output import (
    REVIEW_PROMPT, _ProviderDecision, review_output, task_context,
)

if TYPE_CHECKING:
    from agenttree.orchestration.models import SpecialistExecution


class BaseManagerReviewer(ABC):
    """Review all specialist executions associated with one subtask."""

    @abstractmethod
    def review(
        self,
        task: Task,
        subtask: Subtask,
        manager: ManagerAgent,
        specialist_executions: tuple[SpecialistExecution, ...],
    ) -> ReviewResult:
        """Return a manager-level PASS, REVISE, or FAIL decision."""
        raise NotImplementedError


class ProviderManagerReviewer(_ProviderDecision, BaseManagerReviewer):
    """Decide manager acceptance through a provider without executing work."""

    def review(
        self, task: Task, subtask: Subtask, manager: ManagerAgent,
        specialist_executions: tuple[SpecialistExecution, ...],
    ) -> ReviewResult:
        """Review all supplied executions, including failures and their errors."""
        from agenttree.orchestration.models import SpecialistExecution

        context = task_context(task)
        if not isinstance(subtask, Subtask):
            raise TypeError("subtask must be a Subtask")
        if not isinstance(manager, ManagerAgent):
            raise TypeError("manager must be a ManagerAgent")
        if not isinstance(specialist_executions, tuple) or any(
            not isinstance(item, SpecialistExecution) for item in specialist_executions
        ):
            raise TypeError("specialist_executions must be a tuple of SpecialistExecution")
        if subtask.parent_task_id != task.id or subtask.manager_id != manager.id:
            raise ValueError("Task, Subtask, and Manager identities must match")
        if any(item.subtask_id != subtask.id for item in specialist_executions):
            raise ValueError("Specialist executions must belong to the Subtask")
        def validate(data: dict[str, Any]) -> ReviewResult:
            return review_output(data, manager.id)

        return self._generate(
            "manager_review", REVIEW_PROMPT,
            "Review the specialist results for this subtask as its manager.",
            {"task": context, "subtask": asdict(subtask), "manager": {
                "id": manager.id,
                "name": manager.name, "description": manager.description,
                "capabilities": manager.capabilities,
            }, "specialist_executions": [asdict(item) for item in specialist_executions]},
            validate=validate,
        )


class StaticManagerReviewer(BaseManagerReviewer):
    """Return deterministic configured outcomes independently per subtask.

    A single ``decision`` and ``feedback`` form the default outcome. Optional
    ``outcomes`` provide an ordered sequence of decision/feedback pairs. After
    a sequence is exhausted, its final outcome is repeated.
    """

    def __init__(
        self,
        decision: ReviewDecision = ReviewDecision.PASS,
        feedback: str = "",
        *,
        outcomes: Iterable[tuple[ReviewDecision, str]] | None = None,
    ) -> None:
        prepared = (
            tuple(outcomes)
            if outcomes is not None
            else ((decision, feedback),)
        )
        if not prepared:
            raise ValueError("At least one review outcome is required")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], ReviewDecision)
            or not isinstance(item[1], str)
            for item in prepared
        ):
            raise TypeError(
                "outcomes must contain (ReviewDecision, feedback) tuples",
            )
        self._outcomes = prepared
        self._review_counts: dict[str, int] = {}

    def review(
        self,
        task: Task,
        subtask: Subtask,
        manager: ManagerAgent,
        specialist_executions: tuple[SpecialistExecution, ...],
    ) -> ReviewResult:
        """Return the next configured outcome for this subtask."""
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if not isinstance(subtask, Subtask):
            raise TypeError("subtask must be a Subtask")
        if not isinstance(manager, ManagerAgent):
            raise TypeError("manager must be a ManagerAgent")
        if not isinstance(specialist_executions, tuple):
            raise TypeError("specialist_executions must be a tuple")
        if subtask.parent_task_id != task.id or subtask.manager_id != manager.id:
            raise ValueError("Task, Subtask, and Manager identities must match")

        review_index = self._review_counts.get(subtask.id, 0)
        outcome = self._outcomes[min(review_index, len(self._outcomes) - 1)]
        self._review_counts[subtask.id] = review_index + 1
        return ReviewResult(
            decision=outcome[0],
            feedback=outcome[1],
            reviewer_id=manager.id,
            metadata={"review_number": review_index + 1},
        )
