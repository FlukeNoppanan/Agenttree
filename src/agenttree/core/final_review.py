"""Root-level final review contract and deterministic implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING

from agenttree.agents import RootAgent
from agenttree.models import ReviewDecision, ReviewResult, Task
from agenttree.core.structured_output import (
    REVIEW_PROMPT, _ProviderDecision, review_output, task_context,
)

if TYPE_CHECKING:
    from agenttree.orchestration.models import TaskManagerReviewResult


class BaseFinalReviewer(ABC):
    """Evaluate task-wide manager review results at the Root level."""

    @abstractmethod
    def review(
        self,
        task: Task,
        root_agent: RootAgent,
        manager_review_result: TaskManagerReviewResult,
    ) -> ReviewResult:
        """Return a Root-level PASS, REVISE, or FAIL decision."""
        raise NotImplementedError


class ProviderFinalReviewer(_ProviderDecision, BaseFinalReviewer):
    """Evaluate task-wide manager results through an injected provider."""

    def review(
        self, task: Task, root_agent: RootAgent,
        manager_review_result: TaskManagerReviewResult,
    ) -> ReviewResult:
        """Return a Root decision while leaving revision control to the engine."""
        from agenttree.orchestration.models import TaskManagerReviewResult

        context = task_context(task)
        if not isinstance(root_agent, RootAgent):
            raise TypeError("root_agent must be a RootAgent")
        if not isinstance(manager_review_result, TaskManagerReviewResult):
            raise TypeError("manager_review_result must be a TaskManagerReviewResult")
        if manager_review_result.task_id != task.id:
            raise ValueError("Manager review result must match the Task id")
        def validate(data: dict[str, Any]) -> ReviewResult:
            return review_output(data, root_agent.id)

        return self._generate(
            "final_review", REVIEW_PROMPT,
            "Review overall task completion from the manager results as the Root.",
            {"task": context, "root": {
                "id": root_agent.id,
                "name": root_agent.name, "description": root_agent.description,
                "capabilities": root_agent.capabilities,
            }, "manager_review_result": {
                "status": manager_review_result.status.value,
                "manager_results": [asdict(item) for item in manager_review_result.manager_results],
                "metadata": manager_review_result.metadata,
            }},
            validate=validate,
        )


class StaticFinalReviewer(BaseFinalReviewer):
    """Return deterministic configured final-review outcomes in order.

    A single decision and feedback value form the default outcome. When an
    outcome sequence is supplied, each call advances through it and the final
    configured outcome is repeated after the sequence is exhausted. Sequence
    progress is tracked independently for each task ID.
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
            raise ValueError("At least one final-review outcome is required")
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
        root_agent: RootAgent,
        manager_review_result: TaskManagerReviewResult,
    ) -> ReviewResult:
        """Return the next configured outcome for the whole task."""
        from agenttree.orchestration.models import TaskManagerReviewResult

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if not isinstance(root_agent, RootAgent):
            raise TypeError("root_agent must be a RootAgent")
        if not isinstance(manager_review_result, TaskManagerReviewResult):
            raise TypeError(
                "manager_review_result must be a TaskManagerReviewResult",
            )
        if manager_review_result.task_id != task.id:
            raise ValueError("Manager review result must match the Task id")

        review_index = self._review_counts.get(task.id, 0)
        decision, feedback = self._outcomes[
            min(review_index, len(self._outcomes) - 1)
        ]
        self._review_counts[task.id] = review_index + 1
        return ReviewResult(
            decision=decision,
            reviewer_id=root_agent.id,
            feedback=feedback,
            metadata={"review_number": review_index + 1},
        )
