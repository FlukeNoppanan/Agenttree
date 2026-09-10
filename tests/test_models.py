"""Behavior checks for the public data contracts."""

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from agenttree.models import (
    AgentResult,
    ExecutionEvent,
    ExecutionTrace,
    ReviewDecision,
    ReviewResult,
    Task,
    TaskContext,
    TaskStatus,
)


def test_task_defaults() -> None:
    before = datetime.now(timezone.utc)
    task = Task(objective="Summarize inputs")
    assert task.status is TaskStatus.PENDING
    assert task.context.data == {}
    assert task.metadata == {}
    assert before <= task.created_at <= datetime.now(timezone.utc)
    assert task.created_at.utcoffset() == timedelta(0)


def test_task_ids_are_unique_and_can_be_supplied() -> None:
    first = Task(objective="First")
    second = Task(objective="Second")
    assert first.id != second.id
    assert UUID(first.id).version == 4
    assert UUID(second.id).version == 4
    assert Task(objective="Imported", id="host-task-1").id == "host-task-1"


def test_structured_context_and_metadata() -> None:
    context = TaskContext(data={"inputs": [{"value": 42}], "options": {"active": True}})
    task = Task(objective="Process inputs", context=context, metadata={"labels": ["sample"]})
    assert task.context.data["inputs"][0]["value"] == 42
    assert task.context.data["options"]["active"] is True
    assert task.metadata["labels"] == ["sample"]


def test_mutable_defaults_are_independent() -> None:
    first, second = Task(objective="First"), Task(objective="Second")
    first.context.data["input"] = 1
    first.metadata["label"] = "first"
    assert second.context.data == {}
    assert second.metadata == {}
    result = AgentResult(agent_id="a", success=True)
    result.metadata["count"] = 1
    assert AgentResult(agent_id="b", success=True).metadata == {}
    review = ReviewResult(decision=ReviewDecision.PASS, reviewer_id="a")
    review.metadata["count"] = 1
    assert ReviewResult(decision=ReviewDecision.PASS, reviewer_id="b").metadata == {}
    event = ExecutionEvent(event_type="sample", task_id="task")
    event.metadata["count"] = 1
    assert ExecutionEvent(event_type="sample", task_id="task").metadata == {}


def test_agent_result_success() -> None:
    result = AgentResult(
        agent_id="worker-1", success=True,
        output={"items": [{"value": 3}], "complete": True},
        metadata={"attempt": 1},
    )
    assert result.success is True
    assert result.output["items"][0]["value"] == 3
    assert result.metadata == {"attempt": 1}
    assert result.error is None


def test_agent_result_failure() -> None:
    result = AgentResult(agent_id="worker-1", success=False, error="Unable to finish")
    assert result.success is False
    assert result.output is None
    assert result.error == "Unable to finish"


def test_review_decision_values() -> None:
    assert {decision.value for decision in ReviewDecision} == {"pass", "revise", "fail"}
    assert ReviewDecision("pass") is ReviewDecision.PASS


def test_review_result() -> None:
    result = ReviewResult(
        decision=ReviewDecision.REVISE, reviewer_id="reviewer-1",
        feedback="Add supporting detail", metadata={"missing": ["evidence"]},
    )
    assert result.decision is ReviewDecision.REVISE
    assert result.reviewer_id == "reviewer-1"
    assert result.feedback == "Add supporting detail"
    assert result.metadata == {"missing": ["evidence"]}


def test_task_status_values() -> None:
    assert {status.value for status in TaskStatus} == {
        "pending", "triage", "planning", "running", "review", "revision", "completed", "failed",
    }


def test_trace_preserves_insertion_order_and_snapshot() -> None:
    trace = ExecutionTrace(task_id="task-1")
    now = datetime.now(timezone.utc)
    first = ExecutionEvent(event_type="first", task_id="task-1", timestamp=now)
    second = ExecutionEvent(
        event_type="second", task_id="task-1", timestamp=now - timedelta(seconds=1),
    )
    trace.append(first)
    snapshot = trace.events
    trace.append(second)
    assert trace.events == (first, second)
    assert snapshot == (first,)
    assert ExecutionTrace(task_id="task-2").events == ()


def test_trace_rejects_event_for_another_task() -> None:
    trace = ExecutionTrace(task_id="task-1")
    with pytest.raises(ValueError, match="task_id"):
        trace.append(ExecutionEvent(event_type="sample", task_id="task-2"))
    assert trace.events == ()


def test_event_default_timestamp_and_optional_actor() -> None:
    before = datetime.now(timezone.utc)
    event = ExecutionEvent(event_type="sample", task_id="task-1")
    assert before <= event.timestamp <= datetime.now(timezone.utc)
    assert event.timestamp.utcoffset() == timedelta(0)
    assert event.actor_id is None
