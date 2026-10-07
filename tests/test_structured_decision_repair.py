"""One-shot structured-decision repair, strict validation, and safe events."""

from contextlib import contextmanager
import json

import pytest

from agenttree import AgentTreeConfig, ManagerAgent, RootAgent, Task
from agenttree.core import (
    ProviderFinalReviewer, ProviderManagerReviewer, ProviderTaskDecomposer,
    ProviderTaskTriage,
)
from agenttree.core.execution_store import (
    ExecutionRecord, ExecutionState, InMemoryExecutionStore, utc_now,
)
from agenttree.core.operation_journal import OperationJournal, _active_operation_journal
from agenttree.exceptions import DecisionOutputError, DecisionParseError
from agenttree.models import ExecutionTrace, ReviewDecision, Subtask, TriageResult
from agenttree.orchestration import ManagerReviewStatus, TaskManagerReviewResult
from agenttree.providers import (
    BaseProvider, ProviderCapabilities, ProviderConfig, ProviderRequest,
    ProviderResponse, ProviderRuntimeError, ProviderUsage,
)


class SequenceProvider(BaseProvider):
    def __init__(self, responses):
        super().__init__(ProviderConfig("sequence", model="fixture"))
        self.responses = list(responses)
        self.requests: list[ProviderRequest] = []

    @property
    def capabilities(self):
        return ProviderCapabilities()

    def generate(self, request):
        self.requests.append(request)
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ProviderResponse(content=outcome, provider=self.name, model="fixture",
                                usage=ProviderUsage(1, 1, 2))


@contextmanager
def journal_context():
    store = InMemoryExecutionStore()
    now = utc_now()
    store.create(ExecutionRecord("decision-run", ExecutionState.QUEUED,
                                 "a" * 64, "{}", now, now))
    journal = OperationJournal(store, "decision-run")
    journal.start_phase("planning")
    token = _active_operation_journal.set(journal)
    try:
        yield store
    finally:
        _active_operation_journal.reset(token)


def manager_review(provider):
    task = Task(objective="Review this work")
    manager = ManagerAgent(name="Review Manager")
    subtask = Subtask(parent_task_id=task.id, manager_id=manager.id,
                      objective="Review the result")
    return ProviderManagerReviewer(provider).review(task, subtask, manager, ())


def final_review(provider):
    task = Task(objective="Review the full result")
    root = RootAgent(name="Root")
    reviewed = TaskManagerReviewResult(
        task_id=task.id, manager_results=(), status=ManagerReviewStatus.PASSED,
        trace=ExecutionTrace(task_id=task.id),
    )
    return ProviderFinalReviewer(provider).review(task, root, reviewed)


def structured_events(store):
    return [item.event for item in store.read_events("decision-run", after=0, limit=1000)
            if item.event.event_type.startswith("structured_decision.")]


def test_malformed_review_is_repaired_once_and_emits_safe_events():
    provider = SequenceProvider(["not JSON", '{"decision":"pass","feedback":"accepted"}'])
    with journal_context() as store:
        result = manager_review(provider)
    assert result.decision is ReviewDecision.PASS
    assert len(provider.requests) == 2
    assert provider.requests[1].context["structured_decision_repair"] == {
        "attempt": 1, "rejected_response_class": "malformed_syntax", "reason_code": "malformed_json",
    }
    events = structured_events(store)
    assert [e.event_type for e in events] == [
        "structured_decision.decision_attempt",
        "structured_decision.validation_failed",
        "structured_decision.repair.started",
        "structured_decision.decision_attempt",
        "structured_decision.normalized",
        "structured_decision.repair.succeeded",
    ]
    assert events[1].metadata["failure_class"] == "malformed_syntax"
    assert events[-1].metadata["repair_attempt"] == 1
    assert all("not JSON" not in json.dumps(event.metadata) for event in events)


def test_schema_invalid_review_is_repaired_then_strictly_validated():
    provider = SequenceProvider([
        '{"decision":"approve","feedback":"accepted"}',
        '{"decision":"revise","feedback":"add a source"}',
    ])
    with journal_context() as store:
        result = manager_review(provider)
    assert result.decision is ReviewDecision.REVISE
    events = structured_events(store)
    assert [e.event_type for e in events if e.event_type.endswith(("failed", "started", "succeeded"))] == [
        "structured_decision.validation_failed",
        "structured_decision.repair.started",
        "structured_decision.repair.succeeded",
    ]
    assert events[1].metadata["failure_class"] == "schema_invalid"


def test_semantically_invalid_capability_reference_is_repaired():
    provider = SequenceProvider([
        json.dumps({"objective": "Research", "required_capabilities": ["unregistered"]}),
        json.dumps({"objective": "Research", "required_capabilities": ["research"]}),
    ])
    with journal_context() as store:
        result = ProviderTaskTriage(provider).triage(
            Task(objective="Research the question"),
            available_capabilities=("research",),
        )
    assert result.required_capabilities == ("research",)
    events = structured_events(store)
    rejected = next(e for e in events if e.event_type == "structured_decision.validation_failed")
    assert rejected.metadata["failure_class"] == "invalid_reference"
    assert "unregistered" not in json.dumps(rejected.metadata)


def test_root_planning_schema_invalid_response_is_repaired():
    from agenttree.core.root import ProviderRootPlanner

    provider = SequenceProvider([
        '{"delegate":"yes"}',
        '{"delegate":true}',
    ])
    with journal_context() as store:
        result = ProviderRootPlanner(provider).plan(Task(objective="Delegate this work"), RootAgent(name="Root"))
    assert result.delegate is True
    assert len(provider.requests) == 2
    assert any(e.event_type == "structured_decision.repair.succeeded" for e in structured_events(store))


def test_manager_decomposition_semantic_reference_is_repaired():
    provider = SequenceProvider([
        json.dumps({"subtasks": [{"objective": "Research", "required_capabilities": ["missing"]}]}),
        json.dumps({"subtasks": [{"objective": "Research", "required_capabilities": ["research"]}]}),
    ])
    task = Task(objective="Research")
    manager = ManagerAgent(name="Manager")
    triage = TriageResult(task.id, task.objective)
    with journal_context() as store:
        result = ProviderTaskDecomposer(provider).decompose(
            task, manager, triage, available_capabilities=("research",),
        )
    assert len(result) == 1 and result[0].required_capabilities == ("research",)
    rejected = next(e for e in structured_events(store)
                    if e.event_type == "structured_decision.validation_failed")
    assert rejected.metadata["failure_class"] == "invalid_reference"


def test_root_final_review_malformed_output_is_repaired():
    provider = SequenceProvider(["`bad json`", '{"decision":"pass","feedback":"complete"}'])
    with journal_context() as store:
        result = final_review(provider)
    assert result.decision is ReviewDecision.PASS
    assert len(provider.requests) == 2
    assert any(e.event_type == "structured_decision.repair.succeeded" for e in structured_events(store))


def test_repair_failure_is_bounded_and_never_accepts_invalid_decision():
    provider = SequenceProvider(["{", "still not JSON"])
    with journal_context() as store:
        with pytest.raises(DecisionParseError):
            manager_review(provider)
    assert len(provider.requests) == 2
    events = structured_events(store)
    assert sum(e.event_type == "structured_decision.decision_attempt" for e in events) == 2
    assert sum(e.event_type == "structured_decision.repair.started" for e in events) == 1
    assert sum(e.event_type == "structured_decision.repair.failed" for e in events) == 1
    assert not any(e.event_type == "structured_decision.repair.succeeded" for e in events)


def test_provider_transport_failure_is_not_retried_as_decision_repair():
    provider = SequenceProvider([ProviderRuntimeError("transport failed")])
    with journal_context() as store:
        with pytest.raises(ProviderRuntimeError, match="transport failed"):
            manager_review(provider)
    assert len(provider.requests) == 1
    events = structured_events(store)
    assert sum(e.event_type == "structured_decision.decision_attempt" for e in events) == 1
    assert not any("repair" in e.event_type for e in events)


def test_provider_failure_during_the_single_repair_is_observable_and_propagates():
    provider = SequenceProvider(["invalid", ProviderRuntimeError("provider unavailable")])
    with journal_context() as store:
        with pytest.raises(ProviderRuntimeError, match="provider unavailable"):
            manager_review(provider)
    events = structured_events(store)
    failure = next(e for e in events if e.event_type == "structured_decision.repair.failed")
    assert failure.metadata["failure_class"] == "provider_failure"
    assert failure.metadata["provider_error_type"] == "ProviderRuntimeError"
    assert "provider unavailable" not in json.dumps(failure.metadata)


def test_configuration_repair_budget_is_not_derived_from_revision_or_tool_limits():
    assert AgentTreeConfig(max_manager_revisions=20, max_final_revisions=10,
                           max_tool_rounds=3).max_tool_rounds == 3
    provider = SequenceProvider(["bad", '{"decision":"pass","feedback":"ok"}'])
    with journal_context():
        assert manager_review(provider).decision is ReviewDecision.PASS
    assert len(provider.requests) == 2
