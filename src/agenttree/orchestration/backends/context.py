"""Backend-neutral phase context extracted from the Step 17 SDK."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

from agenttree.agents import RootAgent
from agenttree.config import AgentTreeConfig
from agenttree.core import (
    BaseFinalReviewer, BaseManagerReviewer, BaseSpecialistExecutor,
    BaseTaskDecomposer, ProviderSpecialistExecutor, WorkflowStateManager,
)
from agenttree.models import (
    ExecutionEvent, ReviewDecision, ReviewResult, Task, WorkflowPhase, WorkflowState,
)
from agenttree.orchestration.engine import OrchestrationEngine
from agenttree.orchestration.models import FinalResult, FinalStatus
from agenttree.registry import CapabilityRegistry
from agenttree.tracing import ExecutionEventType


class OrchestrationContext:
    """One run's phase operations and AgentTree state for backend implementers.

    Backends call the five phase methods in order and stop when result is set.
    This context owns no framework registry mutations or backend-specific types.
    Engine methods remain responsible for business decisions and revision loops.
    Instances are per-run and must not be reused after a terminal result.
    """

    def __init__(
        self, *, task: Task, state: WorkflowState, engine: OrchestrationEngine,
        root_agent: RootAgent, config: AgentTreeConfig, agents: CapabilityRegistry,
        decomposer: BaseTaskDecomposer, executor: BaseSpecialistExecutor,
        manager_reviewer: BaseManagerReviewer, final_reviewer: BaseFinalReviewer,
    ) -> None:
        self.task = task
        self._last_state = state
        self._states = WorkflowStateManager()
        self._engine = engine
        self._root = root_agent
        self._config = config
        self._agents = agents
        self._decomposer = decomposer
        self.executor = executor
        self._manager_reviewer = manager_reviewer
        self._final_reviewer = final_reviewer

    @property
    def state(self) -> WorkflowState:
        """Return an isolated public state snapshot of current phase progress."""
        return replace(self._last_state)

    @property
    def result(self) -> FinalResult | None:
        """Return the terminal result, or None while phase coordination continues."""
        return self._last_state.final_result

    def _enter(self, phase: WorkflowPhase, **values: Any) -> None:
        assert self._last_state is not None
        self._last_state = self._states.transition(self._last_state, phase, **values)

    def _record(self, **values: Any) -> None:
        assert self._last_state is not None
        self._last_state = replace(self._last_state, **values)

    def _early_failure(self, reason: str) -> FinalResult:
        assert self._last_state is not None
        state = self._last_state
        trace = state.trace.clone()
        metadata = {"source": "framework", "review_performed": False, "reason": reason}
        review = ReviewResult(
            decision=ReviewDecision.FAIL, reviewer_id=self._root.id,
            feedback=reason, metadata=deepcopy(metadata),
        )
        trace.append(ExecutionEvent(
            task_id=state.task_id,
            event_type=ExecutionEventType.FINAL_RESULT_CREATED.value,
            message=reason, metadata=deepcopy(metadata),
        ))
        result = FinalResult(
            task_id=state.task_id, root_agent_id=self._root.id, success=False,
            content={"managers": []}, status=FinalStatus.FAILED, manager_results=(),
            final_review=review, final_reviews=(), revision_count=0, revisions=(),
            trace=trace, metadata=metadata,
        )
        self._record(final_result=result, trace=trace)
        self._last_state = self._states.fail(self._last_state, reason=reason, status=result.status)
        return result

    def planning(self) -> None:
        """Run combined triage/planning and preserve the no-manager outcome."""
        working = self.task
        self._enter(WorkflowPhase.TRIAGE)
        plan = self._engine.orchestrate(working)
        self._record(triage_result=plan.triage_result, trace=plan.trace)
        self._enter(WorkflowPhase.PLANNING, result=plan)
        if not plan.selected_manager_ids:
            self._early_failure("No matching manager found")
            return

    def delegation(self) -> None:
        """Delegate using the existing engine and preserve no-assignment failure."""
        working = self.task
        plan = self._last_state.orchestration_plan
        assert plan is not None
        self._enter(WorkflowPhase.DELEGATION)
        delegation = self._engine.delegate(
            plan, working, decomposer=self._decomposer,
            specialist_match_all=self._config.specialist_match_all,
        )
        self._record(delegation_result=delegation, trace=delegation.trace, status=delegation.status)
        if not any(
            assignment.specialist_ids
            for manager in delegation.manager_delegations
            for assignment in manager.assignments
        ):
            self._early_failure("No specialist assignments produced")
            return

    def execution(self) -> None:
        """Resolve bindings and execute existing specialist assignments."""
        working = self.task
        executor = self.executor
        delegation = self._last_state.delegation_result
        assert delegation is not None
        self._enter(WorkflowPhase.EXECUTION)
        if isinstance(executor, ProviderSpecialistExecutor):
            for manager in delegation.manager_delegations:
                for assignment in manager.assignments:
                    for specialist_id in assignment.specialist_ids:
                        executor.resolve_provider(self._agents.get(specialist_id))
        execution = self._engine.execute(delegation, working, executor=executor)
        self._record(execution_result=execution, trace=execution.trace, status=execution.status)

    def manager_review(self) -> None:
        """Call the existing manager review and bounded revision loop."""
        working = self.task
        executor = self.executor
        execution = self._last_state.execution_result
        assert execution is not None
        self._enter(WorkflowPhase.MANAGER_REVIEW)
        reviewed = self._engine.review(
            execution, working, reviewer=self._manager_reviewer, executor=executor,
            max_revisions=self._config.max_manager_revisions,
        )
        self._record(manager_review_result=reviewed, trace=reviewed.trace, status=reviewed.status)

    def final_review(self) -> None:
        """Call existing finalization and retain terminal state semantics."""
        working = self.task
        executor = self.executor
        reviewed = self._last_state.manager_review_result
        assert reviewed is not None
        self._enter(WorkflowPhase.FINAL_REVIEW)
        result = self._engine.finalize(
            reviewed, working, root_agent=self._root,
            final_reviewer=self._final_reviewer,
            manager_reviewer=self._manager_reviewer, executor=executor,
            max_manager_revisions=self._config.max_manager_revisions,
            max_final_revisions=self._config.max_final_revisions,
        )
        self._record(final_result=result, trace=result.trace, status=result.status)
        if result.success:
            self._enter(WorkflowPhase.COMPLETED, result=result)
        else:
            self._last_state = self._states.fail(
                self._last_state, reason=result.final_review.feedback or result.status.value,
                status=result.status,
            )

