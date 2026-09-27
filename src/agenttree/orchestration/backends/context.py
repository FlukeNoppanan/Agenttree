"""Backend-neutral phase context extracted from the Step 17 SDK."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable

from agenttree.agents import RootAgent
from agenttree.config import AgentTreeConfig
from agenttree.core import (
    BaseFinalReviewer, BaseManagerReviewer, BaseSpecialistExecutor,
    BaseTaskDecomposer, ProviderSpecialistExecutor, WorkflowStateManager,
    BaseRootPlanner, BaseRootSynthesizer,
)
from agenttree.models import (
    ExecutionEvent, ReviewDecision, ReviewResult, Task, WorkflowPhase, WorkflowState,
)
from agenttree.orchestration.engine import OrchestrationEngine
from agenttree.orchestration.models import FinalResult, FinalStatus
from agenttree.core.usage import UsageCollector
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
        root_planner: BaseRootPlanner | None = None,
        root_synthesizer: BaseRootSynthesizer | None = None,
        usage: UsageCollector | None = None,
        on_phase_completed: Callable[[WorkflowState], None] | None = None,
        on_phase_started: Callable[[str], None] | None = None,
        resume_phase: WorkflowPhase | None = None,
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
        self._root_planner = root_planner
        self._root_synthesizer = root_synthesizer
        self._usage = usage if usage is not None else UsageCollector()
        self._on_phase_completed = on_phase_completed
        self._on_phase_started = on_phase_started
        self._resume_phase = resume_phase

    def run_phase(self, name: str) -> None:
        """Run one existing phase, or skip a durably completed phase on recovery."""
        from agenttree.core.execution_control import check_execution
        order = ("planning", "delegation", "execution", "manager_review", "final_review")
        completed = {
            WorkflowPhase.PLANNING: 0, WorkflowPhase.DELEGATION: 1,
            WorkflowPhase.EXECUTION: 2, WorkflowPhase.MANAGER_REVIEW: 3,
            WorkflowPhase.FINAL_REVIEW: 4,
        }
        if name not in order:
            raise ValueError("Unknown orchestration phase")
        if self._resume_phase in completed and order.index(name) <= completed[self._resume_phase]:
            return
        check_execution()
        if self._on_phase_started is not None:
            self._on_phase_started(name)
        getattr(self, name)()
        from agenttree.core.artifacts import current_artifact_session
        artifact_session = current_artifact_session()
        if artifact_session is not None:
            artifact_session.refs()
            fresh = artifact_session.unrecorded_events()
            if fresh:
                trace = self._last_state.trace.clone()
                for event in fresh:
                    trace.append(event)
                self._record(trace=trace)
                if self._last_state.final_result is not None:
                    self._record(final_result=replace(self._last_state.final_result, trace=trace))
            if self._last_state.final_result is not None:
                self._record(final_result=replace(self._last_state.final_result,
                                                  artifacts=artifact_session.final_refs(
                                                      self._last_state.final_result.manager_results,
                                                      success=self._last_state.final_result.success)))
        check_execution()
        if self._on_phase_completed is not None:
            self._on_phase_completed(self.state)

    def _event(self, event_type: ExecutionEventType, trace=None) -> None:
        target = trace if trace is not None else self._last_state.trace.clone()
        target.append(ExecutionEvent(task_id=self.task.id, event_type=event_type.value,
                                     actor_id=self._root.id))
        if trace is None:
            self._record(trace=target)

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
        self._event(ExecutionEventType.RUN_FAILED, trace)
        result = FinalResult(
            task_id=state.task_id, root_agent_id=self._root.id, success=False,
            content={"managers": []}, status=FinalStatus.FAILED, manager_results=(),
            final_review=review, final_reviews=(), revision_count=0, revisions=(),
            trace=trace, metadata=metadata, usage=self._usage.summary(),
            error={"type": "NoWork", "message": reason},
        )
        self._record(final_result=result, trace=trace)
        self._last_state = self._states.fail(self._last_state, reason=reason, status=result.status)
        return result

    def planning(self) -> None:
        """Run combined triage/planning and preserve the no-manager outcome."""
        working = self.task
        self._enter(WorkflowPhase.TRIAGE)
        if self._root_planner is not None:
            self._event(ExecutionEventType.RUN_STARTED)
            self._event(ExecutionEventType.ROOT_PLANNING_STARTED)
            from agenttree.core.operation_journal import current_journal
            journal = current_journal()
            plan_choice = (journal.run(journal.child_key("root:plan"), "root.plan",
                                       (working.id, working.objective, working.context.data),
                                       lambda: self._root_planner.plan(working, self._root),
                                       agent_id=self._root.id)
                           if journal is not None else self._root_planner.plan(working, self._root))
            from agenttree.core.root import RootPlan
            if not isinstance(plan_choice, RootPlan):
                raise TypeError("Root planner must return RootPlan")
            self._event(ExecutionEventType.ROOT_PLANNING_COMPLETED)
            if not plan_choice.delegate:
                if journal is not None:
                    journal.run(journal.child_key("root:direct_response"), "root.direct_response",
                                (working.id, plan_choice.direct_output),
                                lambda: plan_choice.direct_output, agent_id=self._root.id)
                if not isinstance(plan_choice.direct_output, str) or not plan_choice.direct_output.strip():
                    raise ValueError("Direct Root response requires nonempty output")
                self._enter(WorkflowPhase.PLANNING)
                trace = self._last_state.trace.clone()
                self._event(ExecutionEventType.ROOT_DIRECT_RESPONSE, trace)
                self._event(ExecutionEventType.ROOT_SYNTHESIS_COMPLETED, trace)
                self._event(ExecutionEventType.RUN_COMPLETED, trace)
                result = FinalResult(
                    task_id=working.id, root_agent_id=self._root.id, success=True,
                    content={"managers": []}, status=FinalStatus.COMPLETED,
                    manager_results=(), final_review=ReviewResult(
                        decision=ReviewDecision.PASS, reviewer_id=self._root.id,
                        metadata={"review_performed": False, "direct_response": True}),
                    final_reviews=(), revision_count=0, revisions=(), trace=trace,
                    final_output=plan_choice.direct_output.strip(),
                    usage=self._usage.summary(), metadata={"direct_response": True},
                )
                self._record(final_result=result, trace=trace)
                self._enter(WorkflowPhase.COMPLETED, result=result)
                return
        plan = self._engine.orchestrate(working)
        if self._root_planner is not None:
            from agenttree.models import ExecutionTrace
            trace = ExecutionTrace(task_id=working.id)
            for event in self._last_state.trace.events:
                trace.append(event)
            for event in plan.trace.events:
                trace.append(event)
            from dataclasses import replace as replace_plan
            plan = replace_plan(plan, trace=trace)
        self._record(triage_result=plan.triage_result, trace=plan.trace)
        self._enter(WorkflowPhase.PLANNING, result=plan)
        from agenttree.core.collaboration import _active_collaboration_session
        collaboration_session = _active_collaboration_session.get()
        if collaboration_session is not None:
            collaboration_session.activate(plan.selected_manager_ids)
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
        if result.success and self._root_synthesizer is not None:
            trace = result.trace.clone()
            self._event(ExecutionEventType.ROOT_SYNTHESIS_STARTED, trace)
            try:
                from agenttree.core.operation_journal import current_journal
                journal = current_journal()
                if journal is not None:
                    from agenttree.core.root import synthesis_context
                    output = journal.run(journal.child_key("root:synthesis"), "root.synthesis",
                                         (working.id, synthesis_context(working, result)),
                                         lambda: self._root_synthesizer.synthesize(working, self._root, result),
                                         agent_id=self._root.id)
                else:
                    output = self._root_synthesizer.synthesize(working, self._root, result)
                if not isinstance(output, str) or not output.strip():
                    raise ValueError("Root synthesis returned empty output")
            except Exception as error:
                from agenttree.core.execution_store import ExecutionRecoveryBlocked
                if isinstance(error, ExecutionRecoveryBlocked):
                    raise
                self._event(ExecutionEventType.ROOT_SYNTHESIS_FAILED, trace)
                self._event(ExecutionEventType.RUN_FAILED, trace)
                result = replace(result, success=False, status=FinalStatus.FAILED,
                                 final_output=None, trace=trace, usage=self._usage.summary(),
                                 error={"type": type(error).__name__,
                                        "message": "Root synthesis failed"})
            else:
                self._event(ExecutionEventType.ROOT_SYNTHESIS_COMPLETED, trace)
                self._event(ExecutionEventType.RUN_COMPLETED, trace)
                result = replace(result, final_output=output.strip(), trace=trace,
                                 usage=self._usage.summary())
        elif not result.success:
            trace = result.trace.clone()
            self._event(ExecutionEventType.RUN_FAILED, trace)
            result = replace(result, trace=trace, usage=self._usage.summary(),
                             error={"type": result.status.value,
                                    "message": "Root final review did not pass"})
        self._record(final_result=result, trace=result.trace, status=result.status)
        if result.success:
            self._enter(WorkflowPhase.COMPLETED, result=result)
        else:
            self._last_state = self._states.fail(
                self._last_state, reason=(result.error or {}).get("message") or
                result.final_review.feedback or result.status.value,
                status=result.status,
            )
