"""Deterministic orchestration planning foundation."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable


from agenttree.agents import ManagerAgent, RootAgent, SpecialistAgent
from agenttree.core import (
    BaseFinalReviewer,
    BaseManagerReviewer,
    BaseSpecialistExecutor,
    BaseTaskDecomposer,
    BaseTaskTriage,
)
from agenttree.models import (
    AgentResult,
    ExecutionEvent,
    ExecutionTrace,
    ReviewDecision,
    ReviewResult,
    Subtask,
    Task,
    TriageResult,
)
from agenttree.orchestration.models import (
    DelegationPlan,
    DelegationStatus,
    ExecutionStatus,
    FinalResult,
    FinalRevisionRecord,
    FinalStatus,
    ManagerDelegation,
    ManagerExecution,
    ManagerReviewResult,
    ManagerReviewStatus,
    OrchestrationEventType,
    OrchestrationPlan,
    OrchestrationStatus,
    SpecialistAssignment,
    SpecialistExecution,
    SubtaskReviewOutcome,
    TaskExecutionResult,
    TaskManagerReviewResult,
    RevisionRecord,
)
from agenttree.registry import CapabilityRegistry


def _journal_action(part: str, kind: str, identity: Any, action: Callable[[], Any],
                    agent_id: str | None = None) -> Any:
    from agenttree.core.operation_journal import current_journal
    from agenttree.core.artifacts import _active_artifact_producer, _active_artifact_operation
    role = ("specialist" if kind.startswith("specialist.") else
            "manager" if kind.startswith("manager.") else "root")
    producer_token = _active_artifact_producer.set((role, agent_id))
    operation_token = _active_artifact_operation.set(part)
    journal = current_journal()
    try:
        if journal is None:
            return action()
        return journal.run(journal.child_key(part), kind, identity, action,
                           agent_id=agent_id)
    finally:
        _active_artifact_operation.reset(operation_token)
        _active_artifact_producer.reset(producer_token)


class OrchestrationEngine:
    """Run triage and discover matching managers without executing agents.

    Dependencies are supplied by the host application. Manager discovery uses
    ANY-capability matching by default; ``manager_match_all=True`` requires each
    selected manager to advertise every required capability.
    """

    def __init__(
        self,
        *,
        triage: BaseTaskTriage,
        registry: CapabilityRegistry,
        manager_match_all: bool = False,
    ) -> None:
        if not isinstance(triage, BaseTaskTriage):
            raise TypeError("triage must be a BaseTaskTriage")
        if not isinstance(registry, CapabilityRegistry):
            raise TypeError("registry must be a CapabilityRegistry")
        if not isinstance(manager_match_all, bool):
            raise TypeError("manager_match_all must be a bool")
        self._triage = triage
        self._registry = registry
        self._manager_match_all = manager_match_all

    @property
    def triage(self) -> BaseTaskTriage:
        """Return the injected task triage implementation."""
        return self._triage

    @property
    def registry(self) -> CapabilityRegistry:
        """Return the injected capability registry."""
        return self._registry

    @property
    def manager_match_all(self) -> bool:
        """Return whether manager discovery requires all capabilities."""
        return self._manager_match_all

    def orchestrate(self, task: Task) -> OrchestrationPlan:
        """Triage ``task``, discover managers, and return a traced plan."""
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

        trace = ExecutionTrace(task_id=task.id)
        trace.append(ExecutionEvent(
            event_type=OrchestrationEventType.STARTED.value,
            task_id=task.id,
            message="Orchestration planning started",
        ))

        available_manager_capabilities = self._available_capabilities(
            ManagerAgent,
        )
        triage_result = self._triage.triage_with_capabilities(
            task,
            available_manager_capabilities,
        )
        self._validate_triage_result(task, triage_result)
        trace.append(ExecutionEvent(
            event_type=OrchestrationEventType.TRIAGE_COMPLETED.value,
            task_id=task.id,
            message="Task triage completed",
            metadata={
                "available_manager_capabilities": available_manager_capabilities,
                "required_capabilities": triage_result.required_capabilities,
            },
        ))

        managers = self._registry.find_by_capabilities(
            triage_result.required_capabilities,
            match_all=self._manager_match_all,
            agent_type=ManagerAgent,
        )
        manager_ids = tuple(manager.id for manager in managers)
        trace.append(ExecutionEvent(
            event_type=OrchestrationEventType.MANAGER_DISCOVERY_COMPLETED.value,
            task_id=task.id,
            message="Manager discovery completed",
            metadata={"selected_manager_ids": manager_ids},
        ))

        status = (
            OrchestrationStatus.READY
            if manager_ids
            else OrchestrationStatus.NO_MANAGER
        )
        if status is OrchestrationStatus.NO_MANAGER:
            trace.append(ExecutionEvent(
                event_type=OrchestrationEventType.NO_MANAGER.value,
                task_id=task.id,
                message="No matching manager was found",
                metadata={
                    "required_capabilities": triage_result.required_capabilities,
                },
            ))

        trace.append(ExecutionEvent(
            event_type=OrchestrationEventType.PLANNING_COMPLETED.value,
            task_id=task.id,
            message="Orchestration planning completed",
            metadata={"status": status.value},
        ))
        return OrchestrationPlan(
            task_id=task.id,
            objective=triage_result.objective,
            required_capabilities=triage_result.required_capabilities,
            selected_manager_ids=manager_ids,
            status=status,
            trace=trace,
            triage_result=triage_result,
        )

    def delegate(
        self,
        plan: OrchestrationPlan,
        task: Task,
        *,
        decomposer: BaseTaskDecomposer,
        specialist_match_all: bool = False,
    ) -> DelegationPlan:
        """Decompose selected-manager work and discover owned specialists.

        Specialist matching uses ANY required capability by default. Registry
        order is retained, then results are restricted to specialists registered
        under each manager. No agent or provider is executed.
        """
        self._validate_delegation_inputs(
            plan, task, decomposer, specialist_match_all,
        )
        triage_result = plan.triage_result or TriageResult(
            task_id=plan.task_id,
            objective=plan.objective,
            required_capabilities=plan.required_capabilities,
        )
        manager_delegations: list[ManagerDelegation] = []
        seen_subtask_ids: set[str] = set()

        for manager_id in dict.fromkeys(plan.selected_manager_ids):
            manager = self._get_selected_manager(manager_id)
            owned_specialist_ids = {
                specialist.id for specialist in manager.specialists
            }
            available_specialist_capabilities = self._available_capabilities(
                SpecialistAgent,
                eligible_ids=owned_specialist_ids,
            )
            plan.trace.append(ExecutionEvent(
                event_type=OrchestrationEventType.MANAGER_DELEGATION_STARTED.value,
                task_id=task.id,
                actor_id=manager.id,
                message="Manager delegation started",
                metadata={
                    "manager_id": manager.id,
                    "available_specialist_capabilities": (
                        available_specialist_capabilities
                    ),
                },
            ))
            subtasks = _journal_action(
                f"manager:{manager.id}:decompose", "manager.decompose",
                (task.id, manager.id, triage_result, available_specialist_capabilities),
                lambda: decomposer.decompose_with_capabilities(
                    task, manager, triage_result, available_specialist_capabilities),
                manager.id)
            self._validate_subtasks(task, manager, subtasks, seen_subtask_ids)
            assignments: list[SpecialistAssignment] = []

            for subtask in subtasks:
                plan.trace.append(ExecutionEvent(
                    event_type=OrchestrationEventType.SUBTASK_CREATED.value,
                    task_id=task.id,
                    actor_id=manager.id,
                    message="Subtask created",
                    metadata={
                        "subtask_id": subtask.id,
                        "available_specialist_capabilities": (
                            available_specialist_capabilities
                        ),
                        "required_capabilities": subtask.required_capabilities,
                    },
                ))
                specialists = self._find_owned_specialists(
                    manager, subtask, specialist_match_all,
                )
                specialist_ids = tuple(item.id for item in specialists)
                assignment_status = (
                    DelegationStatus.READY
                    if specialist_ids
                    else DelegationStatus.NO_SPECIALIST
                )
                assignments.append(SpecialistAssignment(
                    subtask_id=subtask.id,
                    specialist_ids=specialist_ids,
                    status=assignment_status,
                ))
                plan.trace.append(ExecutionEvent(
                    event_type=(
                        OrchestrationEventType.SPECIALIST_DISCOVERY_COMPLETED.value
                    ),
                    task_id=task.id,
                    actor_id=manager.id,
                    message="Specialist discovery completed",
                    metadata={
                        "subtask_id": subtask.id,
                        "specialist_ids": specialist_ids,
                        "available_specialist_capabilities": (
                            available_specialist_capabilities
                        ),
                        "requested_capabilities": subtask.required_capabilities,
                    },
                ))
                if assignment_status is DelegationStatus.NO_SPECIALIST:
                    plan.trace.append(ExecutionEvent(
                        event_type=OrchestrationEventType.NO_SPECIALIST.value,
                        task_id=task.id,
                        actor_id=manager.id,
                        message="No eligible specialist was found",
                        metadata={"subtask_id": subtask.id},
                    ))

            manager_status = self._manager_delegation_status(assignments)
            delegation = ManagerDelegation(
                manager_id=manager.id,
                subtasks=subtasks,
                assignments=tuple(assignments),
                status=manager_status,
            )
            manager_delegations.append(delegation)
            plan.trace.append(ExecutionEvent(
                event_type=OrchestrationEventType.MANAGER_DELEGATION_COMPLETED.value,
                task_id=task.id,
                actor_id=manager.id,
                message="Manager delegation completed",
                metadata={
                    "manager_id": manager.id,
                    "status": manager_status.value,
                },
            ))

        delegations = tuple(manager_delegations)
        return DelegationPlan(
            task_id=task.id,
            manager_delegations=delegations,
            status=self._overall_delegation_status(delegations),
            trace=plan.trace,
        )

    def execute(
        self,
        delegation: DelegationPlan,
        task: Task,
        *,
        executor: BaseSpecialistExecutor,
    ) -> TaskExecutionResult:
        """Sequentially execute assigned specialists and aggregate outcomes."""
        self._validate_execution_inputs(delegation, task, executor)
        trace = delegation.trace.clone()
        trace.append(ExecutionEvent(
            event_type=OrchestrationEventType.EXECUTION_STARTED.value,
            task_id=task.id,
            message="Specialist execution started",
        ))
        manager_executions: list[ManagerExecution] = []

        for manager_delegation in delegation.manager_delegations:
            manager = self._get_selected_manager(manager_delegation.manager_id)
            subtasks = self._validate_manager_execution_input(
                task, manager_delegation,
            )
            executions: list[SpecialistExecution] = []
            owned_ids = {specialist.id for specialist in manager.specialists}

            for assignment in manager_delegation.assignments:
                subtask = subtasks[assignment.subtask_id]
                if assignment.status is DelegationStatus.NO_SPECIALIST:
                    if assignment.specialist_ids:
                        raise ValueError(
                            "NO_SPECIALIST assignments cannot contain specialist IDs",
                        )
                    executions.append(SpecialistExecution(
                        subtask_id=subtask.id,
                        specialist_id=None,
                        status=ExecutionStatus.SKIPPED,
                    ))
                    trace.append(ExecutionEvent(
                        event_type=OrchestrationEventType.ASSIGNMENT_SKIPPED.value,
                        task_id=task.id,
                        actor_id=manager.id,
                        message="Assignment skipped because no specialist was selected",
                        metadata={"subtask_id": subtask.id},
                    ))
                    continue
                if not assignment.specialist_ids:
                    raise ValueError(
                        "Executable assignments must contain specialist IDs",
                    )

                for specialist_id in assignment.specialist_ids:
                    specialist = self._get_assigned_specialist(
                        specialist_id, owned_ids, manager.id,
                    )
                    trace.append(ExecutionEvent(
                        event_type=(
                            OrchestrationEventType.SPECIALIST_EXECUTION_STARTED.value
                        ),
                        task_id=task.id,
                        actor_id=specialist.id,
                        message="Specialist execution started",
                        metadata={
                            "manager_id": manager.id,
                            "subtask_id": subtask.id,
                        },
                    ))
                    agent_result = _journal_action(
                        f"manager:{manager.id}:subtask:{subtask.id}:specialist:{specialist.id}",
                        "specialist.generate", (task.id, subtask, specialist.id),
                        lambda: executor.execute(task, subtask, specialist), specialist.id)
                    self._validate_agent_result(specialist.id, agent_result)
                    execution_status = (
                        ExecutionStatus.COMPLETED
                        if agent_result.success
                        else ExecutionStatus.FAILED
                    )
                    executions.append(SpecialistExecution(
                        subtask_id=subtask.id,
                        specialist_id=specialist.id,
                        status=execution_status,
                        agent_result=agent_result,
                    ))
                    event_type = (
                        OrchestrationEventType.SPECIALIST_EXECUTION_COMPLETED
                        if agent_result.success
                        else OrchestrationEventType.SPECIALIST_EXECUTION_FAILED
                    )
                    trace.append(ExecutionEvent(
                        event_type=event_type.value,
                        task_id=task.id,
                        actor_id=specialist.id,
                        message=(
                            "Specialist execution completed"
                            if agent_result.success
                            else "Specialist execution failed"
                        ),
                        metadata={
                            "manager_id": manager.id,
                            "subtask_id": subtask.id,
                            "error": agent_result.error,
                        },
                    ))

            manager_executions.append(ManagerExecution(
                manager_id=manager.id,
                specialist_executions=tuple(executions),
                status=self._execution_status(executions),
            ))

        manager_results = tuple(manager_executions)
        overall_status = self._manager_execution_status(manager_results)
        trace.append(ExecutionEvent(
            event_type=OrchestrationEventType.EXECUTION_COMPLETED.value,
            task_id=task.id,
            message="Specialist execution phase completed",
            metadata={"status": overall_status.value},
        ))
        return TaskExecutionResult(
            task_id=task.id,
            manager_executions=manager_results,
            status=overall_status,
            trace=trace,
            delegation=delegation,
        )

    def review(
        self,
        execution: TaskExecutionResult,
        task: Task,
        *,
        reviewer: BaseManagerReviewer,
        executor: BaseSpecialistExecutor,
        max_revisions: int = 2,
    ) -> TaskManagerReviewResult:
        """Review specialist results and coordinate bounded revision rounds.

        The initial execution is revision zero. ``max_revisions`` is the number
        of feedback-driven re-execution rounds permitted for each subtask.
        """
        delegation = self._validate_review_inputs(
            execution, task, reviewer, executor, max_revisions,
        )
        trace = execution.trace.clone()
        delegations = {
            item.manager_id: item for item in delegation.manager_delegations
        }
        manager_results: list[ManagerReviewResult] = []

        for manager_execution in execution.manager_executions:
            manager = self._get_selected_manager(manager_execution.manager_id)
            try:
                manager_delegation = delegations[manager.id]
            except KeyError as error:
                raise ValueError(
                    f"No delegation for executed manager: {manager.id}",
                ) from error
            subtask_by_id = {
                subtask.id: subtask for subtask in manager_delegation.subtasks
            }
            assignment_by_id = {
                assignment.subtask_id: assignment
                for assignment in manager_delegation.assignments
            }
            initial_by_subtask: dict[str, list[SpecialistExecution]] = {
                subtask.id: [] for subtask in manager_delegation.subtasks
            }
            for specialist_execution in manager_execution.specialist_executions:
                try:
                    initial_by_subtask[specialist_execution.subtask_id].append(
                        specialist_execution,
                    )
                except KeyError as error:
                    raise ValueError(
                        "Execution contains an unknown Subtask id",
                    ) from error

            outcomes: list[SubtaskReviewOutcome] = []
            for subtask in manager_delegation.subtasks:
                assignment = assignment_by_id[subtask.id]
                initial_executions = tuple(initial_by_subtask[subtask.id])
                outcomes.append(self._review_subtask(
                    task=task,
                    subtask=subtask_by_id[subtask.id],
                    manager=manager,
                    initial_executions=initial_executions,
                    specialist_ids=assignment.specialist_ids,
                    reviewer=reviewer,
                    executor=executor,
                    max_revisions=max_revisions,
                    trace=trace,
                ))

            outcome_tuple = tuple(outcomes)
            manager_status = self._review_aggregate_status(
                tuple(item.status for item in outcome_tuple),
            )
            manager_results.append(ManagerReviewResult(
                manager_id=manager.id,
                subtask_outcomes=outcome_tuple,
                status=manager_status,
            ))
            trace.append(ExecutionEvent(
                event_type=OrchestrationEventType.MANAGER_REVIEW_COMPLETED.value,
                task_id=task.id,
                actor_id=manager.id,
                message="Manager review completed",
                metadata={"status": manager_status.value},
            ))

        results = tuple(manager_results)
        return TaskManagerReviewResult(
            task_id=task.id,
            manager_results=results,
            status=self._review_aggregate_status(
                tuple(item.status for item in results),
            ),
            trace=trace,
            execution=execution,
        )

    def finalize(
        self,
        manager_review_result: TaskManagerReviewResult,
        task: Task,
        *,
        root_agent: RootAgent,
        final_reviewer: BaseFinalReviewer,
        manager_reviewer: BaseManagerReviewer,
        executor: BaseSpecialistExecutor,
        max_manager_revisions: int = 2,
        max_final_revisions: int = 1,
    ) -> FinalResult:
        """Run Root-level review and return a normalized terminal result.

        Final-review REVISE reconsiders every manager in input order through
        :meth:`review`. The initial final review is revision zero, and the two
        configured revision limits remain independent.
        """
        self._validate_finalization_inputs(
            manager_review_result=manager_review_result,
            task=task,
            root_agent=root_agent,
            final_reviewer=final_reviewer,
            manager_reviewer=manager_reviewer,
            executor=executor,
            max_manager_revisions=max_manager_revisions,
            max_final_revisions=max_final_revisions,
        )
        trace = manager_review_result.trace.clone()
        current = manager_review_result
        final_reviews: list[ReviewResult] = []
        revisions: list[FinalRevisionRecord] = []
        revision_count = 0

        while True:
            trace.append(ExecutionEvent(
                event_type=OrchestrationEventType.FINAL_REVIEW_STARTED.value,
                task_id=task.id,
                actor_id=root_agent.id,
                message="Root final review started",
                metadata={"review_number": len(final_reviews) + 1},
            ))
            final_review = _journal_action(
                f"root:final_review:{len(final_reviews) + 1}", "root.final_review",
                (task.id, root_agent.id, current.manager_results),
                lambda: final_reviewer.review(task, root_agent, current), root_agent.id)
            self._validate_final_review_result(root_agent.id, final_review)
            final_reviews.append(final_review)

            if final_review.decision is ReviewDecision.PASS:
                status = (FinalStatus.COMPLETED if current.status is ManagerReviewStatus.PASSED
                          else FinalStatus.PARTIAL)
                success = current.status in (ManagerReviewStatus.PASSED,
                                             ManagerReviewStatus.PARTIAL)
                trace.append(ExecutionEvent(
                    event_type=OrchestrationEventType.FINAL_REVIEW_PASSED.value,
                    task_id=task.id,
                    actor_id=root_agent.id,
                    message="Root final review passed",
                    metadata={"manager_review_status": current.status.value},
                ))
                break

            if final_review.decision is ReviewDecision.FAIL:
                status = FinalStatus.FAILED
                success = False
                trace.append(ExecutionEvent(
                    event_type=OrchestrationEventType.FINAL_REVIEW_FAILED.value,
                    task_id=task.id,
                    actor_id=root_agent.id,
                    message="Root final review failed",
                    metadata={"feedback": final_review.feedback},
                ))
                break

            trace.append(ExecutionEvent(
                event_type=(
                    OrchestrationEventType.FINAL_REVIEW_REVISION_REQUESTED.value
                ),
                task_id=task.id,
                actor_id=root_agent.id,
                message="Root final review requested manager reconsideration",
                metadata={
                    "feedback": final_review.feedback,
                    "requested_revision_number": revision_count + 1,
                    "manager_ids": tuple(
                        item.manager_id for item in current.manager_results
                    ),
                },
            ))
            if revision_count >= max_final_revisions:
                status = FinalStatus.FINAL_REVISION_LIMIT_REACHED
                success = False
                trace.append(ExecutionEvent(
                    event_type=(
                        OrchestrationEventType.FINAL_REVISION_LIMIT_REACHED.value
                    ),
                    task_id=task.id,
                    actor_id=root_agent.id,
                    message="Final-review revision limit reached",
                    metadata={"max_final_revisions": max_final_revisions},
                ))
                break

            revision_count += 1
            trace.append(ExecutionEvent(
                event_type=OrchestrationEventType.FINAL_REVISION_STARTED.value,
                task_id=task.id,
                actor_id=root_agent.id,
                message="Manager reconsideration started",
                metadata={
                    "revision_number": revision_count,
                    "feedback": final_review.feedback,
                },
            ))
            revision_task = self._final_revision_task(
                task=task,
                root_agent_id=root_agent.id,
                revision_number=revision_count,
                feedback=final_review.feedback,
            )
            revision_execution = self._execution_from_manager_review(current)
            revised = self.review(
                revision_execution,
                revision_task,
                reviewer=manager_reviewer,
                executor=executor,
                max_revisions=max_manager_revisions,
            )
            for event in revised.trace.events[len(revision_execution.trace.events):]:
                trace.append(event)
            current = revised
            revisions.append(FinalRevisionRecord(
                revision_number=revision_count,
                feedback=final_review.feedback,
                manager_review_result=current,
            ))
            trace.append(ExecutionEvent(
                event_type=OrchestrationEventType.FINAL_REVISION_COMPLETED.value,
                task_id=task.id,
                actor_id=root_agent.id,
                message="Manager reconsideration completed",
                metadata={
                    "revision_number": revision_count,
                    "manager_review_status": current.status.value,
                },
            ))

        content = self._aggregate_final_content(current)
        trace.append(ExecutionEvent(
            event_type=OrchestrationEventType.FINAL_RESULT_CREATED.value,
            task_id=task.id,
            actor_id=root_agent.id,
            message="Final result created",
            metadata={"status": status.value, "success": success},
        ))
        return FinalResult(
            task_id=task.id,
            root_agent_id=root_agent.id,
            success=success,
            content=content,
            status=status,
            manager_results=current.manager_results,
            final_review=final_reviews[-1],
            final_reviews=tuple(final_reviews),
            revision_count=revision_count,
            revisions=tuple(revisions),
            trace=trace,
        )

    @staticmethod
    def _final_revision_task(
        *,
        task: Task,
        root_agent_id: str,
        revision_number: int,
        feedback: str,
    ) -> Task:
        metadata = deepcopy(task.metadata)
        metadata["final_revision"] = {
            "root_agent_id": root_agent_id,
            "revision_number": revision_number,
            "feedback": feedback,
        }
        return replace(
            task,
            context=deepcopy(task.context),
            metadata=metadata,
        )

    def _execution_from_manager_review(
        self,
        result: TaskManagerReviewResult,
    ) -> TaskExecutionResult:
        if result.execution is None or result.execution.delegation is None:
            raise ValueError(
                "Manager review result cannot be revised without its execution context",
            )
        managers: list[ManagerExecution] = []
        for manager_result in result.manager_results:
            executions: list[SpecialistExecution] = []
            for outcome in manager_result.subtask_outcomes:
                latest = (
                    outcome.revisions[-1].executions
                    if outcome.revisions
                    else outcome.executions
                )
                executions.extend(latest)
            managers.append(ManagerExecution(
                manager_id=manager_result.manager_id,
                specialist_executions=tuple(executions),
                status=self._execution_status(executions),
            ))
        manager_tuple = tuple(managers)
        return TaskExecutionResult(
            task_id=result.task_id,
            manager_executions=manager_tuple,
            status=self._manager_execution_status(manager_tuple),
            trace=result.trace,
            delegation=result.execution.delegation,
        )

    @staticmethod
    def _aggregate_final_content(
        result: TaskManagerReviewResult,
    ) -> dict[str, object]:
        managers: list[dict[str, object]] = []
        for manager in result.manager_results:
            subtasks: list[dict[str, object]] = []
            for outcome in manager.subtask_outcomes:
                latest = (
                    outcome.revisions[-1].executions
                    if outcome.revisions
                    else outcome.executions
                )
                specialists: list[dict[str, object]] = []
                if outcome.decision is ReviewDecision.PASS:
                    for execution in latest:
                        if execution.agent_result is None:
                            continue
                        agent_result = execution.agent_result
                        specialists.append({
                            "specialist_id": execution.specialist_id,
                            "success": agent_result.success,
                            "output": deepcopy(agent_result.output),
                            "error": agent_result.error,
                            "metadata": deepcopy(agent_result.metadata),
                        })
                subtasks.append({
                    "subtask_id": outcome.subtask_id,
                    "status": outcome.status.value,
                    "decision": outcome.decision.value,
                    "feedback": outcome.feedback,
                    "manager_revision_count": outcome.revision_count,
                    "specialists": specialists,
                })
            managers.append({
                "manager_id": manager.manager_id,
                "status": manager.status.value,
                "subtasks": subtasks,
            })
        return {"managers": managers}

    @staticmethod
    def _validate_finalization_inputs(
        *,
        manager_review_result: object,
        task: object,
        root_agent: object,
        final_reviewer: object,
        manager_reviewer: object,
        executor: object,
        max_manager_revisions: object,
        max_final_revisions: object,
    ) -> None:
        if not isinstance(manager_review_result, TaskManagerReviewResult):
            raise TypeError(
                "manager_review_result must be a TaskManagerReviewResult",
            )
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if (
            manager_review_result.task_id != task.id
            or manager_review_result.trace.task_id != task.id
        ):
            raise ValueError("Manager review and trace must match the input Task id")
        if not isinstance(root_agent, RootAgent):
            raise TypeError("root_agent must be a RootAgent")
        if not isinstance(final_reviewer, BaseFinalReviewer):
            raise TypeError("final_reviewer must be a BaseFinalReviewer")
        if not isinstance(manager_reviewer, BaseManagerReviewer):
            raise TypeError("manager_reviewer must be a BaseManagerReviewer")
        if not isinstance(executor, BaseSpecialistExecutor):
            raise TypeError("executor must be a BaseSpecialistExecutor")
        for name, value in (
            ("max_manager_revisions", max_manager_revisions),
            ("max_final_revisions", max_final_revisions),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")

    @staticmethod
    def _validate_final_review_result(
        root_agent_id: str,
        result: object,
    ) -> None:
        if not isinstance(result, ReviewResult):
            raise TypeError("final_reviewer must return a ReviewResult")
        if result.reviewer_id != root_agent_id:
            raise ValueError("ReviewResult reviewer_id must match the RootAgent id")
        if not isinstance(result.decision, ReviewDecision):
            raise TypeError("ReviewResult decision must be a ReviewDecision")

    def _review_subtask(
        self,
        *,
        task: Task,
        subtask: Subtask,
        manager: ManagerAgent,
        initial_executions: tuple[SpecialistExecution, ...],
        specialist_ids: tuple[str, ...],
        reviewer: BaseManagerReviewer,
        executor: BaseSpecialistExecutor,
        max_revisions: int,
        trace: ExecutionTrace,
    ) -> SubtaskReviewOutcome:
        current_executions = initial_executions
        all_executions = list(initial_executions)
        reviews: list[ReviewResult] = []
        revisions: list[RevisionRecord] = []
        revision_count = 0

        while True:
            trace.append(ExecutionEvent(
                event_type=OrchestrationEventType.MANAGER_REVIEW_STARTED.value,
                task_id=task.id,
                actor_id=manager.id,
                message="Manager review started",
                metadata={
                    "subtask_id": subtask.id,
                    "review_number": len(reviews) + 1,
                },
            ))
            review_result = _journal_action(
                f"manager:{manager.id}:subtask:{subtask.id}:review:{len(reviews) + 1}",
                "manager.review", (task.id, subtask, manager.id, current_executions),
                lambda: reviewer.review(task, subtask, manager, current_executions), manager.id)
            self._validate_review_result(manager.id, review_result)
            reviews.append(review_result)

            if review_result.decision is ReviewDecision.PASS:
                successful = sum(
                    execution.agent_result is not None and execution.agent_result.success
                    for execution in current_executions
                )
                status = (ManagerReviewStatus.PASSED if successful == len(current_executions)
                          and successful > 0 else ManagerReviewStatus.PARTIAL if successful
                          else ManagerReviewStatus.FAILED)
                trace.append(ExecutionEvent(
                    event_type=OrchestrationEventType.MANAGER_REVIEW_PASSED.value,
                    task_id=task.id,
                    actor_id=manager.id,
                    message="Manager review passed",
                    metadata={"subtask_id": subtask.id},
                ))
                break
            if review_result.decision is ReviewDecision.FAIL:
                status = ManagerReviewStatus.FAILED
                trace.append(ExecutionEvent(
                    event_type=OrchestrationEventType.MANAGER_REVIEW_FAILED.value,
                    task_id=task.id,
                    actor_id=manager.id,
                    message="Manager review failed",
                    metadata={
                        "subtask_id": subtask.id,
                        "feedback": review_result.feedback,
                    },
                ))
                break

            trace.append(ExecutionEvent(
                event_type=(
                    OrchestrationEventType.MANAGER_REVIEW_REVISION_REQUESTED.value
                ),
                task_id=task.id,
                actor_id=manager.id,
                message="Manager requested a revision",
                metadata={
                    "subtask_id": subtask.id,
                    "feedback": review_result.feedback,
                    "requested_revision_number": revision_count + 1,
                },
            ))
            if revision_count >= max_revisions:
                status = ManagerReviewStatus.REVISION_LIMIT_REACHED
                trace.append(ExecutionEvent(
                    event_type=OrchestrationEventType.REVISION_LIMIT_REACHED.value,
                    task_id=task.id,
                    actor_id=manager.id,
                    message="Revision limit reached",
                    metadata={
                        "subtask_id": subtask.id,
                        "max_revisions": max_revisions,
                    },
                ))
                break

            revision_count += 1
            revision_subtask = self._revision_subtask(
                subtask, manager.id, revision_count, review_result.feedback,
            )
            trace.append(ExecutionEvent(
                event_type=OrchestrationEventType.REVISION_STARTED.value,
                task_id=task.id,
                actor_id=manager.id,
                message="Revision execution started",
                metadata={
                    "subtask_id": subtask.id,
                    "revision_number": revision_count,
                    "specialist_ids": specialist_ids,
                },
            ))
            revision_executions = self._execute_revision(
                task=task,
                subtask=revision_subtask,
                manager=manager,
                specialist_ids=specialist_ids,
                executor=executor,
                trace=trace,
            )
            revision_status = self._execution_status(list(revision_executions))
            revisions.append(RevisionRecord(
                subtask_id=subtask.id,
                manager_id=manager.id,
                revision_number=revision_count,
                feedback=review_result.feedback,
                specialist_ids=specialist_ids,
                executions=revision_executions,
                status=revision_status,
            ))
            all_executions.extend(revision_executions)
            current_executions = revision_executions
            trace.append(ExecutionEvent(
                event_type=(
                    OrchestrationEventType.REVISION_EXECUTION_COMPLETED.value
                ),
                task_id=task.id,
                actor_id=manager.id,
                message="Revision execution completed",
                metadata={
                    "subtask_id": subtask.id,
                    "revision_number": revision_count,
                    "status": revision_status.value,
                },
            ))

        return SubtaskReviewOutcome(
            subtask_id=subtask.id,
            manager_id=manager.id,
            decision=reviews[-1].decision,
            feedback=reviews[-1].feedback,
            revision_count=revision_count,
            executions=tuple(all_executions),
            reviews=tuple(reviews),
            revisions=tuple(revisions),
            status=status,
        )

    def _execute_revision(
        self,
        *,
        task: Task,
        subtask: Subtask,
        manager: ManagerAgent,
        specialist_ids: tuple[str, ...],
        executor: BaseSpecialistExecutor,
        trace: ExecutionTrace,
    ) -> tuple[SpecialistExecution, ...]:
        owned_ids = {specialist.id for specialist in manager.specialists}
        executions: list[SpecialistExecution] = []
        for specialist_id in specialist_ids:
            specialist = self._get_assigned_specialist(
                specialist_id, owned_ids, manager.id,
            )
            trace.append(ExecutionEvent(
                event_type=OrchestrationEventType.SPECIALIST_EXECUTION_STARTED.value,
                task_id=task.id,
                actor_id=specialist.id,
                message="Revision specialist execution started",
                metadata={"subtask_id": subtask.id},
            ))
            agent_result = _journal_action(
                f"subtask:{subtask.id}:revision:{subtask.metadata['revision']['revision_number']}:specialist:{specialist.id}",
                "specialist.revision", (task.id, subtask, specialist.id),
                lambda: executor.execute(task, subtask, specialist), specialist.id)
            self._validate_agent_result(specialist.id, agent_result)
            execution_status = (
                ExecutionStatus.COMPLETED
                if agent_result.success
                else ExecutionStatus.FAILED
            )
            executions.append(SpecialistExecution(
                subtask_id=subtask.id,
                specialist_id=specialist.id,
                status=execution_status,
                agent_result=agent_result,
                metadata={
                    "revision_number": subtask.metadata["revision"][
                        "revision_number"
                    ],
                },
            ))
            trace.append(ExecutionEvent(
                event_type=(
                    OrchestrationEventType.SPECIALIST_EXECUTION_COMPLETED.value
                    if agent_result.success
                    else OrchestrationEventType.SPECIALIST_EXECUTION_FAILED.value
                ),
                task_id=task.id,
                actor_id=specialist.id,
                message=(
                    "Revision specialist execution completed"
                    if agent_result.success
                    else "Revision specialist execution failed"
                ),
                metadata={
                    "subtask_id": subtask.id,
                    "revision_number": subtask.metadata["revision"][
                        "revision_number"
                    ],
                    "error": agent_result.error,
                },
            ))
        return tuple(executions)

    @staticmethod
    def _revision_subtask(
        subtask: Subtask,
        manager_id: str,
        revision_number: int,
        feedback: str,
    ) -> Subtask:
        metadata = deepcopy(subtask.metadata)
        metadata["revision"] = {
            "manager_id": manager_id,
            "revision_number": revision_number,
            "feedback": feedback,
        }
        return replace(subtask, metadata=metadata)

    @staticmethod
    def _validate_review_inputs(
        execution: object,
        task: object,
        reviewer: object,
        executor: object,
        max_revisions: object,
    ) -> DelegationPlan:
        if not isinstance(execution, TaskExecutionResult):
            raise TypeError("execution must be a TaskExecutionResult")
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if execution.task_id != task.id or execution.trace.task_id != task.id:
            raise ValueError("Execution and trace must match the input Task id")
        if execution.delegation is None:
            raise ValueError("Execution result does not contain its DelegationPlan")
        delegation = execution.delegation
        if delegation.task_id != task.id or delegation.trace.task_id != task.id:
            raise ValueError("Delegation and trace must match the input Task id")
        if not isinstance(reviewer, BaseManagerReviewer):
            raise TypeError("reviewer must be a BaseManagerReviewer")
        if not isinstance(executor, BaseSpecialistExecutor):
            raise TypeError("executor must be a BaseSpecialistExecutor")
        if (
            not isinstance(max_revisions, int)
            or isinstance(max_revisions, bool)
            or max_revisions < 0
        ):
            raise ValueError("max_revisions must be a non-negative integer")
        manager_ids = tuple(item.manager_id for item in execution.manager_executions)
        delegation_ids = tuple(
            item.manager_id for item in delegation.manager_delegations
        )
        if len(set(manager_ids)) != len(manager_ids):
            raise ValueError("Execution contains duplicate manager results")
        if len(set(delegation_ids)) != len(delegation_ids):
            raise ValueError("Delegation contains duplicate managers")
        if manager_ids != delegation_ids:
            raise ValueError("Execution managers must match delegation order")
        return delegation

    @staticmethod
    def _validate_review_result(manager_id: str, result: object) -> None:
        if not isinstance(result, ReviewResult):
            raise TypeError("reviewer must return a ReviewResult")
        if result.reviewer_id != manager_id:
            raise ValueError("ReviewResult reviewer_id must match the ManagerAgent id")
        if not isinstance(result.decision, ReviewDecision):
            raise TypeError("ReviewResult decision must be a ReviewDecision")

    @staticmethod
    def _review_aggregate_status(
        statuses: tuple[ManagerReviewStatus, ...],
    ) -> ManagerReviewStatus:
        if not statuses:
            return ManagerReviewStatus.SKIPPED
        unique = set(statuses)
        if len(unique) == 1:
            return statuses[0]
        return ManagerReviewStatus.PARTIAL

    def _get_assigned_specialist(
        self,
        specialist_id: str,
        owned_ids: set[str],
        manager_id: str,
    ) -> SpecialistAgent:
        try:
            specialist = self._registry.get(specialist_id)
        except KeyError as error:
            raise ValueError(
                f"Assigned specialist is no longer registered: {specialist_id}",
            ) from error
        if not isinstance(specialist, SpecialistAgent):
            raise TypeError(f"Assigned agent is not a SpecialistAgent: {specialist_id}")
        if specialist.id not in owned_ids:
            raise ValueError(
                f"Specialist {specialist_id} is not owned by manager {manager_id}",
            )
        return specialist

    @staticmethod
    def _validate_execution_inputs(
        delegation: object,
        task: object,
        executor: object,
    ) -> None:
        if not isinstance(delegation, DelegationPlan):
            raise TypeError("delegation must be a DelegationPlan")
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if delegation.task_id != task.id or delegation.trace.task_id != task.id:
            raise ValueError(
                "Delegation and trace task IDs must match the input Task id",
            )
        if not isinstance(executor, BaseSpecialistExecutor):
            raise TypeError("executor must be a BaseSpecialistExecutor")

    @staticmethod
    def _validate_manager_execution_input(
        task: Task,
        delegation: ManagerDelegation,
    ) -> dict[str, Subtask]:
        subtasks = {subtask.id: subtask for subtask in delegation.subtasks}
        if len(subtasks) != len(delegation.subtasks):
            raise ValueError("Manager delegation contains duplicate Subtask IDs")
        if any(
            subtask.parent_task_id != task.id
            or subtask.manager_id != delegation.manager_id
            for subtask in delegation.subtasks
        ):
            raise ValueError("Manager delegation contains incorrectly linked Subtasks")
        assignment_ids = tuple(
            assignment.subtask_id for assignment in delegation.assignments
        )
        if len(set(assignment_ids)) != len(assignment_ids):
            raise ValueError("Manager delegation contains duplicate assignments")
        if set(assignment_ids) != set(subtasks):
            raise ValueError("Every Subtask must have exactly one assignment")
        return subtasks

    @staticmethod
    def _validate_agent_result(
        specialist_id: str,
        result: object,
    ) -> None:
        if not isinstance(result, AgentResult):
            raise TypeError("executor must return an AgentResult")
        if result.agent_id != specialist_id:
            raise ValueError("AgentResult agent_id must match the SpecialistAgent id")

    @staticmethod
    def _execution_status(
        executions: list[SpecialistExecution],
    ) -> ExecutionStatus:
        if not executions:
            return ExecutionStatus.SKIPPED
        statuses = {execution.status for execution in executions}
        if statuses == {ExecutionStatus.COMPLETED}:
            return ExecutionStatus.COMPLETED
        if statuses == {ExecutionStatus.FAILED}:
            return ExecutionStatus.FAILED
        if statuses == {ExecutionStatus.SKIPPED}:
            return ExecutionStatus.SKIPPED
        return ExecutionStatus.PARTIAL

    @staticmethod
    def _manager_execution_status(
        managers: tuple[ManagerExecution, ...],
    ) -> ExecutionStatus:
        if not managers:
            return ExecutionStatus.SKIPPED
        statuses = {manager.status for manager in managers}
        if statuses == {ExecutionStatus.COMPLETED}:
            return ExecutionStatus.COMPLETED
        if statuses == {ExecutionStatus.FAILED}:
            return ExecutionStatus.FAILED
        if statuses == {ExecutionStatus.SKIPPED}:
            return ExecutionStatus.SKIPPED
        return ExecutionStatus.PARTIAL

    def _get_selected_manager(self, manager_id: str) -> ManagerAgent:
        try:
            manager = self._registry.get(manager_id)
        except KeyError as error:
            raise ValueError(
                f"Selected manager is no longer registered: {manager_id}",
            ) from error
        if not isinstance(manager, ManagerAgent):
            raise TypeError(f"Selected agent is not a ManagerAgent: {manager_id}")
        return manager

    def _available_capabilities(
        self,
        agent_type: type[ManagerAgent] | type[SpecialistAgent],
        *,
        eligible_ids: set[str] | None = None,
    ) -> tuple[str, ...]:
        """Return normalized registry choices in first-registration order."""
        return tuple(
            capability
            for capability, agents in self._registry.capability_index.items()
            if any(
                isinstance(agent, agent_type)
                and (eligible_ids is None or agent.id in eligible_ids)
                for agent in agents
            )
        )

    def _find_owned_specialists(
        self,
        manager: ManagerAgent,
        subtask: Subtask,
        match_all: bool,
    ) -> tuple[SpecialistAgent, ...]:
        compatible = self._registry.find_by_capabilities(
            subtask.required_capabilities,
            match_all=match_all,
            agent_type=SpecialistAgent,
        )
        owned_ids = {specialist.id for specialist in manager.specialists}
        return tuple(
            specialist
            for specialist in compatible
            if specialist.id in owned_ids
        )

    @staticmethod
    def _validate_delegation_inputs(
        plan: object,
        task: object,
        decomposer: object,
        specialist_match_all: object,
    ) -> None:
        if not isinstance(plan, OrchestrationPlan):
            raise TypeError("plan must be an OrchestrationPlan")
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if plan.task_id != task.id or plan.trace.task_id != task.id:
            raise ValueError("Plan and trace task IDs must match the input Task id")
        if plan.triage_result is not None and plan.triage_result.task_id != task.id:
            raise ValueError("Plan triage result must match the input Task id")
        if not isinstance(decomposer, BaseTaskDecomposer):
            raise TypeError("decomposer must be a BaseTaskDecomposer")
        if not isinstance(specialist_match_all, bool):
            raise TypeError("specialist_match_all must be a bool")

    @staticmethod
    def _validate_subtasks(
        task: Task,
        manager: ManagerAgent,
        subtasks: object,
        seen_ids: set[str],
    ) -> None:
        if not isinstance(subtasks, tuple) or any(
            not isinstance(subtask, Subtask) for subtask in subtasks
        ):
            raise TypeError("decomposer must return a tuple of Subtask instances")
        for subtask in subtasks:
            if subtask.parent_task_id != task.id:
                raise ValueError("Subtask parent_task_id must match the Task id")
            if subtask.manager_id != manager.id:
                raise ValueError("Subtask manager_id must match the selected manager")
            if subtask.id in seen_ids:
                raise ValueError(f"Duplicate Subtask id: {subtask.id}")
            seen_ids.add(subtask.id)

    @staticmethod
    def _manager_delegation_status(
        assignments: list[SpecialistAssignment],
    ) -> DelegationStatus:
        if not assignments:
            return DelegationStatus.NO_SUBTASKS
        ready_count = sum(
            assignment.status is DelegationStatus.READY
            for assignment in assignments
        )
        if ready_count == len(assignments):
            return DelegationStatus.READY
        if ready_count:
            return DelegationStatus.PARTIAL
        return DelegationStatus.NO_SPECIALIST

    @staticmethod
    def _overall_delegation_status(
        delegations: tuple[ManagerDelegation, ...],
    ) -> DelegationStatus:
        if not delegations:
            return DelegationStatus.NO_MANAGER
        statuses = {delegation.status for delegation in delegations}
        if statuses == {DelegationStatus.READY}:
            return DelegationStatus.READY
        if statuses == {DelegationStatus.NO_SUBTASKS}:
            return DelegationStatus.NO_SUBTASKS
        if statuses == {DelegationStatus.NO_SPECIALIST}:
            return DelegationStatus.NO_SPECIALIST
        return DelegationStatus.PARTIAL

    @staticmethod
    def _validate_triage_result(task: Task, result: object) -> None:
        if not isinstance(result, TriageResult):
            raise TypeError("triage must return a TriageResult")
        if result.task_id != task.id:
            raise ValueError("TriageResult task_id must match the input Task id")
