"""Public SDK workflows, registration transactions, and state ownership."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from agenttree import AgentTree, AgentTreeConfig, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    BaseSpecialistExecutor, ProviderFinalReviewer, ProviderManagerReviewer,
    ProviderTaskDecomposer, ProviderTaskTriage, RuleBasedTaskTriage,
    StaticFinalReviewer, StaticManagerReviewer, StaticTaskDecomposer,
)
from agenttree.exceptions import DecisionParseError
from agenttree.models import (
    AgentResult, ExecutionEvent, ReviewDecision, Subtask, SubtaskTemplate,
    TaskContext, WorkflowPhase,
)
from agenttree.orchestration import FinalResult, FinalStatus, ExecutionStatus
from agenttree.providers import (
    MockProvider, ProviderConfig, ProviderRegistry, ProviderRequest,
    ProviderResponse, ProviderRuntimeError,
)
from agenttree.registry import CapabilityRegistry
from agenttree.tools import FunctionTool, ToolBindingRegistry, ToolRegistry
from agenttree.tools.mcp import MCPTool, MCPToolDefinition, MockMCPClient


def framework(**overrides: Any) -> AgentTree:
    options = dict(
        root_agent=RootAgent(name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        decomposer=StaticTaskDecomposer((SubtaskTemplate("Work", ("work",)),)),
        manager_reviewer=StaticManagerReviewer(), final_reviewer=StaticFinalReviewer(),
    )
    options.update(overrides)
    return AgentTree(**options)


def register_pair(tree: AgentTree) -> tuple[ManagerAgent, SpecialistAgent]:
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    specialist = SpecialistAgent(name="Specialist", capabilities=("work",))
    tree.register_manager(manager)
    tree.register_specialist(manager, specialist)
    return manager, specialist


def ready(**overrides: Any) -> tuple[AgentTree, MockProvider]:
    tree = framework(**overrides)
    _, specialist = register_pair(tree)
    provider = MockProvider(response_content="finished")
    tree.register_provider(provider)
    tree.bind_provider(specialist, provider)
    return tree, provider


def test_public_sdk_runs_complete_workflow_and_preserves_task_and_trace() -> None:
    tree, provider = ready()
    task = Task(objective="Work", context=TaskContext({"nested": [1]}), metadata={"x": [1]})
    before = deepcopy(task)
    assert tree.last_state is None
    result = tree.run(task)
    assert isinstance(result, FinalResult)
    assert result.success and result.status is FinalStatus.COMPLETED
    assert result.task_id == task.id and task == before
    assert len(provider.requests) == 1
    state = tree.last_state
    assert state.current_phase is WorkflowPhase.COMPLETED
    assert state.final_result == result
    assert all(getattr(state, field) is not None for field in (
        "triage_result", "orchestration_plan", "delegation_result",
        "execution_result", "manager_review_result", "final_result",
    ))
    events = [event.event_type for event in result.trace.events]
    assert events[0] == "orchestration.started"
    assert events[-2:] == ["root.synthesis.completed", "execution.completed"]
    for partial in (state.orchestration_plan, state.delegation_result,
                    state.execution_result, state.manager_review_result):
        count = partial.trace.event_count
        assert result.trace.events[:count] == partial.trace.events
    assert state.trace.events == result.trace.events


def test_provider_powered_sdk_with_mixed_providers() -> None:
    a = MockProvider(response_content=
        '{"objective":"Work","required_capabilities":["manage"],"decision":"pass","feedback":"ok"}')
    b = MockProvider(response_content=
        '{"subtasks":[{"objective":"Work","required_capabilities":["work"]}],"decision":"pass","feedback":"ok"}')
    tree, worker = ready(
        triage=ProviderTaskTriage(a), decomposer=ProviderTaskDecomposer(b),
        manager_reviewer=ProviderManagerReviewer(a), final_reviewer=ProviderFinalReviewer(b),
    )
    assert tree.run(Task(objective="Work")).status is FinalStatus.COMPLETED
    assert [r.metadata["strategy"] for r in a.requests] == ["triage", "manager_review"]
    assert [r.metadata["strategy"] for r in b.requests] == ["decomposition", "final_review"]
    assert len(worker.requests) == 1


@pytest.mark.parametrize("manager_count,specialist_count", [(2, 1), (1, 3), (2, 2)])
def test_multiple_managers_and_specialists(manager_count: int, specialist_count: int) -> None:
    tree = framework()
    provider = MockProvider()
    tree.register_provider(provider)
    for index in range(manager_count):
        manager = ManagerAgent(name=f"Manager {index}", capabilities=("manage",))
        tree.register_manager(manager)
        for worker in range(specialist_count):
            specialist = SpecialistAgent(name=f"Worker {worker}", capabilities=("work",))
            tree.register_specialist(manager, specialist)
            tree.bind_provider(specialist, provider)
    result = tree.run(Task(objective="Work"))
    assert result.status is FinalStatus.COMPLETED
    assert len(result.manager_results) == manager_count
    assert len(provider.requests) == manager_count * specialist_count


def test_duplicate_manager_does_not_change_registry() -> None:
    tree = framework()
    manager, specialist = register_pair(tree)
    for duplicate in (manager, ManagerAgent(name="Alias", id=manager.id)):
        with pytest.raises(ValueError):
            tree.register_manager(duplicate)
    assert tree.managers == (manager,) and tree.specialists == (specialist,)


def test_duplicate_specialist_and_cross_manager_ownership_are_rejected() -> None:
    tree = framework()
    manager, specialist = register_pair(tree)
    second = ManagerAgent(name="Other")
    tree.register_manager(second)
    for owner, duplicate in (
        (manager, specialist), (second, specialist),
        (manager, SpecialistAgent(name="Alias", id=specialist.id)),
    ):
        with pytest.raises(ValueError):
            tree.register_specialist(owner, duplicate)
    assert manager.specialists == (specialist,)
    assert second.specialists == () and tree.specialists == (specialist,)


def test_manager_must_be_registered_and_exact_object() -> None:
    tree = framework()
    manager = ManagerAgent(name="Manager")
    specialist = SpecialistAgent(name="Worker")
    with pytest.raises(KeyError):
        tree.register_specialist(manager, specialist)
    assert manager.specialists == () and tree.specialists == ()
    tree.register_manager(manager)
    with pytest.raises(ValueError):
        tree.register_specialist(ManagerAgent(name="Alias", id=manager.id), specialist)


def test_preowned_specialists_are_registered_as_one_batch() -> None:
    tree = framework()
    manager = ManagerAgent(name="Manager")
    one, two = SpecialistAgent(name="One"), SpecialistAgent(name="Two")
    manager.register_specialist(one)
    manager.register_specialist(two)
    tree.register_manager(manager)
    assert tree.specialists == (one, two)
    three = SpecialistAgent(name="Three")
    manager.register_specialist(three)
    tree.register_specialist(manager, three)
    assert tree.specialists == manager.specialists == (one, two, three)


def test_batch_preflight_rejects_identity_collisions_without_partial_state() -> None:
    tree = framework()
    existing, _ = register_pair(tree)
    manager = ManagerAgent(name="New manager")
    first = SpecialistAgent(name="First")
    conflict = SpecialistAgent(name="Conflict", id=existing.id)
    manager.register_specialist(first)
    manager.register_specialist(conflict)
    before = tree.managers, tree.specialists
    with pytest.raises(ValueError):
        tree.register_manager(manager)
    assert (tree.managers, tree.specialists) == before
    assert manager.specialists == (first, conflict)


def test_invalid_capabilities_do_not_add_membership() -> None:
    tree = framework()
    manager = ManagerAgent(name="Manager")
    tree.register_manager(manager)
    invalid = SpecialistAgent(name="Invalid", capabilities=(" ",))
    with pytest.raises(ValueError):
        tree.register_specialist(manager, invalid)
    assert manager.specialists == tree.specialists == ()


def test_membership_failure_rolls_back_registry_and_membership() -> None:
    class RejectingManager(ManagerAgent):
        def register_specialist(self, specialist: SpecialistAgent) -> None:
            super().register_specialist(specialist)
            raise RuntimeError("Rejected")

    tree = framework()
    manager = RejectingManager(name="Manager")
    tree.register_manager(manager)
    with pytest.raises(RuntimeError):
        tree.register_specialist(manager, SpecialistAgent(name="Worker"))
    assert tree.specialists == manager.specialists == ()


def test_registry_failure_rolls_back_manager_batch() -> None:
    class RejectingRegistry(CapabilityRegistry):
        def register(self, agent: Any) -> None:
            if agent.name == "Reject":
                raise RuntimeError("Rejected by registry")
            super().register(agent)

    registry = RejectingRegistry()
    tree = framework(capability_registry=registry)
    manager = ManagerAgent(name="Manager")
    one = SpecialistAgent(name="One")
    rejected = SpecialistAgent(name="Reject")
    manager.register_specialist(one)
    manager.register_specialist(rejected)
    with pytest.raises(RuntimeError):
        tree.register_manager(manager)
    assert registry.agents == ()
    assert manager.specialists == (one, rejected)


def test_inconsistent_injected_agent_registry_is_rejected() -> None:
    registry = CapabilityRegistry()
    registry.register(SpecialistAgent(name="Orphan"))
    with pytest.raises(ValueError, match="no manager"):
        framework(capability_registry=registry)


def test_reentrant_run_and_registration_are_rejected_without_mutation() -> None:
    class ReentrantTriage(RuleBasedTaskTriage):
        def triage(self, task: Task) -> Any:
            with pytest.raises(RuntimeError, match="already running"):
                tree.run(Task(objective="Nested"))
            with pytest.raises(RuntimeError, match="already running"):
                tree.register_manager(ManagerAgent(name="Late"))
            return super().triage(task)

    tree, _ = ready(triage=ReentrantTriage({}, fallback_capabilities=("manage",)))
    assert tree.run(Task(objective="Work")).success
    assert len(tree.managers) == 1


def test_run_after_configuration_failure_can_succeed() -> None:
    tree = framework()
    _, specialist = register_pair(tree)
    with pytest.raises(KeyError):
        tree.run(Task(objective="First"))
    failed = tree.last_state
    provider = MockProvider()
    tree.register_provider(provider)
    tree.bind_provider(specialist, provider)
    assert tree.run(Task(objective="Second")).success
    assert failed.current_phase is WorkflowPhase.FAILED
    assert tree.last_state.current_phase is WorkflowPhase.COMPLETED


def test_root_identity_is_reserved() -> None:
    tree = framework()
    with pytest.raises(ValueError):
        tree.register_manager(ManagerAgent(name="Conflict", id=tree.root_agent.id))
    assert tree.managers == ()


def test_binding_validation_rebinding_and_snapshots() -> None:
    tree, first = ready()
    specialist = tree.specialists[0]
    snapshot = tree.provider_bindings
    second = MockProvider(ProviderConfig("second"))
    tree.register_provider(second)
    tree.bind_provider(specialist, " SECOND ")
    assert snapshot[specialist.id] == first.name
    assert tree.provider_bindings[specialist.id] == second.name
    with pytest.raises(TypeError):
        snapshot[specialist.id] = "bad"
    for value in ("missing", MockProvider(ProviderConfig("missing"))):
        with pytest.raises(KeyError):
            tree.bind_provider(specialist, value)
    with pytest.raises(ValueError):
        tree.bind_provider(specialist, MockProvider(ProviderConfig("second")))
    with pytest.raises(KeyError):
        tree.bind_provider(SpecialistAgent(name="Unknown"), second)
    assert tree.provider_bindings[specialist.id] == second.name
    with pytest.raises(ValueError):
        tree.register_provider(second)


def test_tool_binding_handles_function_and_mcp_tools_equally() -> None:
    tree, _ = ready()
    specialist = tree.specialists[0]
    function = FunctionTool(name="echo", function=lambda value: value)
    definition = MCPToolDefinition(name="remote")
    client = MockMCPClient(tools=(definition,))
    mcp = MCPTool(client=client, definition=definition)
    for tool in (function, mcp):
        tree.register_tool(tool)
        tree.bind_tool(specialist, tool)
    assert tree.tools == (function, mcp)
    assert tree.tool_bindings[specialist.id] == (function.id, mcp.id)
    with pytest.raises(ValueError):
        tree.bind_tool(specialist, function.id)
    with pytest.raises(KeyError):
        tree.bind_tool(specialist, "unknown")
    with pytest.raises(TypeError):
        tree.tool_bindings[specialist.id] = ()
    assert tree.run(Task(objective="Work")).success
    assert client.calls == ()  # No automatic tool execution.


def test_registration_snapshots_have_stable_membership() -> None:
    tree = framework()
    snapshots = tree.managers, tree.specialists, tree.providers, tree.tools
    register_pair(tree)
    tree.register_provider(MockProvider())
    assert snapshots == ((), (), (), ())
    assert all(isinstance(snapshot, tuple) for snapshot in snapshots)


@pytest.mark.parametrize("missing", ["manager", "specialist", "subtasks"])
def test_no_work_returns_explicit_framework_failure(missing: str) -> None:
    tree = framework(decomposer=StaticTaskDecomposer(()) if missing == "subtasks" else
                     StaticTaskDecomposer((SubtaskTemplate("Work", ("work",)),)))
    if missing != "manager":
        tree.register_manager(ManagerAgent(name="Manager", capabilities=("manage",)))
    result = tree.run(Task(objective="Work"))
    assert result.status is FinalStatus.FAILED and not result.success
    assert result.final_reviews == ()
    assert result.final_review.metadata["review_performed"] is False
    assert tree.last_state.current_phase is WorkflowPhase.FAILED
    assert tree.last_state.final_result == result
    assert result.trace.events[-1].event_type == "execution.failed"


def test_missing_provider_binding_is_exception_before_any_execution() -> None:
    tree, provider = ready()
    tree.register_specialist(tree.managers[0], SpecialistAgent(name="Unbound", capabilities=("work",)))
    with pytest.raises(KeyError):
        tree.run(Task(objective="Work"))
    assert provider.requests == ()
    assert tree.last_state.failed_phase is WorkflowPhase.EXECUTION
    assert tree.last_state.delegation_result is not None


class BrokenProvider(MockProvider):
    """An offline generation failure shared by execution and decision tests."""

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        raise ProviderRuntimeError("Offline failure")


def test_specialist_provider_failure_remains_structured_and_reviewable() -> None:
    tree = framework(manager_reviewer=StaticManagerReviewer(ReviewDecision.FAIL),
                     final_reviewer=StaticFinalReviewer(ReviewDecision.FAIL))
    _, specialist = register_pair(tree)
    provider = BrokenProvider()
    tree.register_provider(provider)
    tree.bind_provider(specialist, provider)
    result = tree.run(Task(objective="Work"))
    assert not result.success and result.status is FinalStatus.FAILED
    execution = tree.last_state.execution_result.manager_executions[0].specialist_executions[0]
    assert execution.status is ExecutionStatus.FAILED
    assert "ProviderRuntimeError" in execution.agent_result.error
    assert result.manager_results[0].subtask_outcomes[0].decision is ReviewDecision.FAIL


@pytest.mark.parametrize("strategy", ["triage", "decomposer", "manager_reviewer", "final_reviewer"])
@pytest.mark.parametrize("runtime", [False, True])
def test_decision_errors_propagate_and_retain_failed_state(strategy: str, runtime: bool) -> None:
    provider = BrokenProvider() if runtime else MockProvider(response_content="bad json")
    cls = dict(triage=ProviderTaskTriage, decomposer=ProviderTaskDecomposer,
               manager_reviewer=ProviderManagerReviewer, final_reviewer=ProviderFinalReviewer)[strategy]
    tree, _ = ready(**{strategy: cls(provider)})
    with pytest.raises(ProviderRuntimeError if runtime else DecisionParseError):
        tree.run(Task(objective="Work"))
    state = tree.last_state
    assert state.current_phase is WorkflowPhase.FAILED
    expected = dict(triage=WorkflowPhase.TRIAGE, decomposer=WorkflowPhase.DELEGATION,
                    manager_reviewer=WorkflowPhase.MANAGER_REVIEW, final_reviewer=WorkflowPhase.FINAL_REVIEW)
    assert state.failed_phase is expected[strategy]
    assert state.final_result is None


@pytest.mark.parametrize("layer", ["manager", "final"])
def test_review_revisions_reuse_engine(layer: str) -> None:
    outcomes = ((ReviewDecision.REVISE, "Improve"), (ReviewDecision.PASS, "Accepted"))
    options = {"manager_reviewer": StaticManagerReviewer(outcomes=outcomes)} if layer == "manager" else {
        "final_reviewer": StaticFinalReviewer(outcomes=outcomes)}
    tree, provider = ready(**options)
    result = tree.run(Task(objective="Work"))
    assert result.status is FinalStatus.COMPLETED
    if layer == "manager":
        assert len(provider.requests) == 2
        assert result.manager_results[0].subtask_outcomes[0].revision_count == 1
    else:
        assert result.revision_count == 1 and len(result.final_reviews) == 2


@pytest.mark.parametrize("layer", ["manager", "final"])
def test_revision_limits_are_respected(layer: str) -> None:
    options = {"manager_reviewer": StaticManagerReviewer(ReviewDecision.REVISE),
               "final_reviewer": StaticFinalReviewer(ReviewDecision.FAIL)} if layer == "manager" else {
        "final_reviewer": StaticFinalReviewer(ReviewDecision.REVISE)}
    tree, provider = ready(config=AgentTreeConfig(max_manager_revisions=0, max_final_revisions=0), **options)
    result = tree.run(Task(objective="Work"))
    assert not result.success and len(provider.requests) == 1
    if layer == "manager":
        assert result.manager_results[0].subtask_outcomes[0].status.value == "revision_limit_reached"
    else:
        assert result.status is FinalStatus.FINAL_REVISION_LIMIT_REACHED
    assert tree.last_state.current_phase is WorkflowPhase.FAILED


def test_final_fail_is_returned_without_exception() -> None:
    tree, _ = ready(final_reviewer=StaticFinalReviewer(ReviewDecision.FAIL, "Incomplete"))
    result = tree.run(Task(objective="Work"))
    assert not result.success and result.final_review.feedback == "Incomplete"
    assert tree.last_state.failed_phase is WorkflowPhase.FINAL_REVIEW


def test_last_state_is_isolated_and_replaced_for_each_run() -> None:
    tree, _ = ready()
    result = tree.run(Task(objective="First"))
    first = tree.last_state
    original = deepcopy(first.final_result.content)
    result.content.clear()
    first.final_result.content.clear()
    assert tree.last_state.final_result.content == original
    with pytest.raises(TypeError):
        first.trace.append(ExecutionEvent(task_id=first.task_id, event_type="extra"))
    tree.run(Task(objective="Second"))
    assert tree.last_state.task_id != first.task_id
    assert first.current_phase is WorkflowPhase.COMPLETED


def test_injected_registries_and_custom_executor() -> None:
    class LocalExecutor(BaseSpecialistExecutor):
        def execute(self, task: Task, subtask: Subtask, specialist: SpecialistAgent) -> AgentResult:
            task.metadata["mutated"] = True
            return AgentResult(agent_id=specialist.id, success=True, output="local")

    agents, providers, tools, bindings = CapabilityRegistry(), ProviderRegistry(), ToolRegistry(), ToolBindingRegistry()
    tree = framework(capability_registry=agents, provider_registry=providers,
                     tool_registry=tools, tool_bindings=bindings, executor=LocalExecutor())
    manager, specialist = register_pair(tree)
    assert agents.agents == (manager, specialist)
    provider = MockProvider()
    tree.register_provider(provider)
    assert providers.providers == (provider,)
    with pytest.raises(ValueError, match="own provider bindings"):
        tree.bind_provider(specialist, provider)
    task = Task(objective="Work")
    assert tree.run(task).success
    assert task.metadata == {} and provider.requests == ()


def test_external_membership_drift_is_rejected_before_generation() -> None:
    tree, provider = ready()
    tree.managers[0].remove_specialist(tree.specialists[0].id)
    with pytest.raises(ValueError, match="no manager"):
        tree.run(Task(objective="Work"))
    assert provider.requests == ()


@pytest.mark.parametrize("values", [
    {"manager_match_all": 1}, {"specialist_match_all": "yes"},
    {"max_manager_revisions": -1}, {"max_final_revisions": True},
    {"max_final_revisions": 1.5},
])
def test_config_validation(values: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        AgentTreeConfig(**values)


def test_config_is_immutable_and_strategies_are_explicit() -> None:
    config = AgentTreeConfig()
    with pytest.raises(FrozenInstanceError):
        config.max_manager_revisions = 5
    with pytest.raises(TypeError):
        AgentTree()
    with pytest.raises(TypeError):
        framework(triage=None)
    tree = framework()
    with pytest.raises(TypeError):
        tree.run("work")


@pytest.mark.parametrize("field", ["manager_match_all", "specialist_match_all"])
def test_match_all_configuration_reaches_existing_discovery(field: str) -> None:
    options = {"triage": RuleBasedTaskTriage({}, fallback_capabilities=("manage", "extra"))} if field == "manager_match_all" else {
        "decomposer": StaticTaskDecomposer((SubtaskTemplate("Work", ("work", "extra")),))}
    loose, _ = ready(**options)
    strict, _ = ready(config=AgentTreeConfig(**{field: True}), **options)
    assert loose.run(Task(objective="Work")).success
    assert not strict.run(Task(objective="Work")).success
