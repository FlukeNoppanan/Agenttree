"""Provider-backed specialist execution and aggregation tests."""

from types import MappingProxyType

import pytest

from agenttree.agents import ManagerAgent, SpecialistAgent
from agenttree.core import (
    BaseSpecialistExecutor,
    ProviderSpecialistExecutor,
    RuleBasedTaskTriage,
    StaticTaskDecomposer,
)
from agenttree.models import AgentResult, Subtask, SubtaskTemplate, Task, TaskContext
from agenttree.orchestration import (
    DelegationStatus,
    ExecutionStatus,
    OrchestrationEngine,
    OrchestrationEventType,
)
from agenttree.providers import (
    BaseProvider,
    MockProvider,
    ProviderConfig,
    ProviderRegistry,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
)
from agenttree.registry import CapabilityRegistry


class FailingProvider(BaseProvider):
    """Raise a runtime provider failure for boundary tests."""

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        raise RuntimeError("provider unavailable")


class InvalidResponseProvider(BaseProvider):
    """Violate the provider response contract for validation tests."""

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return object()  # type: ignore[return-value]


def _pipeline(
    specialists: tuple[SpecialistAgent, ...],
    templates: tuple[SubtaskTemplate, ...],
) -> tuple[OrchestrationEngine, Task, object, CapabilityRegistry, ManagerAgent]:
    registry = CapabilityRegistry()
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    registry.register(manager)
    for specialist in specialists:
        manager.register_specialist(specialist)
        registry.register(specialist)
    engine = OrchestrationEngine(
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        registry=registry,
    )
    task = Task(
        objective="Parent objective",
        context=TaskContext(data={"inputs": [{"value": 7}]}),
        metadata={"request": {"id": "request-1"}},
    )
    planning = engine.orchestrate(task)
    delegation = engine.delegate(
        planning,
        task,
        decomposer=StaticTaskDecomposer(templates),
    )
    return engine, task, delegation, registry, manager


def test_base_specialist_executor_is_an_abstract_contract() -> None:
    with pytest.raises(TypeError):
        BaseSpecialistExecutor()  # type: ignore[abstract]

    class IncompleteExecutor(BaseSpecialistExecutor):
        pass

    with pytest.raises(TypeError):
        IncompleteExecutor()


def test_provider_executor_construction_and_binding_snapshot() -> None:
    providers = ProviderRegistry()
    provider = MockProvider(ProviderConfig(provider_name="fixture"))
    providers.register(provider)
    bindings = {"specialist-1": "fixture"}
    executor = ProviderSpecialistExecutor(
        provider_registry=providers, provider_bindings=bindings,
    )
    snapshot = executor.provider_bindings
    bindings["specialist-2"] = "other"
    assert executor.provider_registry is providers
    assert snapshot == {"specialist-1": "fixture"}
    assert isinstance(snapshot, MappingProxyType)
    with pytest.raises(TypeError):
        snapshot["new"] = "fixture"  # type: ignore[index]


def test_provider_resolution_uses_specialist_id_binding() -> None:
    providers = ProviderRegistry()
    provider = MockProvider(ProviderConfig(provider_name="fixture"))
    providers.register(provider)
    specialist = SpecialistAgent(name="Worker", id="specialist-1")
    executor = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={specialist.id: " FIXTURE "},
    )
    assert executor.resolve_provider(specialist) is provider


def test_missing_provider_binding_and_registration_surface() -> None:
    providers = ProviderRegistry()
    specialist = SpecialistAgent(name="Worker")
    missing_binding = ProviderSpecialistExecutor(
        provider_registry=providers, provider_bindings={},
    )
    with pytest.raises(KeyError, match="No provider binding"):
        missing_binding.resolve_provider(specialist)

    missing_provider = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={specialist.id: "absent"},
    )
    with pytest.raises(KeyError):
        missing_provider.resolve_provider(specialist)


def test_provider_request_contains_generic_framework_context() -> None:
    provider = MockProvider(
        ProviderConfig(provider_name="fixture", model="model-a"),
        response_content="Done",
    )
    providers = ProviderRegistry()
    providers.register(provider)
    specialist = SpecialistAgent(
        name="Worker", description="Produce the configured output",
        capabilities=("work",), metadata={"settings": {"mode": "fixed"}},
    )
    task = Task(
        objective="Parent objective",
        context=TaskContext(data={"input": {"value": 3}}),
        metadata={"source": {"id": 1}},
    )
    subtask = Subtask(
        parent_task_id=task.id, manager_id="manager-1",
        objective="Subtask objective", required_capabilities=("work",),
        metadata={"sequence": 1},
    )
    executor = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={specialist.id: "fixture"},
    )
    result = executor.execute(task, subtask, specialist)
    request = provider.requests[0]
    assert request.prompt == "Subtask objective"
    assert request.system_prompt == "Produce the configured output"
    assert request.context["task"] == {
        "id": task.id,
        "objective": "Parent objective",
        "context": {"input": {"value": 3}},
        "metadata": {"source": {"id": 1}},
    }
    assert request.context["subtask"]["id"] == subtask.id
    assert request.context["specialist"]["id"] == specialist.id
    assert request.metadata == {
        "task_id": task.id,
        "subtask_id": subtask.id,
        "specialist_id": specialist.id,
    }
    assert result.success is True


def test_provider_request_context_does_not_expose_input_metadata() -> None:
    provider = MockProvider()
    providers = ProviderRegistry()
    providers.register(provider)
    specialist = SpecialistAgent(
        name="Worker", metadata={"nested": {"value": 1}},
    )
    task = Task(
        objective="Parent",
        context=TaskContext(data={"nested": {"value": 1}}),
    )
    subtask = Subtask(
        parent_task_id=task.id, manager_id="manager",
        objective="Work", metadata={"nested": {"value": 1}},
    )
    executor = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={specialist.id: provider.name},
    )
    executor.execute(task, subtask, specialist)
    request = provider.requests[0]
    request.context["task"]["context"]["nested"]["value"] = 2
    request.context["subtask"]["metadata"]["nested"]["value"] = 2
    request.context["specialist"]["metadata"]["nested"]["value"] = 2
    assert task.context.data["nested"]["value"] == 1
    assert subtask.metadata["nested"]["value"] == 1
    assert specialist.metadata["nested"]["value"] == 1


def test_provider_response_normalizes_to_successful_agent_result() -> None:
    usage = ProviderUsage(input_tokens=2, output_tokens=3, total_tokens=5)
    response = ProviderResponse(
        content="Normalized output", provider="fixture", model="model-a",
        usage=usage, metadata={"finish": "complete"},
    )
    provider = MockProvider(
        ProviderConfig(provider_name="fixture"), response=response,
    )
    providers = ProviderRegistry()
    providers.register(provider)
    specialist = SpecialistAgent(name="Worker")
    task = Task(objective="Parent")
    subtask = Subtask(
        parent_task_id=task.id, manager_id="manager", objective="Work",
    )
    result = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={specialist.id: provider.name},
    ).execute(task, subtask, specialist)
    assert result == AgentResult(
        agent_id=specialist.id,
        success=True,
        output="Normalized output",
        metadata={
            "provider": "fixture",
            "model": "model-a",
            "usage": usage,
            "provider_metadata": {"finish": "complete"},
        },
    )


def test_provider_failure_becomes_failed_agent_result() -> None:
    provider = FailingProvider(ProviderConfig(provider_name="failing"))
    providers = ProviderRegistry()
    providers.register(provider)
    specialist = SpecialistAgent(name="Worker")
    task = Task(objective="Parent")
    subtask = Subtask(
        parent_task_id=task.id, manager_id="manager", objective="Work",
    )
    result = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={specialist.id: provider.name},
    ).execute(task, subtask, specialist)
    assert result.success is False
    assert result.output is None
    assert result.error == "RuntimeError: provider call failed"
    assert result.metadata == {"provider": "failing", "error_type": "RuntimeError"}


def test_invalid_provider_response_is_a_contract_error() -> None:
    provider = InvalidResponseProvider(ProviderConfig(provider_name="invalid"))
    providers = ProviderRegistry()
    providers.register(provider)
    specialist = SpecialistAgent(name="Worker")
    task = Task(objective="Parent")
    subtask = Subtask(
        parent_task_id=task.id, manager_id="manager", objective="Work",
    )
    with pytest.raises(TypeError, match="ProviderResponse"):
        ProviderSpecialistExecutor(
            provider_registry=providers,
            provider_bindings={specialist.id: provider.name},
        ).execute(task, subtask, specialist)


def test_multiple_specialists_execute_sequentially_in_assignment_order() -> None:
    first = SpecialistAgent(name="First", capabilities=("work",))
    second = SpecialistAgent(name="Second", capabilities=("work",))
    engine, task, delegation, _, _ = _pipeline(
        (first, second),
        (SubtaskTemplate(objective="Work", required_capabilities=("work",)),),
    )
    provider = MockProvider(response_content="Done")
    providers = ProviderRegistry()
    providers.register(provider)
    executor = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={first.id: provider.name, second.id: provider.name},
    )
    result = engine.execute(delegation, task, executor=executor)
    executions = result.manager_executions[0].specialist_executions
    assert tuple(item.specialist_id for item in executions) == (first.id, second.id)
    assert tuple(request.metadata["specialist_id"] for request in provider.requests) == (
        first.id, second.id,
    )
    assert result.status is ExecutionStatus.COMPLETED


def test_provider_failure_does_not_erase_or_stop_other_results() -> None:
    first = SpecialistAgent(name="First", capabilities=("work",))
    second = SpecialistAgent(name="Second", capabilities=("work",))
    engine, task, delegation, _, _ = _pipeline(
        (first, second),
        (SubtaskTemplate(objective="Work", required_capabilities=("work",)),),
    )
    success = MockProvider(ProviderConfig(provider_name="success"))
    failure = FailingProvider(ProviderConfig(provider_name="failure"))
    providers = ProviderRegistry()
    providers.register(success)
    providers.register(failure)
    executor = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={first.id: "failure", second.id: "success"},
    )
    result = engine.execute(delegation, task, executor=executor)
    executions = result.manager_executions[0].specialist_executions
    assert tuple(item.status for item in executions) == (
        ExecutionStatus.FAILED, ExecutionStatus.COMPLETED,
    )
    assert executions[0].agent_result is not None
    assert executions[0].agent_result.error == "RuntimeError: provider call failed"
    assert executions[1].agent_result is not None
    assert executions[1].agent_result.success is True
    assert result.manager_executions[0].status is ExecutionStatus.PARTIAL
    assert result.status is ExecutionStatus.PARTIAL


def test_no_specialist_assignment_is_skipped_without_fake_result() -> None:
    engine, task, delegation, _, _ = _pipeline(
        (),
        (SubtaskTemplate(objective="Work", required_capabilities=("missing",)),),
    )
    assert delegation.manager_delegations[0].assignments[0].status is (
        DelegationStatus.NO_SPECIALIST
    )
    result = engine.execute(
        delegation,
        task,
        executor=ProviderSpecialistExecutor(
            provider_registry=ProviderRegistry(), provider_bindings={},
        ),
    )
    execution = result.manager_executions[0].specialist_executions[0]
    assert execution.status is ExecutionStatus.SKIPPED
    assert execution.specialist_id is None
    assert execution.agent_result is None
    assert result.status is ExecutionStatus.SKIPPED


def test_empty_delegation_is_skipped() -> None:
    registry = CapabilityRegistry()
    engine = OrchestrationEngine(
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("missing",)),
        registry=registry,
    )
    task = Task(objective="Parent")
    delegation = engine.delegate(
        engine.orchestrate(task), task, decomposer=StaticTaskDecomposer(()),
    )
    result = engine.execute(
        delegation,
        task,
        executor=ProviderSpecialistExecutor(
            provider_registry=ProviderRegistry(), provider_bindings={},
        ),
    )
    assert result.manager_executions == ()
    assert result.status is ExecutionStatus.SKIPPED


def test_execution_does_not_mutate_task_delegation_or_registry() -> None:
    specialist = SpecialistAgent(name="Worker", capabilities=("work",))
    engine, task, delegation, registry, manager = _pipeline(
        (specialist,),
        (SubtaskTemplate(objective="Work", required_capabilities=("work",)),),
    )
    provider = MockProvider()
    providers = ProviderRegistry()
    providers.register(provider)
    executor = ProviderSpecialistExecutor(
        provider_registry=providers,
        provider_bindings={specialist.id: provider.name},
    )
    task_before = (
        task.id, task.objective, task.context.data.copy(), task.metadata.copy(),
        task.status, task.created_at,
    )
    delegation_before = (
        delegation.manager_delegations,
        delegation.status,
        delegation.trace.events,
        delegation.metadata.copy(),
    )
    registry_before = (
        registry.agents, registry.capabilities, registry.capability_index,
        manager.specialists,
    )
    result = engine.execute(delegation, task, executor=executor)
    assert (
        task.id, task.objective, task.context.data, task.metadata,
        task.status, task.created_at,
    ) == task_before
    assert (
        delegation.manager_delegations,
        delegation.status,
        delegation.trace.events,
        delegation.metadata,
    ) == delegation_before
    assert (
        registry.agents, registry.capabilities, registry.capability_index,
        manager.specialists,
    ) == registry_before
    assert result.trace is not delegation.trace
    assert result.trace.events[:len(delegation.trace.events)] == delegation.trace.events


def test_execution_trace_orders_success_failure_and_completion_events() -> None:
    first = SpecialistAgent(name="First", capabilities=("work",))
    second = SpecialistAgent(name="Second", capabilities=("work",))
    engine, task, delegation, _, _ = _pipeline(
        (first, second),
        (SubtaskTemplate(objective="Work", required_capabilities=("work",)),),
    )
    success = MockProvider(ProviderConfig(provider_name="success"))
    failure = FailingProvider(ProviderConfig(provider_name="failure"))
    providers = ProviderRegistry()
    providers.register(success)
    providers.register(failure)
    before_count = len(delegation.trace.events)
    result = engine.execute(
        delegation,
        task,
        executor=ProviderSpecialistExecutor(
            provider_registry=providers,
            provider_bindings={first.id: "success", second.id: "failure"},
        ),
    )
    assert tuple(event.event_type for event in result.trace.events[before_count:]) == (
        OrchestrationEventType.EXECUTION_STARTED.value,
        OrchestrationEventType.SPECIALIST_EXECUTION_STARTED.value,
        OrchestrationEventType.SPECIALIST_EXECUTION_COMPLETED.value,
        OrchestrationEventType.SPECIALIST_EXECUTION_STARTED.value,
        OrchestrationEventType.SPECIALIST_EXECUTION_FAILED.value,
        OrchestrationEventType.EXECUTION_COMPLETED.value,
    )
    failure_event = result.trace.events[-2]
    assert failure_event.actor_id == second.id
    assert failure_event.metadata["error"] == "RuntimeError: provider call failed"


def test_skipped_assignment_trace_event() -> None:
    engine, task, delegation, _, _ = _pipeline(
        (),
        (SubtaskTemplate(objective="Work", required_capabilities=("missing",)),),
    )
    before_count = len(delegation.trace.events)
    result = engine.execute(
        delegation,
        task,
        executor=ProviderSpecialistExecutor(
            provider_registry=ProviderRegistry(), provider_bindings={},
        ),
    )
    assert tuple(event.event_type for event in result.trace.events[before_count:]) == (
        OrchestrationEventType.EXECUTION_STARTED.value,
        OrchestrationEventType.ASSIGNMENT_SKIPPED.value,
        OrchestrationEventType.EXECUTION_COMPLETED.value,
    )


def test_execute_validates_executor_result_contract() -> None:
    class WrongResultExecutor(BaseSpecialistExecutor):
        def execute(
            self, task: Task, subtask: Subtask, specialist: SpecialistAgent,
        ) -> AgentResult:
            return AgentResult(agent_id="other", success=True)

    specialist = SpecialistAgent(name="Worker", capabilities=("work",))
    engine, task, delegation, _, _ = _pipeline(
        (specialist,),
        (SubtaskTemplate(objective="Work", required_capabilities=("work",)),),
    )
    with pytest.raises(ValueError, match="AgentResult agent_id"):
        engine.execute(delegation, task, executor=WrongResultExecutor())


def test_step_8_delegation_output_executes_without_conversion() -> None:
    specialist = SpecialistAgent(name="Worker", capabilities=("work",))
    engine, task, delegation, _, _ = _pipeline(
        (specialist,),
        (SubtaskTemplate(objective="Work", required_capabilities=("work",)),),
    )
    provider = MockProvider(response_content="Result")
    providers = ProviderRegistry()
    providers.register(provider)
    result = engine.execute(
        delegation,
        task,
        executor=ProviderSpecialistExecutor(
            provider_registry=providers,
            provider_bindings={specialist.id: provider.name},
        ),
    )
    execution = result.manager_executions[0].specialist_executions[0]
    assert execution.agent_result is not None
    assert execution.agent_result.output == "Result"
