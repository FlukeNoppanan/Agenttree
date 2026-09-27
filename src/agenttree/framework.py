"""High-level composition of the existing synchronous AgentTree workflow."""

from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType
from typing import Callable, Mapping

from agenttree.agents import BaseAgent, ManagerAgent, RootAgent, SpecialistAgent
from agenttree.config import AgentTreeConfig
from agenttree.core import (
    BaseFinalReviewer, BaseManagerReviewer, BaseSpecialistExecutor,
    BaseTaskDecomposer, BaseTaskTriage, ProviderSpecialistExecutor,
    WorkflowStateManager, BaseRootPlanner, BaseRootSynthesizer,
    ExtractiveRootSynthesizer, ProviderRootPlanner, ProviderRootSynthesizer,
)
from agenttree.core.usage import UsageCollector, _active_usage
from agenttree.core.provider_routing import _active_provider_bindings, _active_root_id
from agenttree.models import (
    Task, WorkflowPhase, WorkflowState, ExecutionEvent,
    ManagerMessage,
)
from agenttree.tracing import ExecutionEventType
from agenttree.orchestration import FinalResult, OrchestrationEngine
from agenttree.orchestration.backends import (
    BaseOrchestrationBackend, SequentialOrchestrationBackend,
    BackendConfigurationError, OrchestrationContext,
)
from agenttree.providers import BaseProvider, ProviderRegistry
from agenttree.registry import CapabilityRegistry
from agenttree.tools import BaseTool, ToolBindingRegistry, ToolRegistry
from agenttree.tools.runtime import ToolSession, _active_tool_session
from agenttree.core.collaboration import ManagerCollaborationSession, _active_collaboration_session
from agenttree.core.artifact_store import ArtifactStore, InMemoryArtifactStore
from agenttree.core.artifacts import ArtifactSession, _active_artifact_session


class AgentTree:
    """Compose explicit strategies into one synchronous ``run(Task)`` API.

    Construction makes no model calls and supplies no decision strategies.
    Registries default to empty framework-owned instances. Injected registries
    and component objects remain shared with the caller; do not mutate them
    during a run. One instance supports sequential runs, not concurrent runs.
    """

    def __init__(
        self, *, root_agent: RootAgent, triage: BaseTaskTriage,
        decomposer: BaseTaskDecomposer, manager_reviewer: BaseManagerReviewer,
        final_reviewer: BaseFinalReviewer, config: AgentTreeConfig | None = None,
        capability_registry: CapabilityRegistry | None = None,
        provider_registry: ProviderRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_bindings: ToolBindingRegistry | None = None,
        executor: BaseSpecialistExecutor | None = None,
        orchestration_backend: BaseOrchestrationBackend | None = None,
        root_planner: BaseRootPlanner | None = None,
        root_synthesizer: BaseRootSynthesizer | None = None,
    ) -> None:
        for name, value, expected in (
            ("root_agent", root_agent, RootAgent),
            ("triage", triage, BaseTaskTriage),
            ("decomposer", decomposer, BaseTaskDecomposer),
            ("manager_reviewer", manager_reviewer, BaseManagerReviewer),
            ("final_reviewer", final_reviewer, BaseFinalReviewer),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be a {expected.__name__}")
        for name, value, expected in (
            ("config", config, AgentTreeConfig),
            ("capability_registry", capability_registry, CapabilityRegistry),
            ("provider_registry", provider_registry, ProviderRegistry),
            ("tool_registry", tool_registry, ToolRegistry),
            ("tool_bindings", tool_bindings, ToolBindingRegistry),
            ("executor", executor, BaseSpecialistExecutor),
            ("orchestration_backend", orchestration_backend, BaseOrchestrationBackend),
            ("root_planner", root_planner, BaseRootPlanner),
            ("root_synthesizer", root_synthesizer, BaseRootSynthesizer),
        ):
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{name} must be a {expected.__name__}")
        self._backend = orchestration_backend if orchestration_backend is not None else SequentialOrchestrationBackend()
        self._root = root_agent
        self._config = config if config is not None else AgentTreeConfig()
        self._agents = capability_registry if capability_registry is not None else CapabilityRegistry()
        self._providers = provider_registry if provider_registry is not None else ProviderRegistry()
        self._tools = tool_registry if tool_registry is not None else ToolRegistry()
        self._tool_bindings = tool_bindings if tool_bindings is not None else ToolBindingRegistry()
        self._provider_bindings: dict[str, str] = {}
        self._model_bindings: dict[str, str] = {}
        self._manager_permissions: dict[str, list[str]] = {}
        self._executor = executor
        self._decomposer = decomposer
        self._manager_reviewer = manager_reviewer
        self._final_reviewer = final_reviewer
        self._root_planner = root_planner
        self._root_synthesizer = root_synthesizer or ExtractiveRootSynthesizer()
        self._engine = OrchestrationEngine(
            triage=triage, registry=self._agents,
            manager_match_all=self._config.manager_match_all,
        )
        self._states = WorkflowStateManager()
        self._last_state: WorkflowState | None = None
        self._last_collaboration_messages: tuple[ManagerMessage, ...] = ()
        self._active_context: OrchestrationContext | None = None
        self._running = False
        self._execution_runtime = None
        self._artifact_store: ArtifactStore = InMemoryArtifactStore()
        self._validate_relationships()

    @property
    def config(self) -> AgentTreeConfig:
        """Return the immutable workflow configuration."""
        return self._config

    def start(self, task: Task, *, timeout: float | None = None,
              runtime=None):
        """Submit to a bounded runtime; supply one for durable storage."""
        from agenttree.core.execution_runtime import ExecutionRuntime
        if runtime is None:
            if self._execution_runtime is None:
                self._execution_runtime = ExecutionRuntime()
            runtime = self._execution_runtime
        if not isinstance(runtime, ExecutionRuntime):
            raise TypeError("runtime must be an ExecutionRuntime")
        return runtime.submit(self, task, timeout=timeout)

    @property
    def execution_runtime(self):
        """Return the lazily created in-memory runtime, if any."""
        return self._execution_runtime

    @property
    def artifact_store(self) -> ArtifactStore:
        """Store for artifacts produced by synchronous runs."""
        return self._artifact_store

    @property
    def root_agent(self) -> RootAgent:
        """Return the configured Root identity."""
        return self._root

    @property
    def managers(self) -> tuple[ManagerAgent, ...]:
        """Return registered manager objects in registration order."""
        return tuple(a for a in self._agents.agents if isinstance(a, ManagerAgent))

    @property
    def specialists(self) -> tuple[SpecialistAgent, ...]:
        """Return registered specialist objects in registration order."""
        return tuple(a for a in self._agents.agents if isinstance(a, SpecialistAgent))

    @property
    def providers(self) -> tuple[BaseProvider, ...]:
        """Return a registration snapshot of provider instances."""
        return self._providers.providers

    @property
    def tools(self) -> tuple[BaseTool, ...]:
        """Return a registration snapshot of function and MCP tools alike."""
        return self._tools.tools

    @property
    def provider_bindings(self) -> Mapping[str, str]:
        """Return a read-only snapshot of agent IDs to provider names."""
        return MappingProxyType(dict(self._provider_bindings))

    @property
    def model_bindings(self) -> Mapping[str, str]:
        """Return configured per-agent model overrides."""
        return MappingProxyType(dict(self._model_bindings))

    @property
    def tool_bindings(self) -> Mapping[str, tuple[str, ...]]:
        """Return a read-only snapshot of agent tool assignments."""
        return self._tool_bindings.bindings

    @property
    def manager_permissions(self) -> Mapping[str, tuple[str, ...]]:
        """Return directed Manager peer permissions in assignment order."""
        return MappingProxyType({key: tuple(value) for key, value in self._manager_permissions.items()})

    def allow_manager_communication(self, sender: ManagerAgent, peer: ManagerAgent) -> None:
        """Permit one directed Manager to Manager coordination relationship."""
        self._ensure_idle()
        self._require_registered(sender, ManagerAgent)
        self._require_registered(peer, ManagerAgent)
        if sender.id == peer.id:
            raise ValueError("A Manager cannot be its own collaboration peer")
        peers = self._manager_permissions.setdefault(sender.id, [])
        if peer.id in peers:
            raise ValueError("Manager communication permission already exists")
        peers.append(peer.id)

    @property
    def last_state(self) -> WorkflowState | None:
        """Return an isolated snapshot of the most recent run, or None.

        Nested phase results are copied on every access. State has a sealed
        trace and no run history or persistence. Exceptions retain the most
        recent completed phase trace; engine-local unfinished events may be
        unavailable when a lower-level method raises.
        """
        if self._active_context is not None:
            return self._active_context.state
        return replace(self._last_state) if self._last_state is not None else None

    @property
    def last_collaboration_messages(self) -> tuple[ManagerMessage, ...]:
        """Return an isolated bounded message snapshot from the most recent run."""
        return deepcopy(self._last_collaboration_messages)

    def _ensure_idle(self) -> None:
        if self._running:
            raise RuntimeError("AgentTree is already running")

    def _require_registered(self, agent: BaseAgent, kind: type[BaseAgent]) -> None:
        if not isinstance(agent, kind):
            raise TypeError(f"Expected {kind.__name__}; received {type(agent).__name__}")
        try:
            registered = self._agents.get(agent.id)
        except KeyError:
            raise KeyError(
                f"{kind.__name__} '{agent.id}' is not registered; register it first",
            ) from None
        if registered is not agent:
            raise ValueError(
                f"Agent ID '{agent.id}' belongs to a different registered object",
            )

    def _register_batch(self, agents: tuple[BaseAgent, ...]) -> None:
        # Preflight with the actual registry contract before mutating live state.
        staged = CapabilityRegistry()
        for existing in self._agents.agents:
            staged.register(existing)
        for agent in agents:
            if agent.id == self._root.id:
                raise ValueError(
                    f"Agent ID '{agent.id}' conflicts with RootAgent '{self._root.id}'",
                )
            staged.register(agent)
        added: list[str] = []
        try:
            for agent in agents:
                self._agents.register(agent)
                added.append(agent.id)
        except Exception:
            for agent_id in reversed(added):
                self._agents.unregister(agent_id)
            raise

    def register_manager(self, manager: ManagerAgent) -> None:
        """Atomically register a manager and its pre-owned specialists.

        Repeated registrations and conflicting IDs are errors. Specialists
        belong to exactly one registered manager in the high-level hierarchy.
        """
        self._ensure_idle()
        if not isinstance(manager, ManagerAgent):
            raise TypeError("manager must be a ManagerAgent")
        owned_ids = {s.id for existing in self.managers for s in existing.specialists}
        if any(s.id in owned_ids for s in manager.specialists):
            conflicts = sorted(s.id for s in manager.specialists if s.id in owned_ids)
            raise ValueError(
                f"Specialist IDs already belong to registered managers: {conflicts}",
            )
        self._register_batch((manager, *manager.specialists))

    def register_specialist(
        self, manager: ManagerAgent, specialist: SpecialistAgent,
    ) -> None:
        """Register a specialist under an already registered manager atomically.

        Existing ownership of this same object is accepted if it has not yet
        been registered. Duplicate complete registrations remain errors.
        """
        self._ensure_idle()
        self._require_registered(manager, ManagerAgent)
        if not isinstance(specialist, SpecialistAgent):
            raise TypeError("specialist must be a SpecialistAgent")
        owned = {item.id: item for item in manager.specialists}
        if specialist.id in owned and owned[specialist.id] is not specialist:
            raise ValueError(
                f"Specialist ID '{specialist.id}' belongs to a different owned object",
            )
        for other in self.managers:
            if other is not manager and any(s.id == specialist.id for s in other.specialists):
                raise ValueError(
                    f"Specialist '{specialist.id}' already belongs to manager '{other.id}'",
                )
        self._register_batch((specialist,))
        if specialist.id not in owned:
            try:
                manager.register_specialist(specialist)
            except Exception:
                if any(item is specialist for item in manager.specialists):
                    manager.remove_specialist(specialist.id)
                self._agents.unregister(specialist.id)
                raise

    def register_provider(self, provider: BaseProvider) -> None:
        """Register one provider instance using existing normalized name rules."""
        self._ensure_idle()
        self._providers.register(provider)

    def register_tool(self, tool: BaseTool) -> None:
        """Register any BaseTool without protocol-specific branching."""
        self._ensure_idle()
        self._tools.register(tool)

    def bind_provider(
        self, specialist: BaseAgent, provider: BaseProvider | str,
        *, model: str | None = None,
    ) -> None:
        """Bind Root, a registered Manager, or a registered Specialist.

        Calling again explicitly replaces the binding. A custom injected
        executor manages its own bindings; this API then raises ValueError.
        """
        self._ensure_idle()
        if isinstance(specialist, RootAgent):
            if specialist is not self._root:
                raise ValueError("Root binding must use this Tree's RootAgent")
        elif isinstance(specialist, ManagerAgent):
            self._require_registered(specialist, ManagerAgent)
        else:
            self._require_registered(specialist, SpecialistAgent)
        if self._executor is not None and isinstance(specialist, SpecialistAgent):
            raise ValueError(
                "Injected executors manage their own provider bindings; configure "
                "the executor instead of calling AgentTree.bind_provider()",
            )
        if not isinstance(provider, (BaseProvider, str)):
            raise TypeError("provider must be a BaseProvider or registered name")
        provider_name = provider if isinstance(provider, str) else provider.name
        try:
            resolved = self._providers.get(provider_name)
        except KeyError:
            raise KeyError(
                f"Provider '{provider_name}' is not registered; call register_provider() first",
            ) from None
        if isinstance(provider, BaseProvider) and resolved is not provider:
            raise ValueError(
                f"Provider name '{provider.name}' belongs to a different registered object",
            )
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise ValueError("model must be nonempty text")
        self._provider_bindings[specialist.id] = resolved.name
        if model is not None:
            self._model_bindings[specialist.id] = model
        else:
            self._model_bindings.pop(specialist.id, None)
        if isinstance(specialist, RootAgent):
            if self._root_planner is None:
                self._root_planner = ProviderRootPlanner(resolved)
            if isinstance(self._root_synthesizer, ExtractiveRootSynthesizer):
                self._root_synthesizer = ProviderRootSynthesizer(resolved)

    def bind_tool(self, specialist: BaseAgent, tool: BaseTool | str) -> None:
        """Assign a registered tool object or ID; duplicate pairs are errors."""
        self._ensure_idle()
        if isinstance(specialist, RootAgent):
            if specialist is not self._root:
                raise ValueError("Root binding must use this Tree's RootAgent")
        elif isinstance(specialist, ManagerAgent):
            self._require_registered(specialist, ManagerAgent)
        else:
            self._require_registered(specialist, SpecialistAgent)
        if not isinstance(tool, (BaseTool, str)):
            raise TypeError("tool must be a BaseTool or registered tool ID")
        tool_id = tool if isinstance(tool, str) else tool.id
        try:
            resolved = self._tools.get(tool_id)
        except KeyError:
            raise KeyError(
                f"Tool '{tool_id}' is not registered; call register_tool() first",
            ) from None
        if isinstance(tool, BaseTool) and resolved is not tool:
            raise ValueError(
                f"Tool ID '{tool.id}' belongs to a different registered object",
            )
        self._tool_bindings.assign(specialist.id, resolved.id)

    def _validate_relationships(self) -> None:
        by_id = {agent.id: agent for agent in self._agents.agents}
        if self._root.id in by_id and by_id[self._root.id] is not self._root:
            raise ValueError("Registered agent conflicts with the Root identity")
        owners: dict[str, str] = {}
        for manager in self.managers:
            for specialist in manager.specialists:
                if by_id.get(specialist.id) is not specialist:
                    raise ValueError("Manager membership and capability registry disagree")
                if specialist.id in owners:
                    raise ValueError("Specialist must belong to exactly one manager")
                owners[specialist.id] = manager.id
        if any(specialist.id not in owners for specialist in self.specialists):
            raise ValueError("Registered specialist has no manager")

    def run(self, task: Task, *,
            _resume_state: WorkflowState | None = None,
            _checkpoint: Callable[[WorkflowState, UsageCollector, ToolSession,
                                   ManagerCollaborationSession], None] | None = None,
            _phase_started: Callable[[str], None] | None = None,
            _control: object | None = None,
            _usage_state: dict | None = None,
            _tool_state: dict | None = None,
            _collaboration_state: dict | None = None,
            _artifact_store: ArtifactStore | None = None,
            _execution_store=None) -> FinalResult:
        """Run the five engine phases on an isolated copy of the canonical Task.

        No managers or assignments return explicitly marked framework failures.
        Execution failures remain reviewable outcomes; reviewers retain their
        existing authority. Configuration, decision-output, and decision-provider
        errors are re-raised after recording a FAILED state. Assigned tools are
        offered to provider-backed agents and invoked only on model request.
        """
        self._ensure_idle()
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        working = deepcopy(task)
        self._last_collaboration_messages = ()
        self._last_state = _resume_state if _resume_state is not None else self._states.create_initial(working)
        if self._last_state.task_id != working.id:
            raise ValueError("Recovery state does not match Task")
        self._running = True
        usage = UsageCollector()
        if _usage_state is not None:
            usage.calls = deepcopy(_usage_state["calls"])
        usage_token = _active_usage.set(usage)
        artifact_session = ArtifactSession(working.id,
            _artifact_store if _artifact_store is not None else self._artifact_store,
            execution_store=_execution_store,
            agent_manager_ids={agent.id: manager.id for manager in self.managers
                               for agent in (manager, *manager.specialists)},
            max_artifacts=self._config.max_artifacts,
            max_artifact_bytes=self._config.max_artifact_bytes,
            max_total_bytes=self._config.max_total_artifact_bytes,
            max_metadata_bytes=self._config.max_artifact_metadata_bytes,
            max_path_length=self._config.max_artifact_path_length,
            max_name_length=self._config.max_artifact_name_length)
        artifact_token = _active_artifact_session.set(artifact_session)
        tool_session = ToolSession(self._tools, self._tool_bindings,
                                   (self._root, *self._agents.agents), self._config, working.id)
        if _tool_state is not None:
            from collections import Counter
            tool_session.calls = Counter(_tool_state["calls"])
            tool_session.metrics = Counter(_tool_state["metrics"])
            tool_session.events = list(deepcopy(_tool_state["events"]))
        tool_token = _active_tool_session.set(tool_session)
        routing_token = None
        root_token = None
        collaboration_token = None
        collaboration_session = None
        control_token = None
        try:
            if _control is not None:
                from agenttree.core.execution_control import _active_execution_control
                control_token = _active_execution_control.set(_control)
            resolved_bindings = {
                agent_id: (self._providers.get(name), self._model_bindings.get(agent_id))
                for agent_id, name in self._provider_bindings.items()
            }
            routing_token = _active_provider_bindings.set(resolved_bindings)
            root_token = _active_root_id.set(self._root.id)
            collaboration_session = ManagerCollaborationSession(
                execution_id=working.id, task_objective=working.objective,
                managers=self.managers, permissions=self.manager_permissions,
                provider_bindings=resolved_bindings, config=self._config,
            )
            collaboration_token = _active_collaboration_session.set(collaboration_session)
            if _collaboration_state is not None:
                collaboration_session.restore(_collaboration_state)
            elif _resume_state is not None and _resume_state.orchestration_plan is not None:
                collaboration_session.activate(_resume_state.orchestration_plan.selected_manager_ids)
            self._validate_relationships()
            executor = self._executor if self._executor is not None else ProviderSpecialistExecutor(
                provider_registry=self._providers, provider_bindings=self._provider_bindings,
                model_bindings=self._model_bindings,
            )
            context = OrchestrationContext(
                task=working, state=self._last_state, engine=self._engine,
                root_agent=self._root, config=self._config, agents=self._agents,
                decomposer=self._decomposer, executor=executor,
                manager_reviewer=self._manager_reviewer,
                final_reviewer=self._final_reviewer,
                root_planner=self._root_planner,
                root_synthesizer=self._root_synthesizer, usage=usage,
                resume_phase=_resume_state.current_phase if _resume_state is not None else None,
                on_phase_completed=(lambda state: _checkpoint(state, usage, tool_session,
                                                                collaboration_session)) if _checkpoint else None,
                on_phase_started=_phase_started,
            )
            self._active_context = context
            try:
                result = self._backend.run(context)
                if not isinstance(result, FinalResult) or result.task_id != working.id:
                    raise BackendConfigurationError("Backend must return this task's FinalResult")
                if context.result is None:
                    raise BackendConfigurationError("Backend did not complete the workflow")
                if tool_session.events or collaboration_session.events:
                    from agenttree.models import ExecutionTrace
                    trace = ExecutionTrace(task_id=working.id)
                    for event in sorted((*result.trace.events, *tool_session.events,
                                         *collaboration_session.events),
                                        key=lambda item: item.timestamp):
                        trace.append(event)
                    metrics = dict(result.metadata)
                    if tool_session.events:
                        metrics["tool_metrics"] = dict(tool_session.metrics)
                    if collaboration_session.events:
                        metrics["collaboration_metrics"] = dict(collaboration_session.metrics)
                    result = replace(result, trace=trace,
                                     metadata=metrics)
                    context._record(final_result=result, trace=trace)
                return result
            finally:
                self._last_state = context.state
                self._active_context = None
        except Exception as error:
            assert self._last_state is not None
            if self._last_state.current_phase not in (WorkflowPhase.COMPLETED, WorkflowPhase.FAILED):
                from agenttree.models import ExecutionTrace
                trace = ExecutionTrace(task_id=working.id)
                for event in sorted((*self._last_state.trace.events, *tool_session.events,
                                     *(collaboration_session.events if collaboration_session else ())),
                                    key=lambda item: item.timestamp):
                    trace.append(event)
                trace.append(ExecutionEvent(task_id=working.id,
                    event_type=ExecutionEventType.RUN_FAILED.value,
                    metadata={"error_type": type(error).__name__}))
                self._last_state = self._states.fail(
                    self._last_state, reason=f"Run raised {type(error).__name__}",
                    trace=trace,
                )
            raise
        finally:
            _active_artifact_session.reset(artifact_token)
            if collaboration_session is not None:
                self._last_collaboration_messages = collaboration_session.history
            if collaboration_token is not None:
                _active_collaboration_session.reset(collaboration_token)
            _active_tool_session.reset(tool_token)
            if root_token is not None:
                _active_root_id.reset(root_token)
            if routing_token is not None:
                _active_provider_bindings.reset(routing_token)
            _active_usage.reset(usage_token)
            if control_token is not None:
                _active_execution_control.reset(control_token)
            self._running = False
