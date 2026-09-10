"""High-level composition of the existing synchronous AgentTree workflow."""

from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

from agenttree.agents import BaseAgent, ManagerAgent, RootAgent, SpecialistAgent
from agenttree.config import AgentTreeConfig
from agenttree.core import (
    BaseFinalReviewer, BaseManagerReviewer, BaseSpecialistExecutor,
    BaseTaskDecomposer, BaseTaskTriage, ProviderSpecialistExecutor,
    WorkflowStateManager,
)
from agenttree.models import (
    Task, WorkflowPhase, WorkflowState,
)
from agenttree.orchestration import FinalResult, OrchestrationEngine
from agenttree.orchestration.backends import (
    BaseOrchestrationBackend, SequentialOrchestrationBackend,
    BackendConfigurationError, OrchestrationContext,
)
from agenttree.providers import BaseProvider, ProviderRegistry
from agenttree.registry import CapabilityRegistry
from agenttree.tools import BaseTool, ToolBindingRegistry, ToolRegistry


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
        self._executor = executor
        self._decomposer = decomposer
        self._manager_reviewer = manager_reviewer
        self._final_reviewer = final_reviewer
        self._engine = OrchestrationEngine(
            triage=triage, registry=self._agents,
            manager_match_all=self._config.manager_match_all,
        )
        self._states = WorkflowStateManager()
        self._last_state: WorkflowState | None = None
        self._active_context: OrchestrationContext | None = None
        self._running = False
        self._validate_relationships()

    @property
    def config(self) -> AgentTreeConfig:
        """Return the immutable workflow configuration."""
        return self._config

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
        """Return a read-only snapshot of specialist IDs to provider names."""
        return MappingProxyType(dict(self._provider_bindings))

    @property
    def tool_bindings(self) -> Mapping[str, tuple[str, ...]]:
        """Return a read-only snapshot of external specialist tool assignments."""
        return self._tool_bindings.bindings

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
        self, specialist: SpecialistAgent, provider: BaseProvider | str,
    ) -> None:
        """Bind a registered specialist to a registered provider object or name.

        Calling again explicitly replaces the binding. A custom injected
        executor manages its own bindings; this API then raises ValueError.
        """
        self._ensure_idle()
        self._require_registered(specialist, SpecialistAgent)
        if self._executor is not None:
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
        self._provider_bindings[specialist.id] = resolved.name

    def bind_tool(self, specialist: SpecialistAgent, tool: BaseTool | str) -> None:
        """Assign a registered tool object or ID; duplicate pairs are errors."""
        self._ensure_idle()
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

    def run(self, task: Task) -> FinalResult:
        """Run the five engine phases on an isolated copy of the canonical Task.

        No managers or assignments return explicitly marked framework failures.
        Execution failures remain reviewable outcomes; reviewers retain their
        existing authority. Configuration, decision-output, and decision-provider
        errors are re-raised after recording a FAILED state. Tools are never
        automatically selected or invoked by this method.
        """
        self._ensure_idle()
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        working = deepcopy(task)
        self._last_state = self._states.create_initial(working)
        self._running = True
        try:
            self._validate_relationships()
            executor = self._executor if self._executor is not None else ProviderSpecialistExecutor(
                provider_registry=self._providers, provider_bindings=self._provider_bindings,
            )
            context = OrchestrationContext(
                task=working, state=self._last_state, engine=self._engine,
                root_agent=self._root, config=self._config, agents=self._agents,
                decomposer=self._decomposer, executor=executor,
                manager_reviewer=self._manager_reviewer,
                final_reviewer=self._final_reviewer,
            )
            self._active_context = context
            try:
                result = self._backend.run(context)
                if not isinstance(result, FinalResult) or result.task_id != working.id:
                    raise BackendConfigurationError("Backend must return this task's FinalResult")
                if context.result is None:
                    raise BackendConfigurationError("Backend did not complete the workflow")
                return result
            finally:
                self._last_state = context.state
                self._active_context = None
        except Exception as error:
            assert self._last_state is not None
            if self._last_state.current_phase not in (WorkflowPhase.COMPLETED, WorkflowPhase.FAILED):
                self._last_state = self._states.fail(
                    self._last_state, reason=f"Run raised {type(error).__name__}",
                )
            raise
        finally:
            self._running = False
