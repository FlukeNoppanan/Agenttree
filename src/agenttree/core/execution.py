"""Specialist execution contract and provider-backed implementation."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

from agenttree.agents import SpecialistAgent
from agenttree.core.execution_control import ExecutionCancelled
from agenttree.models import AgentResult, Subtask, Task
from agenttree.providers import (
    BaseProvider,
    ProviderRegistry,
    ProviderRequest,
    ProviderResponse,
)


class BaseSpecialistExecutor(ABC):
    """Execute one specialist for one subtask and return an agent result."""

    @abstractmethod
    def execute(
        self,
        task: Task,
        subtask: Subtask,
        specialist: SpecialistAgent,
    ) -> AgentResult:
        """Execute ``specialist`` without making orchestration decisions."""
        raise NotImplementedError


class ProviderSpecialistExecutor(BaseSpecialistExecutor):
    """Execute specialists through externally configured model providers.

    Bindings map stable specialist IDs to names in an injected
    ``ProviderRegistry``. Missing bindings and invalid provider responses are
    configuration/contract errors and surface to callers. Exceptions raised by
    ``provider.generate`` become failed ``AgentResult`` values.
    """

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        provider_bindings: Mapping[str, str],
        model_bindings: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(provider_registry, ProviderRegistry):
            raise TypeError("provider_registry must be a ProviderRegistry")
        bindings = dict(provider_bindings)
        for specialist_id, provider_name in bindings.items():
            if not isinstance(specialist_id, str) or not specialist_id.strip():
                raise ValueError("Specialist binding IDs must be non-empty strings")
            if not isinstance(provider_name, str) or not provider_name.strip():
                raise ValueError("Provider binding names must be non-empty strings")
        self._provider_registry = provider_registry
        self._provider_bindings = bindings
        self._model_bindings = dict(model_bindings or {})

    @property
    def provider_registry(self) -> ProviderRegistry:
        """Return the injected provider registry."""
        return self._provider_registry

    @property
    def provider_bindings(self) -> Mapping[str, str]:
        """Return a read-only snapshot of specialist-to-provider bindings."""
        return MappingProxyType(dict(self._provider_bindings))

    def resolve_provider(self, specialist: SpecialistAgent) -> BaseProvider:
        """Resolve a specialist's configured provider or raise ``KeyError``."""
        if not isinstance(specialist, SpecialistAgent):
            raise TypeError("specialist must be a SpecialistAgent")
        try:
            provider_name = self._provider_bindings[specialist.id]
        except KeyError as error:
            raise KeyError(
                f"No provider binding for specialist: {specialist.id}",
            ) from error
        try:
            return self._provider_registry.get(provider_name)
        except KeyError:
            raise KeyError(
                f"Provider '{provider_name}' bound to specialist '{specialist.id}' "
                "is not registered",
            ) from None

    def execute(
        self,
        task: Task,
        subtask: Subtask,
        specialist: SpecialistAgent,
    ) -> AgentResult:
        """Build a generic request, invoke a provider, and normalize its output."""
        self._validate_inputs(task, subtask, specialist)
        provider = self.resolve_provider(specialist)
        request = self._build_request(task, subtask, specialist)
        if specialist.id in self._model_bindings:
            request = replace(request, model=self._model_bindings[specialist.id])
        try:
            from agenttree.tools.runtime import generate_with_tools
            response = generate_with_tools(specialist.id, provider, request, "specialist")
        except ExecutionCancelled:
            raise
        except Exception as error:
            from agenttree.core.execution_store import ExecutionRecoveryBlocked
            if isinstance(error, ExecutionRecoveryBlocked):
                raise
            return AgentResult(
                agent_id=specialist.id,
                success=False,
                error=f"{type(error).__name__}: provider call failed",
                metadata={"provider": provider.name, "error_type": type(error).__name__},
            )
        if not isinstance(response, ProviderResponse):
            raise TypeError("provider.generate must return a ProviderResponse")
        from agenttree.core.usage import record_usage
        record_usage("specialist", response)
        return AgentResult(
            agent_id=specialist.id,
            success=True,
            output=response.content,
            metadata={
                "provider": response.provider,
                "model": response.model,
                "usage": response.usage,
                "provider_metadata": deepcopy(response.metadata),
            },
        )

    @staticmethod
    def _build_request(
        task: Task,
        subtask: Subtask,
        specialist: SpecialistAgent,
    ) -> ProviderRequest:
        return ProviderRequest(
            prompt=subtask.objective,
            system_prompt=specialist.description or None,
            context={
                "task": {
                    "id": task.id,
                    "objective": task.objective,
                    "context": deepcopy(task.context.data),
                    "metadata": deepcopy(task.metadata),
                },
                "subtask": {
                    "id": subtask.id,
                    "parent_task_id": subtask.parent_task_id,
                    "manager_id": subtask.manager_id,
                    "required_capabilities": subtask.required_capabilities,
                    "metadata": deepcopy(subtask.metadata),
                },
                "specialist": {
                    "id": specialist.id,
                    "name": specialist.name,
                    "capabilities": specialist.capabilities,
                    "metadata": deepcopy(specialist.metadata),
                },
            },
            metadata={
                "task_id": task.id,
                "subtask_id": subtask.id,
                "specialist_id": specialist.id,
            },
        )

    @staticmethod
    def _validate_inputs(
        task: object,
        subtask: object,
        specialist: object,
    ) -> None:
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if not isinstance(subtask, Subtask):
            raise TypeError("subtask must be a Subtask")
        if not isinstance(specialist, SpecialistAgent):
            raise TypeError("specialist must be a SpecialistAgent")
        if subtask.parent_task_id != task.id:
            raise ValueError("Subtask parent_task_id must match the Task id")
