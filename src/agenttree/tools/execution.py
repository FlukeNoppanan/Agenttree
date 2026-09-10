"""Explicit synchronous tool resolution, authorization, and invocation."""

from typing import Any, Mapping

from agenttree.agents import SpecialistAgent
from agenttree.models import ExecutionEvent, ExecutionTrace
from agenttree.tools.bindings import ToolBindingRegistry
from agenttree.tools.models import ToolResult
from agenttree.tools.registry import ToolRegistry
from agenttree.tracing import ExecutionEventType


class ToolExecutor:
    """Invoke a specifically named or identified registered tool."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        bindings: ToolBindingRegistry | None = None,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry")
        if bindings is not None and not isinstance(bindings, ToolBindingRegistry):
            raise TypeError("bindings must be a ToolBindingRegistry or None")
        self._registry = registry
        self._bindings = bindings

    @property
    def registry(self) -> ToolRegistry:
        """Return the injected tool registry."""
        return self._registry

    @property
    def bindings(self) -> ToolBindingRegistry | None:
        """Return the optional external binding registry."""
        return self._bindings

    def execute(
        self,
        *,
        specialist: SpecialistAgent,
        arguments: Mapping[str, Any],
        tool_id: str | None = None,
        tool_name: str | None = None,
        trace: ExecutionTrace | None = None,
    ) -> ToolResult:
        """Resolve and invoke exactly one explicit tool selection."""
        if not isinstance(specialist, SpecialistAgent):
            raise TypeError("specialist must be a SpecialistAgent")
        if (tool_id is None) == (tool_name is None):
            raise ValueError("Provide exactly one of tool_id or tool_name")
        if trace is not None and not isinstance(trace, ExecutionTrace):
            raise TypeError("trace must be an ExecutionTrace or None")
        try:
            tool = (
                self._registry.get_by_id(tool_id)
                if tool_id is not None
                else self._registry.get_by_name(tool_name)  # type: ignore[arg-type]
            )
        except KeyError:
            requested = f"ID '{tool_id}'" if tool_id is not None else f"name '{tool_name}'"
            raise KeyError(
                f"Tool with {requested} is not registered; register it before execution",
            ) from None
        if self._bindings is not None and not self._bindings.is_assigned(
            specialist.id, tool.id,
        ):
            raise PermissionError(
                f"Tool {tool.id} is not assigned to specialist {specialist.id}",
            )
        self._append_event(
            trace,
            ExecutionEventType.TOOL_EXECUTION_STARTED,
            specialist,
            tool.id,
            tool.name,
            tool.metadata,
        )
        try:
            result = tool.invoke(arguments)
            if not isinstance(result, ToolResult):
                raise TypeError("tool.invoke must return a ToolResult")
            if result.tool_id != tool.id:
                raise ValueError("ToolResult tool_id must match the invoked tool")
        except Exception as error:
            self._append_event(
                trace,
                ExecutionEventType.TOOL_EXECUTION_FAILED,
                specialist,
                tool.id,
                tool.name,
                tool.metadata,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        event_type = (
            ExecutionEventType.TOOL_EXECUTION_COMPLETED
            if result.success
            else ExecutionEventType.TOOL_EXECUTION_FAILED
        )
        self._append_event(
            trace,
            event_type,
            specialist,
            tool.id,
            tool.name,
            tool.metadata,
            error=result.error,
        )
        return result

    @staticmethod
    def _append_event(
        trace: ExecutionTrace | None,
        event_type: ExecutionEventType,
        specialist: SpecialistAgent,
        tool_id: str,
        tool_name: str,
        tool_metadata: Mapping[str, Any],
        *,
        error: str | None = None,
    ) -> None:
        if trace is None:
            return
        trace.append(ExecutionEvent(
            event_type=event_type,
            task_id=trace.task_id,
            actor_id=specialist.id,
            message=event_type.value.replace("orchestration.", "").replace("_", " "),
            metadata={
                "tool_id": tool_id,
                "tool_name": tool_name,
                "tool_metadata": dict(tool_metadata),
                "error": error,
            },
        ))
