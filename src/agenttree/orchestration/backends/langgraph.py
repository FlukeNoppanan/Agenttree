"""Optional LangGraph coordination of existing AgentTree phase operations."""

import importlib
from typing import Any, Callable, TypedDict

from agenttree.orchestration.backends.base import BaseOrchestrationBackend
from agenttree.orchestration.backends.context import OrchestrationContext
from agenttree.orchestration.backends.errors import (
    BackendConfigurationError, BackendDependencyError,
)
from agenttree.orchestration.models import FinalResult


class _GraphState(TypedDict):
    # Ephemeral per-invocation context; never serialized or checkpointed.
    context: OrchestrationContext
    result: FinalResult | None


def _node(phase: str) -> Callable[[_GraphState], dict[str, Any]]:
    def invoke(state: _GraphState) -> dict[str, Any]:
        context = state["context"]
        getattr(context, phase)()
        return {"result": context.result}
    return invoke


def _terminal(state: _GraphState) -> bool:
    return state["result"] is not None


class LangGraphOrchestrationBackend(BaseOrchestrationBackend):
    """Coordinate five synchronous phases through an optional StateGraph.

    Import and compilation occur only at construction. There is no checkpointer,
    persistence, asynchronous public API, or graph-level revision cycle. Each
    invocation receives its own context; compiled nodes retain no task state.
    """

    def __init__(self) -> None:
        try:
            graph_api = importlib.import_module("langgraph.graph")
        except ImportError as error:
            raise BackendDependencyError(
                'LangGraph backend requires its optional dependencies; '
                'install with: pip install "agenttree[langgraph]"',
            ) from error
        try:
            builder = graph_api.StateGraph(_GraphState)
            phases = ("planning", "delegation", "execution", "manager_review", "final_review")
            for phase in phases:
                builder.add_node(phase, _node(phase))
            builder.add_edge(graph_api.START, "planning")
            builder.add_conditional_edges(
                "planning", _terminal, {True: graph_api.END, False: "delegation"},
            )
            builder.add_conditional_edges(
                "delegation", _terminal, {True: graph_api.END, False: "execution"},
            )
            builder.add_edge("execution", "manager_review")
            builder.add_edge("manager_review", "final_review")
            builder.add_edge("final_review", graph_api.END)
            self._graph = builder.compile()
        except Exception as error:
            raise BackendConfigurationError("Could not construct the LangGraph backend") from error

    def run(self, context: OrchestrationContext) -> FinalResult:
        """Invoke synchronously, leaving phase exceptions unwrapped."""
        state = self._graph.invoke({"context": context, "result": None})
        result = state.get("result")
        if not isinstance(result, FinalResult):
            raise BackendConfigurationError("LangGraph backend produced no FinalResult")
        return result
