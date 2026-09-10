"""Run the optional LangGraph backend with deterministic, offline components."""

from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    RuleBasedTaskTriage, StaticTaskDecomposer, StaticManagerReviewer, StaticFinalReviewer,
)
from agenttree.models import SubtaskTemplate
from agenttree.orchestration.backends import LangGraphOrchestrationBackend
from agenttree.providers import MockProvider


def main() -> None:
    """Run after installing AgentTree's langgraph extra; no provider API is used."""
    framework = AgentTree(
        root_agent=RootAgent(name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("analysis",)),
        decomposer=StaticTaskDecomposer((SubtaskTemplate("Summarize", ("summarize",)),)),
        manager_reviewer=StaticManagerReviewer(), final_reviewer=StaticFinalReviewer(),
        orchestration_backend=LangGraphOrchestrationBackend(),
    )
    manager = ManagerAgent(name="Coordinator", capabilities=("analysis",))
    specialist = SpecialistAgent(name="Summarizer", capabilities=("summarize",))
    provider = MockProvider(response_content="Offline summary.")
    framework.register_manager(manager)
    framework.register_specialist(manager, specialist)
    framework.register_provider(provider)
    framework.bind_provider(specialist, provider)
    result = framework.run(Task(objective="Summarize inputs"))
    print(f"Status: {result.status.value}; success: {result.success}")
    print(f"Trace events: {result.trace.event_count}")
    print(f"State: {framework.last_state.current_phase.value}")


if __name__ == "__main__":
    main()
