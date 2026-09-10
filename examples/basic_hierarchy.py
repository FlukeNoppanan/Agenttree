"""Run an explicit deterministic hierarchy offline after editable installation."""

from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    RuleBasedTaskTriage, StaticFinalReviewer, StaticManagerReviewer,
    StaticTaskDecomposer,
)
from agenttree.models import SubtaskTemplate
from agenttree.providers import MockProvider


def main() -> None:
    """Configure identities, strategies, and external bindings, then run a task."""
    framework = AgentTree(
        root_agent=RootAgent(name="Root"),
        triage=RuleBasedTaskTriage({}, fallback_capabilities=("analysis",)),
        decomposer=StaticTaskDecomposer((
            SubtaskTemplate("Summarize the supplied information", ("summarize",)),
        )),
        manager_reviewer=StaticManagerReviewer(),
        final_reviewer=StaticFinalReviewer(),
    )
    manager = ManagerAgent(name="Coordinator", capabilities=("analysis",))
    specialist = SpecialistAgent(name="Summarizer", capabilities=("summarize",))
    provider = MockProvider(response_content="A concise offline summary.")
    framework.register_manager(manager)
    framework.register_specialist(manager, specialist)
    framework.register_provider(provider)
    framework.bind_provider(specialist, provider)
    result = framework.run(Task(objective="Summarize the supplied information"))
    print(f"Status: {result.status.value}; success: {result.success}")
    print(f"Content: {result.content}")
    print(f"Trace events: {result.trace.event_count}")
    print(f"State: {framework.last_state.current_phase.value}")


if __name__ == "__main__":
    main()
