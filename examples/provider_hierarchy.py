"""Run provider-powered decision strategies using only offline MockProviders."""

from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    ProviderFinalReviewer, ProviderManagerReviewer, ProviderTaskDecomposer,
    ProviderTaskTriage,
)
from agenttree.providers import MockProvider, ProviderConfig


def main() -> None:
    """Use independent decision providers and an externally bound worker model."""
    triage_provider = MockProvider(response_content=
        '{"objective":"Prepare a summary","required_capabilities":["analysis"]}')
    decomposer_provider = MockProvider(response_content=
        '{"subtasks":[{"objective":"Summarize","required_capabilities":["summarize"]}]}')
    review_provider = MockProvider(response_content=
        '{"decision":"pass","feedback":"Meets the objective"}')
    framework = AgentTree(
        root_agent=RootAgent(name="Root"),
        triage=ProviderTaskTriage(triage_provider),
        decomposer=ProviderTaskDecomposer(decomposer_provider),
        manager_reviewer=ProviderManagerReviewer(review_provider),
        final_reviewer=ProviderFinalReviewer(review_provider),
    )
    manager = ManagerAgent(name="Coordinator", capabilities=("analysis",))
    specialist = SpecialistAgent(name="Summarizer", capabilities=("summarize",))
    framework.register_manager(manager)
    framework.register_specialist(manager, specialist)
    worker_provider = MockProvider(
        ProviderConfig(provider_name="worker"), response_content="Offline summary.",
    )
    framework.register_provider(worker_provider)
    framework.bind_provider(specialist, "worker")
    result = framework.run(Task(objective="Prepare a summary"))
    print(f"Status: {result.status.value}; success: {result.success}")
    print(f"Trace events: {result.trace.event_count}")


if __name__ == "__main__":
    main()
