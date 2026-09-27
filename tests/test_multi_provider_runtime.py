"""Provider selection through every Root, Manager, and Specialist phase."""

from types import SimpleNamespace

from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    ProviderFinalReviewer, ProviderManagerReviewer, ProviderTaskDecomposer,
    ProviderTaskTriage,
)
from agenttree.providers import (
    BaseProvider, CerebrasProvider, GeminiProvider, GroqProvider, MockProvider,
    OpenRouterProvider, ProviderConfig, ProviderRequest, ProviderResponse,
    ProviderUsage,
)


class GeminiModels:
    def __init__(self):
        self.calls = []
        self.contents = [
            '{"delegate":true}',
            '{"objective":"Explain a gateway","required_capabilities":["manage"]}',
            '{"decision":"pass","feedback":"accepted"}',
            'An API gateway routes and protects requests.',
        ]

    def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents))
        return SimpleNamespace(text=self.contents.pop(0), model_version=model,
            usage_metadata=SimpleNamespace(prompt_token_count=1,
                candidates_token_count=1, total_token_count=2))

    def list(self):
        return [SimpleNamespace(name="models/gemini-test", display_name="Gemini Test",
                                input_token_limit=1000,
                                supported_actions=("generateContent",))]


def _completion(content, model):
    return {"model": model, "choices": [{"message": {"content": content},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 1,
            "completion_tokens": 1, "total_tokens": 2}}


def test_gemini_root_groq_manager_and_two_compatible_specialists(monkeypatch):
    root = RootAgent(name="Root")
    manager = ManagerAgent(name="Manager", capabilities=("manage",))
    a = SpecialistAgent(name="A", capabilities=("work",))
    b = SpecialistAgent(name="B", capabilities=("work",))
    fallback = MockProvider()
    tree = AgentTree(root_agent=root,
        triage=ProviderTaskTriage(fallback),
        decomposer=ProviderTaskDecomposer(fallback),
        manager_reviewer=ProviderManagerReviewer(fallback),
        final_reviewer=ProviderFinalReviewer(fallback))
    tree.register_manager(manager)
    tree.register_specialist(manager, a)
    tree.register_specialist(manager, b)

    native = GeminiModels()
    gemini = GeminiProvider(ProviderConfig("gemini"), client=SimpleNamespace(models=native))
    groq = GroqProvider(ProviderConfig("groq"), api_key="fake")
    openrouter = OpenRouterProvider(ProviderConfig("openrouter"), api_key="fake")
    cerebras = CerebrasProvider(ProviderConfig("cerebras"), api_key="fake")
    calls = {"groq": [], "openrouter": [], "cerebras": []}
    manager_answers = [
        '{"subtasks":[{"objective":"Explain the gateway",'
        '"required_capabilities":["work"]}]}',
        '{"decision":"pass","feedback":"accepted"}',
    ]

    def fake(provider, label, answers):
        def respond(path, payload=None, *, timeout=None):
            calls[label].append((path, payload))
            return _completion(answers.pop(0), payload["model"])
        monkeypatch.setattr(provider, "_json_request", respond)

    fake(groq, "groq", manager_answers)
    fake(openrouter, "openrouter", ["OpenRouter result"])
    fake(cerebras, "cerebras", ["Cerebras result"])
    for provider in (gemini, groq, openrouter, cerebras):
        tree.register_provider(provider)
    tree.bind_provider(root, gemini, model="models/gemini-test")
    tree.bind_provider(manager, groq, model="groq/model")
    tree.bind_provider(a, openrouter, model="vendor/model:free")
    tree.bind_provider(b, cerebras, model="cerebras/model")

    result = tree.run(Task(objective="Explain what an API gateway does."))
    assert result.success
    assert result.final_output == "An API gateway routes and protects requests."
    assert len(native.calls) == 4
    assert all(model == "models/gemini-test" for model, _ in native.calls)
    assert len(calls["groq"]) == 2
    assert all(payload["model"] == "groq/model" for _, payload in calls["groq"])
    assert calls["openrouter"][0][1]["model"] == "vendor/model:free"
    assert calls["cerebras"][0][1]["model"] == "cerebras/model"
    assert fallback.requests == ()
    assert result.usage["total_tokens"] == 16
    assert result.usage["unreported_calls"] == 0
    assert result.trace.events[-2].event_type == "root.synthesis.completed"
    assert result.trace.events[-1].event_type == "execution.completed"
    assert result.orchestration["managers"][0]["subtasks"][0]["specialists"]
    assert "GeminiModels" not in str(result.final_output)


def test_two_managers_route_decisions_to_their_own_providers():
    class SequenceProvider(BaseProvider):
        def __init__(self, name, contents):
            super().__init__(ProviderConfig(name))
            self.contents = list(contents)
            self.requests = []

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            self.requests.append(request)
            return ProviderResponse(content=self.contents.pop(0), provider=self.name,
                                    model=request.model, usage=ProviderUsage(1, 1, 2))

    root_provider = SequenceProvider("root", [
        '{"delegate":true}',
        '{"objective":"Work","required_capabilities":["manage"]}',
        '{"decision":"pass","feedback":"accepted"}',
        'Both departments completed their work.',
    ])
    manager_a = SequenceProvider("manager-a", [
        '{"subtasks":[{"objective":"A work","required_capabilities":["work"]}]}',
        '{"decision":"pass","feedback":"accepted"}',
    ])
    manager_b = SequenceProvider("manager-b", [
        '{"subtasks":[{"objective":"B work","required_capabilities":["work"]}]}',
        '{"decision":"pass","feedback":"accepted"}',
    ])
    worker_a = SequenceProvider("worker-a", ["A result"])
    worker_b = SequenceProvider("worker-b", ["B result"])
    root = RootAgent(name="Root")
    tree = AgentTree(root_agent=root,
        triage=ProviderTaskTriage(root_provider),
        decomposer=ProviderTaskDecomposer(root_provider),
        manager_reviewer=ProviderManagerReviewer(root_provider),
        final_reviewer=ProviderFinalReviewer(root_provider))
    managers = [ManagerAgent(name="A", capabilities=("manage",)),
                ManagerAgent(name="B", capabilities=("manage",))]
    workers = [SpecialistAgent(name="A worker", capabilities=("work",)),
               SpecialistAgent(name="B worker", capabilities=("work",))]
    for manager, worker in zip(managers, workers):
        tree.register_manager(manager)
        tree.register_specialist(manager, worker)
    for provider in (root_provider, manager_a, manager_b, worker_a, worker_b):
        tree.register_provider(provider)
    tree.bind_provider(root, root_provider, model="root/model")
    for manager, provider in zip(managers, (manager_a, manager_b)):
        tree.bind_provider(manager, provider, model=provider.name + "/model")
    for worker, provider in zip(workers, (worker_a, worker_b)):
        tree.bind_provider(worker, provider, model=provider.name + "/model")
    result = tree.run(Task(objective="Complete both departments"))
    assert result.success and len(result.manager_results) == 2
    assert result.final_output == "Both departments completed their work."
    assert [request.metadata["strategy"] for request in manager_a.requests] == [
        "decomposition", "manager_review"]
    assert [request.metadata["strategy"] for request in manager_b.requests] == [
        "decomposition", "manager_review"]
    assert all(request.model == "manager-a/model" for request in manager_a.requests)
    assert all(request.model == "manager-b/model" for request in manager_b.requests)
    assert len(worker_a.requests) == len(worker_b.requests) == 1
    assert result.usage["total_tokens"] == 20
