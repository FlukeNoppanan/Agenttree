"""Opt-in, low-token real provider smoke tests."""

import os

import pytest

from agenttree.providers import (
    CerebrasProvider, GeminiProvider, GroqProvider, OpenRouterProvider,
    ProviderConfig, ProviderModel, ProviderRequest,
)


def _credentials(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not configured")
    return value


def _text_model(models: tuple[ProviderModel, ...], provider: str) -> ProviderModel:
    preferred = os.environ.get(f"{provider.upper()}_TEST_MODEL")
    if preferred:
        for model in models:
            if model.id == preferred:
                return model
        pytest.skip("Configured text test model is not in discovery results")
    if provider == "groq":
        # Groq discovery can list text classifiers and guard models before
        # general chat models. Prefer only an account-visible documented chat
        # model instead of accepting the first item with text modalities.
        documented_text_models = (
            "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
            "openai/gpt-oss-20b", "openai/gpt-oss-120b",
        )
        by_id = {model.id: model for model in models}
        for model_id in documented_text_models:
            model = by_id.get(model_id)
            if model is not None and (model.capabilities is None or
                                      model.capabilities.chat is not False):
                return model
    for model in models:
        metadata = model.metadata
        inputs = metadata.get("input_modalities")
        outputs = metadata.get("output_modalities")
        architecture = metadata.get("architecture")
        if model.capabilities is not None and model.capabilities.chat is False:
            continue
        if (isinstance(inputs, list) and isinstance(outputs, list) and
                "text" in inputs and "text" in outputs):
            return model
        if isinstance(architecture, dict) and architecture.get("modality") == "text":
            return model
        if provider == "gemini" and model.capabilities is not None and model.capabilities.chat:
            return model
    pytest.skip("Discovery did not identify a text chat model; set a provider TEST_MODEL override")


def test_groq_sparse_discovery_selects_only_account_visible_chat_model(monkeypatch):
    monkeypatch.delenv("GROQ_TEST_MODEL", raising=False)
    models = (ProviderModel(id="whisper-large-v3", provider="groq"),
              ProviderModel(id="llama-3.1-8b-instant", provider="groq"))
    assert _text_model(models, "groq").id == "llama-3.1-8b-instant"


@pytest.mark.parametrize("provider_name,key_name,provider_class", [
    ("gemini", "GEMINI_API_KEY", GeminiProvider),
    ("groq", "GROQ_API_KEY", GroqProvider),
    ("openrouter", "OPENROUTER_API_KEY", OpenRouterProvider),
    ("cerebras", "CEREBRAS_API_KEY", CerebrasProvider),
])
def test_live_provider(provider_name, key_name, provider_class):
    key = _credentials(key_name)
    if provider_name == "gemini":
        pytest.importorskip("google.genai")
    provider = provider_class(ProviderConfig(provider_name), api_key=key)
    models = provider.list_models(refresh=True)
    assert models
    model = _text_model(models, provider_name)
    response = provider.generate(ProviderRequest(prompt="Reply with one short greeting.",
                                                  model=model.id, max_tokens=128))
    assert response.content.strip() and response.provider == provider_name
    assert response.model
    if response.usage is not None:
        assert response.usage.total_tokens is None or response.usage.total_tokens >= 0


def test_live_mixed_provider_tree():
    keys = {name: os.environ.get(name) for name in (
        "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY")}
    if not all(keys.values()):
        pytest.skip("All four live provider credentials are required")
    pytest.importorskip("google.genai")
    from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent, Task
    from agenttree.core import (
        ProviderFinalReviewer, ProviderManagerReviewer, ProviderTaskDecomposer,
        ProviderTaskTriage, RootPlan, BaseRootPlanner,
    )

    providers = {
        "gemini": GeminiProvider(ProviderConfig("gemini"), api_key=keys["GEMINI_API_KEY"]),
        "groq": GroqProvider(ProviderConfig("groq"), api_key=keys["GROQ_API_KEY"]),
        "openrouter": OpenRouterProvider(ProviderConfig("openrouter"), api_key=keys["OPENROUTER_API_KEY"]),
        "cerebras": CerebrasProvider(ProviderConfig("cerebras"), api_key=keys["CEREBRAS_API_KEY"]),
    }
    models = {name: _text_model(provider.list_models(refresh=True), name)
              for name, provider in providers.items()}

    class Delegate(BaseRootPlanner):
        def plan(self, task, root):
            return RootPlan(delegate=True)

    root = RootAgent(name="Root")
    manager = ManagerAgent(name="Manager", capabilities=("explain",))
    workers = (SpecialistAgent(name="A", capabilities=("explain",)),
               SpecialistAgent(name="B", capabilities=("explain",)))
    tree = AgentTree(root_agent=root, triage=ProviderTaskTriage(providers["gemini"]),
        decomposer=ProviderTaskDecomposer(providers["groq"]),
        manager_reviewer=ProviderManagerReviewer(providers["groq"]),
        final_reviewer=ProviderFinalReviewer(providers["gemini"]),
        root_planner=Delegate())
    tree.register_manager(manager)
    for worker in workers:
        tree.register_specialist(manager, worker)
    for provider in providers.values():
        tree.register_provider(provider)
    tree.bind_provider(root, providers["gemini"], model=models["gemini"].id)
    tree.bind_provider(manager, providers["groq"], model=models["groq"].id)
    tree.bind_provider(workers[0], providers["openrouter"], model=models["openrouter"].id)
    tree.bind_provider(workers[1], providers["cerebras"], model=models["cerebras"].id)
    result = tree.run(Task(objective="Briefly explain what an API gateway does."))
    assert result.final_output and result.trace.events[-2].event_type == "root.synthesis.completed"
    assert result.trace.events[-1].event_type == "execution.completed"
