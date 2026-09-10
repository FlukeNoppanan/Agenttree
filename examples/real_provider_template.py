"""Configuration template for optional live providers; never run by validation."""

import os

from agenttree.providers import (
    BaseProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderConfig,
)


def configured_provider(kind: str) -> BaseProvider:
    """Construct one explicitly selected live adapter from environment settings."""
    model = os.environ["AGENTTREE_MODEL"]
    config = ProviderConfig(provider_name=f"{kind}-worker", model=model)
    if kind == "openai":
        # The OpenAI SDK reads OPENAI_API_KEY from the environment.
        return OpenAIProvider(config)
    if kind == "gemini":
        # The Google SDK reads GOOGLE_API_KEY from the environment.
        return GeminiProvider(config)
    if kind == "ollama":
        # OLLAMA_HOST may identify an explicitly operated Ollama service.
        return OllamaProvider(config, host=os.environ.get("OLLAMA_HOST"))
    raise ValueError("kind must be openai, gemini, or ollama")


if __name__ == "__main__":
    print(
        "Template only: set AGENTTREE_MODEL and the selected SDK's environment "
        "configuration, then import configured_provider(). No request was sent."
    )
