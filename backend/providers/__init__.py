"""LLM provider factory and aggregation."""

import os

from providers.base import LLMProvider
from providers.ollama import OllamaProvider

_providers: dict[str, LLMProvider] = {}


def _init_providers():
    """Initialize available providers from environment configuration."""
    if _providers:
        return

    # Ollama is always available
    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    _providers["ollama"] = OllamaProvider(ollama_url)

    # Azure OpenAI is available if configured
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    azure_deployments = os.environ.get("AZURE_OPENAI_DEPLOYMENTS", "").strip()
    if azure_endpoint and azure_key and azure_deployments:
        from providers.azure_openai import AzureOpenAIProvider

        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        _providers["azure"] = AzureOpenAIProvider(
            endpoint=azure_endpoint,
            api_key=azure_key,
            deployments=azure_deployments,
            api_version=api_version,
        )


def get_provider(name: str) -> LLMProvider:
    """Get a provider instance by name."""
    _init_providers()
    provider = _providers.get(name)
    if provider is None:
        raise ValueError(f"Unknown or unconfigured provider: {name}")
    return provider


def get_all_models() -> list[dict]:
    """Aggregate models from all available providers."""
    _init_providers()
    models = []
    for provider in _providers.values():
        try:
            models.extend(provider.list_models())
        except Exception:
            pass
    return models


def get_available_providers() -> list[str]:
    """Return names of configured providers."""
    _init_providers()
    return list(_providers.keys())
