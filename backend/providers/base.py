"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base interface for LLM providers.

    Messages use Ollama-style internal format:
        {"role": "system"|"user", "content": str, "images": [b64_str]}

    Responses are normalized to:
        {"content": str,
         "prompt_eval_count": int|None,
         "eval_count": int|None,
         "prompt_eval_duration": int|None,
         "eval_duration": int|None,
         "total_duration": int|None}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'ollama', 'azure')."""

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        stream: bool = False,
        thinking: bool = False,
        keep_alive: str = "15m",
        options: dict | None = None,
    ) -> dict:
        """Send a chat completion request and return normalized response."""

    @abstractmethod
    def list_models(self) -> list[dict]:
        """Return available models as [{"name": str, "provider": str}]."""

    @abstractmethod
    def model_has_vision(self, name: str) -> bool:
        """Check if a model supports vision/image input."""
