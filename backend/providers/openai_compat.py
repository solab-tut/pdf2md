"""OpenAI-compatible LLM provider (standard /v1/chat/completions API).

Works with OpenAI, LM Studio, LocalAI, vLLM, and any service that
implements the OpenAI chat completions API.
"""

import requests

from providers.base import LLMProvider


def _convert_messages(messages: list[dict]) -> list[dict]:
    """Convert Ollama-style messages to OpenAI multimodal format."""
    converted = []
    for msg in messages:
        images = msg.get("images", [])
        if not images:
            converted.append({"role": msg["role"], "content": msg["content"]})
        else:
            content_parts = [{"type": "text", "text": msg["content"]}]
            for b64 in images:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",
                        },
                    }
                )
            converted.append({"role": msg["role"], "content": content_parts})
    return converted


class OpenAICompatProvider(LLMProvider):

    def __init__(self, base_url: str, api_key: str, models: str):
        self._base_url = base_url.rstrip("/")
        self._models = [m.strip() for m in models.split(",") if m.strip()]
        self._http = requests.Session()
        if api_key:
            self._http.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def name(self) -> str:
        return "openai"

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        stream: bool = False,
        thinking: bool = False,
        keep_alive: str = "15m",
        options: dict | None = None,
        timeout: int = 600,
    ) -> dict:
        options = options or {}
        max_tokens = options.get("num_predict", 8192)
        temperature = options.get("temperature", 0.1)

        payload = {
            "model": model,
            "messages": _convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        resp = self._http.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        return {
            "content": data["choices"][0]["message"]["content"],
            "prompt_eval_count": usage.get("prompt_tokens"),
            "eval_count": usage.get("completion_tokens"),
            "prompt_eval_duration": None,
            "eval_duration": None,
            "total_duration": None,
        }

    def list_models(self) -> list[dict]:
        return [{"name": m, "provider": "openai"} for m in self._models]

    def model_has_vision(self, name: str) -> bool:
        vision_patterns = (
            "gpt-4o", "gpt-4-turbo", "gpt-4v", "gpt-4.1",
            "llava", "vision", "vl", "minicpm-v",
        )
        return any(p in name.lower() for p in vision_patterns)
