"""Azure OpenAI LLM provider."""

import requests

from providers.base import LLMProvider


def _convert_messages(messages: list[dict]) -> list[dict]:
    """Convert Ollama-style messages to OpenAI multimodal format.

    Ollama: {"role": ..., "content": str, "images": [b64]}
    OpenAI: {"role": ..., "content": [{"type": "text", ...}, {"type": "image_url", ...}]}
    """
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


class AzureOpenAIProvider(LLMProvider):

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployments: str,
        api_version: str = "2024-10-21",
    ):
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._deployments = [d.strip() for d in deployments.split(",") if d.strip()]
        self._api_version = api_version
        self._http = requests.Session()
        self._http.headers["api-key"] = api_key

    @property
    def name(self) -> str:
        return "azure"

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

        url = (
            f"{self._endpoint}/openai/deployments/{model}"
            f"/chat/completions?api-version={self._api_version}"
        )
        payload = {
            "messages": _convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        resp = self._http.post(url, json=payload, timeout=timeout)
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
        return [{"name": d, "provider": "azure"} for d in self._deployments]

    def model_has_vision(self, name: str) -> bool:
        vision_patterns = ("gpt-4o", "gpt-4-turbo", "gpt-4v", "gpt-4.1")
        return any(p in name.lower() for p in vision_patterns)
