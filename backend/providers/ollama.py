"""Ollama LLM provider."""

import requests

from providers.base import LLMProvider


class OllamaProvider(LLMProvider):

    def __init__(self, url: str):
        self._url = url.rstrip("/")
        self._http = requests.Session()

    @property
    def name(self) -> str:
        return "ollama"

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
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": thinking,
            "keep_alive": keep_alive,
        }
        if options:
            payload["options"] = options

        resp = self._http.post(
            f"{self._url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "content": data["message"]["content"],
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_duration": data.get("eval_duration"),
            "total_duration": data.get("total_duration"),
        }

    def is_model_loaded(self, model: str) -> bool:
        try:
            resp = self._http.get(f"{self._url}/api/ps", timeout=5)
            resp.raise_for_status()
            loaded = [m["name"] for m in resp.json().get("models", [])]
            return model in loaded
        except Exception:
            return False

    def list_models(self) -> list[dict]:
        resp = self._http.get(f"{self._url}/api/tags", timeout=10)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
        return [
            {"name": n, "provider": "ollama"}
            for n in names
            if self.model_has_vision(n)
        ]

    def model_has_vision(self, name: str) -> bool:
        try:
            resp = self._http.post(
                f"{self._url}/api/show", json={"name": name}, timeout=5
            )
            resp.raise_for_status()
            return "vision" in (resp.json().get("capabilities") or [])
        except Exception:
            return False
