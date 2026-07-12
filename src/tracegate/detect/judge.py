"""LLM-judge interface: model-agnostic, cached by prompt hash.

Deterministic checks run first everywhere in TraceGate; judges are only for
semantics (drift, groundedness). CachedJudge keys by prompt content hash so
CI reruns are cheap and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Protocol


class Judge(Protocol):
    def complete(self, prompt: str) -> str: ...


def extract_json(text: str) -> dict:
    """Parse the first JSON object found in text; {} if none parses."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return {}


class CachedJudge:
    def __init__(self, inner: Judge, cache_path: Path | None = None) -> None:
        self._inner = inner
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, str] = {}
        if self._cache_path and self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))

    def complete(self, prompt: str) -> str:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._inner.complete(prompt)
            if self._cache_path:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._cache_path.write_text(
                    json.dumps(self._cache), encoding="utf-8"
                )
        return self._cache[key]


class ClaudeJudge:
    """Judge backed by the Claude API (requires the 'judge' extra)."""

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "ClaudeJudge requires the anthropic package:"
                ' pip install "tracegate[judge]"'
            ) from e
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )


class OpenAICompatibleJudge:
    """Judge backed by any OpenAI-compatible chat endpoint.

    Works with LM Studio, Ollama, vLLM, llama.cpp server, etc. Uses only the
    standard library (urllib), so no extra dependency and no data leaves the
    machine when the endpoint is local -- the same local-first wedge the rest
    of TraceGate is built on. Defaults target LM Studio's server.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._timeout = timeout

    def _payload(self, prompt: str) -> dict:
        return {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": False,
        }

    @staticmethod
    def _extract(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(  # noqa: S310 (user-configured endpoint)
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def complete(self, prompt: str) -> str:
        return self._extract(self._post(self._payload(prompt)))
