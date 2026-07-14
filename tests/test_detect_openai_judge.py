import json
import urllib.error
import urllib.request

import pytest

from agentgates.detect.judge import OpenAICompatibleJudge


def _reachable(url="http://localhost:1234/v1/models") -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)  # noqa: S310 (local only)
        return True
    except (urllib.error.URLError, OSError):
        return False


def test_payload_shape():
    judge = OpenAICompatibleJudge(model="qwen", base_url="http://x/v1")
    payload = judge._payload("hello")
    assert payload["model"] == "qwen"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["temperature"] == 0  # deterministic for reproducible evals
    assert payload["stream"] is False


def test_extract_content():
    data = {"choices": [{"message": {"content": '{"drifted": false}'}}]}
    assert OpenAICompatibleJudge._extract(data) == '{"drifted": false}'


def test_extract_missing_is_empty_string():
    assert OpenAICompatibleJudge._extract({}) == ""
    assert OpenAICompatibleJudge._extract({"choices": []}) == ""


def test_base_url_trailing_slash_normalized():
    judge = OpenAICompatibleJudge(model="m", base_url="http://x/v1/")
    assert judge._endpoint == "http://x/v1/chat/completions"


def test_complete_uses_post(monkeypatch):
    judge = OpenAICompatibleJudge(model="m", base_url="http://x/v1")
    captured = {}

    def fake_post(payload):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(judge, "_post", fake_post)
    assert judge.complete("prompt text") == "ok"
    assert captured["payload"]["messages"][0]["content"] == "prompt text"


@pytest.mark.skipif(not _reachable(), reason="LM Studio not running on :1234")
def test_live_lmstudio_returns_json():
    # Integration: real call to a locally running LM Studio.
    import urllib.request as r

    models = json.load(r.urlopen("http://localhost:1234/v1/models", timeout=5))
    model_id = next(
        m["id"] for m in models["data"] if "embed" not in m["id"].lower()
    )
    judge = OpenAICompatibleJudge(model=model_id)
    out = judge.complete('Reply with ONLY this JSON and nothing else: {"ok": true}')
    assert "{" in out and "}" in out
