import json

from tracegate.detect.judge import CachedJudge, extract_json


class CountingJudge:
    def __init__(self, response='{"ok": true}'):
        self.calls = 0
        self.response = response

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self.response


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_surrounding_text():
    text = 'Here is my verdict:\n```json\n{"drifted": false, "x": [1, 2]}\n```\nDone.'
    assert extract_json(text) == {"drifted": False, "x": [1, 2]}


def test_extract_json_garbage_returns_empty():
    assert extract_json("no json here") == {}
    assert extract_json("{broken") == {}


def test_cached_judge_memoizes():
    inner = CountingJudge()
    judge = CachedJudge(inner)
    assert judge.complete("p") == '{"ok": true}'
    assert judge.complete("p") == '{"ok": true}'
    assert inner.calls == 1
    judge.complete("different")
    assert inner.calls == 2


def test_cached_judge_persists_to_disk(tmp_path):
    path = tmp_path / "cache.json"
    inner = CountingJudge()
    CachedJudge(inner, cache_path=path).complete("p")
    assert path.exists()
    assert len(json.loads(path.read_text())) == 1

    inner2 = CountingJudge(response="never used")
    judge2 = CachedJudge(inner2, cache_path=path)
    assert judge2.complete("p") == '{"ok": true}'
    assert inner2.calls == 0
