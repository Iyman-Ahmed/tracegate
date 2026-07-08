import pytest

from tracegate.schema import AgentInfo, AgentTrace, TaskSpec
from tracegate.store import default_store
from tracegate.store.jsonl import JSONLTraceStore


def make_trace(desc: str = "task") -> AgentTrace:
    return AgentTrace(
        task=TaskSpec(description=desc),
        agent=AgentInfo(framework="test"),
    )


def test_save_and_load_roundtrip(tmp_path):
    store = JSONLTraceStore(tmp_path / "traces.jsonl")
    trace = make_trace()
    store.save(trace)
    loaded = store.load(trace.trace_id)
    assert loaded.trace_id == trace.trace_id
    assert loaded.task.description == "task"


def test_load_missing_raises(tmp_path):
    store = JSONLTraceStore(tmp_path / "traces.jsonl")
    with pytest.raises(KeyError):
        store.load("nope")


def test_list_traces_ordered(tmp_path):
    store = JSONLTraceStore(tmp_path / "traces.jsonl")
    a, b = make_trace("a"), make_trace("b")
    store.save(a)
    store.save(b)
    listed = store.list_traces()
    assert [t.task.description for t in listed] == ["a", "b"]


def test_save_creates_parent_dir(tmp_path):
    store = JSONLTraceStore(tmp_path / "deep" / "traces.jsonl")
    store.save(make_trace())
    assert (tmp_path / "deep" / "traces.jsonl").exists()


def test_default_store_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEGATE_DIR", str(tmp_path / "custom"))
    store = default_store()
    assert store.path == tmp_path / "custom" / "traces.jsonl"
