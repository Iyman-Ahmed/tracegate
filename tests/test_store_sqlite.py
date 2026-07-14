import pytest

from agentgates.schema import AgentInfo, AgentTrace, TaskSpec
from agentgates.store.sqlite import SQLiteTraceStore


def make_trace(desc: str = "task") -> AgentTrace:
    return AgentTrace(
        task=TaskSpec(description=desc),
        agent=AgentInfo(framework="test"),
    )


def test_save_and_load_roundtrip(tmp_path):
    store = SQLiteTraceStore(tmp_path / "traces.sqlite")
    trace = make_trace()
    store.save(trace)
    loaded = store.load(trace.trace_id)
    assert loaded.trace_id == trace.trace_id
    assert loaded.task.description == "task"


def test_load_missing_raises(tmp_path):
    store = SQLiteTraceStore(tmp_path / "traces.sqlite")
    with pytest.raises(KeyError):
        store.load("nope")


def test_save_is_upsert(tmp_path):
    store = SQLiteTraceStore(tmp_path / "traces.sqlite")
    trace = make_trace("v1")
    store.save(trace)
    trace.task.description = "v2"
    store.save(trace)
    assert store.load(trace.trace_id).task.description == "v2"
    assert len(store.list_traces()) == 1


def test_list_traces_ordered(tmp_path):
    store = SQLiteTraceStore(tmp_path / "traces.sqlite")
    a, b = make_trace("a"), make_trace("b")
    b.started_at = b.started_at.replace(year=b.started_at.year + 1)
    store.save(b)
    store.save(a)
    listed = store.list_traces()
    assert [t.task.description for t in listed] == ["a", "b"]
