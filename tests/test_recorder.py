from agentgates.recorder import TraceRecorder
from agentgates.schema import TaskSpec


class FakeStore:
    def __init__(self):
        self.saved = []

    def save(self, trace):
        self.saved.append(trace)


def test_records_steps_with_increasing_index():
    rec = TraceRecorder(task="do things", framework="test", store=FakeStore())
    rec.record_llm_call(model="m", prompt="p", response="r")
    rec.record_tool_call(tool_name="t", arguments={"a": 1}, result="ok")
    assert [s.index for s in rec.trace.steps] == [0, 1]
    assert rec.trace.steps[1].tool_name == "t"


def test_task_string_becomes_taskspec():
    rec = TraceRecorder(task="hello", framework="test", store=FakeStore())
    assert rec.trace.task == TaskSpec(description="hello")


def test_finish_sets_ended_at_and_saves_once():
    store = FakeStore()
    rec = TraceRecorder(task="x", framework="test", store=store)
    trace = rec.finish()
    rec.finish()
    assert trace.ended_at is not None
    assert len(store.saved) == 1


def test_context_manager_finishes():
    store = FakeStore()
    with TraceRecorder(task="x", framework="test", store=store) as rec:
        rec.record_llm_call(model="m", prompt="p", response="r")
    assert len(store.saved) == 1
    assert store.saved[0].ended_at is not None


def test_default_store_used_when_none(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTGATES_DIR", str(tmp_path))
    with TraceRecorder(task="x", framework="test"):
        pass
    assert (tmp_path / "traces.jsonl").exists()


def test_public_api_reexport():
    import agentgates

    assert agentgates.TraceRecorder is TraceRecorder
