from agentgates.store.jsonl import JSONLTraceStore


def test_trace_recorder_fixture_records_and_saves(pytester, monkeypatch, tmp_path):
    store_dir = tmp_path / "traces"
    monkeypatch.setenv("AGENTGATES_DIR", str(store_dir))
    pytester.makepyfile(
        """
        def test_agent(trace_recorder):
            trace_recorder.record_llm_call(model="m", prompt="p", response="r")
            assert trace_recorder.trace.task.description == "test_agent"
        """
    )
    result = pytester.runpytest_inprocess()  # plugin auto-loads via pytest11 entry point
    result.assert_outcomes(passed=1)

    store = JSONLTraceStore(store_dir / "traces.jsonl")
    traces = store.list_traces()
    assert len(traces) == 1
    assert traces[0].task.description == "test_agent"
    assert traces[0].agent.framework == "pytest"
    assert traces[0].steps[0].response == "r"
    assert traces[0].ended_at is not None


def test_entry_point_registered():
    from importlib.metadata import entry_points

    names = [ep.name for ep in entry_points(group="pytest11")]
    assert "agentgates" in names
