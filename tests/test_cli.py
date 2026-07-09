from typer.testing import CliRunner

from tracegate.cli import app
from tracegate.recorder import TraceRecorder
from tracegate.store.jsonl import JSONLTraceStore

runner = CliRunner()


def seed_trace(tmp_path):
    store = JSONLTraceStore(tmp_path / "traces.jsonl")
    rec = TraceRecorder(task="book flight", framework="test", store=store)
    rec.record_llm_call(model="m", prompt="p", response="thinking")
    rec.record_tool_call(tool_name="search", arguments={"to": "NYC"}, result="ok")
    return rec.finish()


def test_list_shows_traces(tmp_path):
    trace = seed_trace(tmp_path)
    result = runner.invoke(app, ["list", "--store-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert trace.trace_id in result.output
    assert "book flight" in result.output


def test_list_empty_store(tmp_path):
    result = runner.invoke(app, ["list", "--store-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no traces" in result.output.lower()


def test_show_renders_steps(tmp_path):
    trace = seed_trace(tmp_path)
    result = runner.invoke(app, ["show", trace.trace_id, "--store-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "book flight" in result.output
    assert "llm_call" in result.output
    assert "tool_call" in result.output
    assert "search" in result.output


def test_show_missing_trace_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["show", "nope", "--store-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "trace not found" in result.output
