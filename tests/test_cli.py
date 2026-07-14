import sys

from typer.testing import CliRunner

from agentgates.cli import app
from agentgates.recorder import TraceRecorder
from agentgates.store.jsonl import JSONLTraceStore

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


def test_record_runs_script_and_stores_trace(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text(
        "from agentgates import TraceRecorder\n"
        "with TraceRecorder(task='demo', framework='test') as rec:\n"
        "    rec.record_llm_call(model='m', prompt='p', response='r')\n"
    )
    store_dir = tmp_path / "store"
    result = runner.invoke(
        app, ["record", str(script), "--store-dir", str(store_dir)]
    )
    assert result.exit_code == 0
    assert (store_dir / "traces.jsonl").exists()
    assert "1 trace" in result.output


def test_record_propagates_failure_exit_code(tmp_path):
    script = tmp_path / "boom.py"
    script.write_text("import sys; sys.exit(3)\n")
    result = runner.invoke(
        app, ["record", str(script), "--store-dir", str(tmp_path / "store")]
    )
    assert result.exit_code == 3


def seed_loop_trace(tmp_path):
    store = JSONLTraceStore(tmp_path / "traces.jsonl")
    rec = TraceRecorder(task="looping", framework="test", store=store)
    for _ in range(3):
        rec.record_tool_call(tool_name="fetch", arguments={"u": 1}, result="pending")
    return rec.finish()


def test_detect_reports_findings(tmp_path):
    trace = seed_loop_trace(tmp_path)
    result = runner.invoke(app, ["detect", trace.trace_id, "--store-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "loop" in result.output
    assert "error" in result.output


def test_detect_clean_trace(tmp_path):
    trace = seed_trace(tmp_path)
    result = runner.invoke(app, ["detect", trace.trace_id, "--store-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no findings" in result.output.lower()


def test_detect_missing_trace(tmp_path):
    result = runner.invoke(app, ["detect", "nope", "--store-dir", str(tmp_path)])
    assert result.exit_code == 1


def test_eval_prints_scores():
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 0
    assert "loop" in result.output
    assert "tool_misuse" in result.output
    assert "contradiction" in result.output
    assert "1.00" in result.output


def test_eval_judge_without_dependency_fails_cleanly(monkeypatch):
    # Force the ImportError path so the test is deterministic whether or not
    # the anthropic package is installed: --judge should print a clear message
    # and exit 1, never a traceback.
    monkeypatch.setitem(sys.modules, "anthropic", None)
    result = runner.invoke(app, ["eval", "--judge"])
    assert result.exit_code == 1
    assert "judge" in result.output.lower()
    # deterministic scores still print before the judge attempt
    assert "loop" in result.output
