"""TraceGate CLI: record, show, list."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

from tracegate.detect import default_detectors, judge_detectors, run_detectors
from tracegate.detect.judge import CachedJudge, ClaudeJudge, OpenAICompatibleJudge
from tracegate.evals import build_corpus, build_judge_corpus, evaluate_detector
from tracegate.gate.baseline import load_baseline, save_baseline
from tracegate.gate.runner import resolve_runner, run_suite
from tracegate.gate.suite import load_suite
from tracegate.report import render_html, render_terminal
from tracegate.schema import LLMCallStep, ToolCallStep
from tracegate.store import TRACES_FILENAME
from tracegate.store.jsonl import JSONLTraceStore

app = typer.Typer(help="The black box flight recorder for AI agents.")

StoreDirOption = typer.Option(
    Path(".tracegate"), "--store-dir", envvar="TRACEGATE_DIR",
    help="Directory holding traces.jsonl.",
)


def _store(store_dir: Path) -> JSONLTraceStore:
    return JSONLTraceStore(store_dir / TRACES_FILENAME)


def _step_line(step) -> str:
    if isinstance(step, LLMCallStep):
        preview = step.response.replace("\n", " ")[:60]
        return f"  [{step.index}] llm_call    {step.model}  {preview!r}"
    if isinstance(step, ToolCallStep):
        status = f"ERROR: {step.error}" if step.error else "ok"
        latency = f"  {step.latency_ms:.0f}ms" if step.latency_ms is not None else ""
        return f"  [{step.index}] tool_call   {step.tool_name}({step.arguments})  {status}{latency}"
    return f"  [{step.index}] {step.type}"


@app.command("list")
def list_cmd(store_dir: Path = StoreDirOption) -> None:
    """List recorded traces."""
    traces = _store(store_dir).list_traces()
    if not traces:
        typer.echo(f"No traces found in {store_dir}.")
        return
    for trace in traces:
        typer.echo(
            f"{trace.trace_id}  {trace.started_at:%Y-%m-%d %H:%M:%S}"
            f"  {len(trace.steps):>3} steps  {trace.task.description}"
        )


@app.command()
def show(trace_id: str, store_dir: Path = StoreDirOption) -> None:
    """Show one trace as a step-by-step timeline."""
    try:
        trace = _store(store_dir).load(trace_id)
    except KeyError:
        typer.echo(f"trace not found: {trace_id}")
        raise typer.Exit(code=1)
    typer.echo(f"trace    {trace.trace_id}")
    typer.echo(f"task     {trace.task.description}")
    if trace.task.constraints:
        typer.echo(f"constraints  {', '.join(trace.task.constraints)}")
    agent = trace.agent.framework + (f" / {trace.agent.model}" if trace.agent.model else "")
    typer.echo(f"agent    {agent}")
    ended = f"{trace.ended_at:%Y-%m-%d %H:%M:%S}" if trace.ended_at else "-"
    typer.echo(f"time     {trace.started_at:%Y-%m-%d %H:%M:%S} -> {ended}")
    typer.echo(f"steps    {len(trace.steps)}")
    for step in trace.steps:
        typer.echo(_step_line(step))


def _run_gate(suite_path: Path, runner_spec: str):
    suite = load_suite(suite_path)
    fn = resolve_runner(runner_spec)
    return suite, run_suite(suite, fn)


@app.command("run")
def run_cmd(
    suite: Path = typer.Option(..., "--suite", help="Path to suite.toml"),
    runner: str = typer.Option(
        ..., "--runner",
        help="module:function or file.py:function returning an AgentTrace",
    ),
    report: Path = typer.Option(None, "--report", help="Write a static HTML report here."),
) -> None:
    """Replay a suite against your agent and print the score."""
    _, result = _run_gate(suite, runner)
    typer.echo(render_terminal(result))
    if report is not None:
        report.write_text(render_html(result), encoding="utf-8")
        typer.echo(f"report: {report}")


@app.command()
def ci(
    suite: Path = typer.Option(..., "--suite"),
    runner: str = typer.Option(..., "--runner"),
    threshold: float = typer.Option(
        None, "--threshold", help="Fail if reliability score is below this."
    ),
    baseline: Path = typer.Option(
        None, "--baseline", help="Baseline JSON to compare/update."
    ),
    update_baseline: bool = typer.Option(False, "--update-baseline"),
    report: Path = typer.Option(None, "--report"),
) -> None:
    """Run the suite and exit non-zero if reliability dropped."""
    suite_obj, result = _run_gate(suite, runner)
    typer.echo(render_terminal(result))
    if report is not None:
        report.write_text(render_html(result), encoding="utf-8")
    score = result.reliability_score
    failed = False
    effective = threshold if threshold is not None else suite_obj.threshold
    if effective is not None and score < effective:
        typer.echo(f"FAIL: score {score:.2f} below threshold {effective:.2f}")
        failed = True
    if baseline is not None and update_baseline:
        save_baseline(result, baseline)
        typer.echo(f"baseline updated: {baseline}")
    elif baseline is not None and baseline.exists():
        base_score = load_baseline(baseline)["reliability_score"]
        if score < base_score - 1e-9:
            typer.echo(f"FAIL: score {score:.2f} regressed from baseline {base_score:.2f}")
            failed = True
    if failed:
        raise typer.Exit(code=1)
    typer.echo(f"gate passed: score {score:.2f}")


JudgeUrlOption = typer.Option(
    None,
    "--judge-url",
    help="OpenAI-compatible endpoint for the judge (e.g. LM Studio http://localhost:1234/v1). Omit to use the Claude API.",
)
JudgeModelOption = typer.Option(
    None, "--judge-model", help="Model id for the judge backend."
)


def _build_judge(judge_url: str | None, judge_model: str | None, cache_path: Path):
    """Construct the judge backend the user asked for, wrapped in the disk cache."""
    if judge_url:
        if not judge_model:
            raise typer.BadParameter("--judge-model is required with --judge-url")
        inner = OpenAICompatibleJudge(model=judge_model, base_url=judge_url)
    else:
        inner = ClaudeJudge(model=judge_model or "claude-opus-4-8")
    return CachedJudge(inner, cache_path=cache_path)


@app.command()
def detect(
    trace_id: str,
    store_dir: Path = StoreDirOption,
    judge: bool = typer.Option(
        False,
        "--judge",
        help="Also run LLM-judge detectors (Claude API by default, or --judge-url for a local model).",
    ),
    judge_url: str = JudgeUrlOption,
    judge_model: str = JudgeModelOption,
) -> None:
    """Run failure detectors over one recorded trace."""
    try:
        trace = _store(store_dir).load(trace_id)
    except KeyError:
        typer.echo(f"trace not found: {trace_id}")
        raise typer.Exit(code=1)
    detectors = default_detectors()
    if judge:
        cached = _build_judge(judge_url, judge_model, store_dir / "judge_cache.json")
        detectors += judge_detectors(cached)
    findings = run_detectors(trace, detectors)
    if not findings:
        typer.echo("no findings")
        return
    for f in findings:
        step = f"step {f.step_index}" if f.step_index is not None else "trace"
        typer.echo(f"[{f.severity}] {step} {f.detector}: {f.message}")


def _print_scores(detectors, corpus) -> None:
    for detector in detectors:
        score = evaluate_detector(detector, corpus)
        typer.echo(f"{score.detector:<22} {score.precision:>9.2f} {score.recall:>7.2f}")


@app.command("eval")
def eval_cmd(
    judge: bool = typer.Option(
        False,
        "--judge",
        help="Also score the LLM-judge detectors (Claude API by default, or --judge-url for a local model).",
    ),
    judge_url: str = JudgeUrlOption,
    judge_model: str = JudgeModelOption,
    store_dir: Path = StoreDirOption,
) -> None:
    """Score the detectors against the built-in labeled corpora."""
    corpus = build_corpus()
    typer.echo(f"deterministic corpus: {len(corpus)} labeled traces")
    typer.echo(f"{'detector':<22} {'precision':>9} {'recall':>7}")
    _print_scores(default_detectors(), corpus)

    if not judge:
        return

    try:
        cached = _build_judge(judge_url, judge_model, store_dir / "eval_judge_cache.json")
    except ImportError as exc:
        typer.echo(f"\njudge eval unavailable: {exc}")
        raise typer.Exit(code=1)

    judge_corpus = build_judge_corpus()
    typer.echo(f"\njudge corpus: {len(judge_corpus)} labeled traces")
    typer.echo(f"{'detector':<22} {'precision':>9} {'recall':>7}")
    try:
        _print_scores(judge_detectors(cached), judge_corpus)
    except Exception as exc:  # network/auth failures shouldn't spew a traceback
        typer.echo(f"\njudge eval failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def record(
    ctx: typer.Context,
    script: Path,
    store_dir: Path = StoreDirOption,
) -> None:
    """Run a Python script with TraceGate recording enabled."""
    env = {**os.environ, "TRACEGATE_DIR": str(store_dir)}
    result = subprocess.run(
        [sys.executable, str(script), *ctx.args], env=env
    )
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
    count = len(_store(store_dir).list_traces())
    plural = "" if count == 1 else "s"
    typer.echo(f"{count} trace{plural} in {store_dir / TRACES_FILENAME}")
