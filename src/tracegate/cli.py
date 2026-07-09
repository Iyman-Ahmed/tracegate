"""TraceGate CLI: record, show, list."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

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
