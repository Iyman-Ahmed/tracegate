# TraceGate

**The black box flight recorder for AI agents** — record every step, detect silent failures, gate your deploys.

An agent that is 95% reliable per step is only **~59% reliable across a 10-step workflow**. Worse: the failures that matter never throw. The agent drops a constraint, invents a fact, loops quietly, ignores a tool error — and still says "Task complete." TraceGate is the missing pipeline for exactly that problem:

- **Record** — a drop-in recorder that captures every LLM call and tool call into a normalized `AgentTrace`. Local-first: traces land in JSONL/SQLite on your machine, no server, no data leaves.
- **Detect** — analyzers that read traces and pinpoint *which step* went wrong: deterministic checks (loops, tool misuse) for free, LLM-judge checks (goal drift, ungrounded assumptions) where semantics demand it.
- **Gate** — replayable regression suites with a reliability score. `tracegate ci` fails your build when quality drops. Pytest for agents.

## See it: the compounding-error demo

```bash
python examples/compounding_demo.py
```

```text
10-step agent, 95% reliable per step, 10 runs
expected failure rate: 40%

run   agent says         reality                  traceGate verdict
1     "Task complete."   actually succeeded       no findings
3     "Task complete."   silently died at step 0  [error] step 0 loop: identical step repeated 3x
8     "Task complete."   silently died at step 3  [error] step 3 loop: identical step repeated 3x
10    "Task complete."   silently died at step 8  [error] step 8 loop: identical step repeated 3x

4/10 runs failed silently — every one pinned to its originating step.
```

Every run *claims* success. TraceGate reads the trace and pins each silent failure to the step that caused it.

## Install

```bash
pip install -e .            # core: recorder, stores, detectors, CLI
pip install -e ".[judge]"   # + LLM-judge detectors (goal drift, ungrounded assumptions)
```

## Layer 1 — Record

Any Python agent, manually:

```python
from tracegate import TraceRecorder

with TraceRecorder(task="user's task", framework="my-agent") as rec:
    rec.record_llm_call(model="claude-sonnet-5", prompt="...", response="...")
    rec.record_tool_call(tool_name="search", arguments={"q": "..."}, result="...")
```

Claude Agent SDK, streaming — two lines around your existing loop:

```python
from claude_agent_sdk import query
from tracegate import TraceRecorder
from tracegate.adapters.claude_agent_sdk import record_stream

recorder = TraceRecorder(task=prompt, framework="claude-agent-sdk")
async for message in record_stream(query(prompt=prompt), recorder):
    ...  # use messages exactly as before; the trace saves itself
```

LangGraph — wrap the stream:

```python
from tracegate import TraceRecorder
from tracegate.adapters.langgraph import record_langgraph_stream

recorder = TraceRecorder(task=task, framework="langgraph")
for update in record_langgraph_stream(graph.stream(inputs), recorder):
    ...  # use updates exactly as before
```

OpenAI Agents SDK — record a completed run:

```python
from tracegate import TraceRecorder
from tracegate.adapters.openai_agents import record_run

recorder = TraceRecorder(task=task, framework="openai-agents")
result = await Runner.run(agent, task)
record_run(result, recorder)
```

All adapters are duck-typed — TraceGate has **zero** hard framework dependencies.

In pytest — a `trace_recorder` fixture registers automatically when tracegate is installed:

```python
def test_my_agent(trace_recorder):
    trace_recorder.record_llm_call(model="...", prompt="...", response="...")
    # trace saves on teardown; inspect it later with `tracegate show`
```

Or wrap any script without touching its code:

```bash
tracegate record my_agent_script.py    # sets TRACEGATE_DIR, runs it, stores traces
tracegate list                         # list recorded traces
tracegate show <trace_id>              # step-by-step timeline
```

Traces land in `.tracegate/traces.jsonl` (override with `TRACEGATE_DIR`).

## Layer 2 — Detect

```bash
tracegate detect <trace_id>            # deterministic detectors (free, instant)
tracegate detect <trace_id> --judge    # + LLM-judge detectors (cached by trace content)
```

| Detector | Catches | Method |
|----------|---------|--------|
| `loop` | Identical steps repeated, budget burn without progress | deterministic |
| `tool_misuse` | Ignored tool errors, blind identical retries | deterministic |
| `contradiction` | Acting on information a later tool result contradicted | deterministic |
| `goal_drift` | Dropped constraints, diverging objective | LLM judge |
| `ungrounded_assumption` | Silent hallucinations — claims with no supporting evidence | LLM judge |

The full failure-mode reference: [The Silent-Failure Taxonomy](docs/silent-failure-taxonomy.md).

**A reliability tool must prove its own reliability.** The detectors ship with a labeled failure-injection corpus and are scored on it (`tracegate eval`):

```text
corpus: 14 labeled traces
detector         precision  recall
loop                  1.00    1.00
tool_misuse           1.00    1.00
contradiction         1.00    1.00
```

Judge-based detectors are evaluated against the same corpus when an API key is present; judge calls are cached by prompt hash so CI reruns cost nothing.

## Layer 3 — Gate

Describe your regression suite in TOML:

```toml
[suite]
name = "booking agent"
runs_per_case = 3        # replay is non-deterministic: score outcomes, pass@k-style
threshold = 0.9

[[case]]
id = "cheap-flight"
task = "Find the cheapest flight SFO->NYC under $500"
expect_contains = ["$420"]
```

Point TraceGate at any function that runs your agent for a task and returns the trace:

```bash
tracegate run --suite suite.toml --runner myproject.agent:run_task --report report.html
tracegate ci  --suite suite.toml --runner myproject.agent:run_task   # exit 1 on drop
```

The reliability score combines task success rates with detector findings; `ci` fails below `--threshold` or a stored `--baseline`. In GitHub Actions:

```yaml
- uses: Iyman-Ahmed/tracegate@main
  with:
    suite: tests/agent_suite.toml
    runner: myproject.agent:run_task
    threshold: "0.9"
```

Try it locally with the bundled example:

```bash
tracegate ci --suite examples/suite.toml --runner examples/suite_runner.py:run_agent
```

## What TraceGate is not

Not a hosted dashboard (local + CI first). Not an agent framework (it instruments yours). Not a security scanner (it catches *accidents*, not attacks). Not a single-call eval library (multi-step workflows only).

## Status & roadmap

All five phases of the [proposal](PROPOSAL.md) are implemented: schema, recorder, adapters (Claude Agent SDK, LangGraph, OpenAI Agents SDK), JSONL/SQLite stores, all five detectors with a labeled eval corpus, replay suites, `tracegate ci`, HTML reports, GitHub Action, pytest plugin. See [docs/STATUS.md](docs/STATUS.md) for the build state and design decisions. Next: benchmark-trace findings, semantic contradiction via judge.

## Development

```bash
python -m venv venv && venv/bin/pip install -e ".[dev]"
venv/bin/pytest
```
