# TraceGate

**The black box flight recorder for AI agents** — record every step, detect silent failures, gate your deploys.

An agent that is 95% reliable per step is only ~59% reliable across a 10-step workflow. TraceGate records what your agent actually did, step by step, so you can see where runs go wrong — and (coming next) detect silent failures and gate deploys in CI.

## Status

Phase 1 (core spine): **AgentTrace schema, recorder, Claude Agent SDK adapter, local JSONL/SQLite stores, CLI**. Detectors and CI gate are next — see [PROPOSAL.md](PROPOSAL.md) for the full roadmap.

## Quickstart

```bash
pip install -e .
tracegate record examples/demo_agent.py   # run a script with recording enabled
tracegate list                            # list recorded traces
tracegate show <trace_id>                 # step-by-step timeline of one run
```

## Recording your own agent

Any Python agent, manually:

```python
from tracegate import TraceRecorder

with TraceRecorder(task="user's task", framework="my-agent") as rec:
    rec.record_llm_call(model="claude-sonnet-5", prompt="...", response="...")
    rec.record_tool_call(tool_name="search", arguments={"q": "..."}, result="...")
```

Claude Agent SDK, streaming:

```python
from claude_agent_sdk import query
from tracegate import TraceRecorder
from tracegate.adapters.claude_agent_sdk import record_stream

recorder = TraceRecorder(task=prompt, framework="claude-agent-sdk")
async for message in record_stream(query(prompt=prompt), recorder):
    ...  # use messages exactly as before; the trace saves itself
```

Traces land in `.tracegate/traces.jsonl` (override with `TRACEGATE_DIR`). Local-first: nothing leaves your machine.
