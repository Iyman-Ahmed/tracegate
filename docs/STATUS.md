# AgentGates — Build Status & Handoff

> **Purpose:** what exists, why it's shaped this way, and what's left. [PROPOSAL.md](../PROPOSAL.md) is the vision; this document is the state of the code. Written 2026-07-10.

## Where things stand

All five roadmap phases from the proposal are implemented on branch `phases-2-5` (Phase 1 lives on `master`). The suite is green and every user-facing command has been exercised end-to-end.

| Phase | Proposal deliverable | Status |
|-------|----------------------|--------|
| 1 — Core spine | AgentTrace schema, recorder, Claude adapter, stores, CLI | done (on `master`) |
| 2 — Detectors | loop + tool-misuse (deterministic), ungrounded + drift (judge), eval corpus | done |
| 3 — Gate | replay engine, reliability scoring, `agentgates ci`, Action, HTML report | done |
| 4 — Launch | compounding-error demo, taxonomy post, README | assets done; posting is the owner's call |
| 5 — Ecosystem | LangGraph + OpenAI Agents adapters, pytest plugin | done |

Beyond the roadmap, the **contradiction** detector (the fifth failure mode) is now built too, so all five modes named in the taxonomy have a detector. Not yet built: the benchmark-trace findings post, and a judge-based pass for semantic contradictions that aren't exact-value conflicts.

## The shape of the code

```
src/agentgates/
  schema.py          AgentTrace — the real product. Everything else reads this.
  recorder.py        TraceRecorder: builds a trace, saves on finish/context exit.
  store/             jsonl.py (default) + sqlite.py, same save/load/list_traces interface.
  adapters/          claude_agent_sdk.py, langgraph.py, openai_agents.py
  detect/            findings.py, loop.py, tool_misuse.py, contradiction.py,
                     judge.py, goal_drift.py, ungrounded.py
  evals/             corpus.py (labeled failure injection), metrics.py (precision/recall)
  gate/              suite.py (TOML), runner.py (replay), score.py, baseline.py
  report.py          terminal + self-contained HTML
  cli.py             record, list, show, detect, eval, run, ci
  pytest_plugin.py   `trace_recorder` fixture via the pytest11 entry point
```

## Decisions worth knowing (and why)

**Adapters import nothing.** All three framework adapters dispatch on `type(obj).__name__` plus duck-typed attributes. AgentGates never imports `claude_agent_sdk`, `langgraph`, or `openai-agents`. This means zero hard framework dependencies, tests that need no API key or network, and — per the proposal's framework-churn risk — adapters thin enough to absorb SDK changes.

**Deterministic before LLM, everywhere.** `default_detectors()` returns only the free, instant checks. Judge detectors are opt-in (`agentgates detect --judge`, or `judge_detectors(judge)` in code). The `Judge` protocol is a one-method interface (`complete(prompt) -> str`), so tests inject fakes and production injects `ClaudeJudge` (model `claude-opus-4-8`, optional `anthropic` extra). `CachedJudge` keys on a SHA-256 of the prompt and persists to disk, so CI reruns cost nothing.

**The detectors are themselves evaluated.** `src/agentgates/evals/corpus.py` builds a labeled corpus of traces with injected failures; `agentgates eval` prints per-detector precision and recall. A reliability tool that can't measure its own reliability has no standing to gate anyone's deploy. All three deterministic detectors currently score 1.00/1.00 over 14 labeled traces, and the eval test iterates `default_detectors()` so a newly added detector cannot ship unmeasured. The two judge detectors have their own labeled corpus (`build_judge_corpus`) scored by `agentgates eval --judge`; the harness is validated in tests with scripted judges (a perfect judge → 1.00/1.00, a judge that misses one claim → recall 0.5, proving the metric rather than a tautology), while real numbers need a live model.

**The gate scores outcomes, never paths.** Agents don't rerun identically, so `run_suite` executes N runs per case (`runs_per_case`), checks `expect_contains` against the final outcome text, and reports a pass@k-style success rate. Reliability score = `mean(case success rate) × (1 − min(0.5, 0.1 × error findings per run))`. A crashing runner becomes a failed run, never an exception that kills the gate.

**Two deliberate deviations from the proposal.** (1) Reports are generated with stdlib f-strings and `html.escape` rather than Jinja — this keeps required runtime dependencies at exactly `pydantic` and `typer`, which matters more for a local-first dev tool than templating ergonomics. (2) Suites are TOML via stdlib `tomllib` rather than YAML, for the same no-dependency reason.

## Verifying it works

```bash
venv/bin/pytest                                    # 120 tests
venv/bin/agentgates eval                            # detector precision/recall
venv/bin/python examples/compounding_demo.py       # the flagship demo
venv/bin/agentgates ci --suite examples/suite.toml --runner examples/suite_runner.py:run_agent
```

## Published

Live at **https://github.com/Iyman-Ahmed/agentgates** (public), default branch `main`, with `phases-2-5` also pushed. Both branches point at the same commit, so `main` is the source of truth. The README's `uses: Iyman-Ahmed/agentgates@main` Action reference resolves.

## What to do next

1. **Benchmark findings** — run the detectors over published agent-benchmark traces (GAIA, SWE-bench) and write up how many failing runs contain a silent hallucination, and at which step. This is the research-flavored content the proposal identifies as citation bait.
2. **Semantic contradiction** — the current detector catches exact-value conflicts only. A judge pass would catch "the flight is cheap" against a $900 fare.
3. **Publish judge numbers** — the judge-eval harness and corpus exist (`agentgates eval --judge`); running it once against a live key and pasting the precision/recall into the README is all that's left to complete the eval story for all five detectors.
4. **PyPI** — the package is `pip install -e .` only. Publishing `agentgates` to PyPI makes the Action's `pip install agentgates` line work for external users.
