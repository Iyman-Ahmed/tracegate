# AgentGates Phase 3 (Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build AgentGates's Gate layer: TOML replay suites, a runner that re-executes an agent against suite tasks and scores outcomes (pass@k-style success rates + detector-finding penalties), baseline save/compare, terminal + static HTML reports, `agentgates run` / `agentgates ci` (exit non-zero on threshold/baseline drop), and a composite GitHub Action.

**Architecture:** A suite is a TOML file of cases (task + expected-outcome substrings). The user supplies a runner — `module:function` or `path/file.py:function` — that takes a task string and returns an `AgentTrace` (usually via `TraceRecorder`). `run_suite` executes N runs per case, scores outcomes (not paths — replay is non-deterministic), runs deterministic detectors on every trace, and computes a reliability score = mean success rate × (1 − min(0.5, 0.1 × error-findings-per-run)). `agentgates ci` fails when the score is below `--threshold` or below a stored baseline. Reports are stdlib-generated (f-strings + `html.escape`) — deviating from the proposal's Jinja to keep zero new required deps, per the local-first principle.

**Tech Stack:** stdlib `tomllib` (Python 3.11+) for suites, `importlib` for runner resolution, no new dependencies.

## Global Constraints

- No new required or optional deps. TOML read via `tomllib`; HTML via stdlib.
- Suite schema: `[suite]` table (`name`, `runs_per_case` default 1, optional `threshold`) + `[[case]]` array (`id`, `task`, optional `constraints`, `expect_contains`, `runs`).
- Runner contract: callable `(task: str) -> AgentTrace`; exceptions in the runner count as failed runs, never crash the suite.
- Reliability score formula (exact): `0.0` for an empty suite; else `mean(case success_rate) * (1 - min(0.5, 0.1 * total_error_findings / total_runs))`.
- Branch `phases-2-5`; `venv/bin/pytest`; commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Suite loading (TOML)

**Files:**
- Create: `src/agentgates/gate/__init__.py`
- Create: `src/agentgates/gate/suite.py`
- Test: `tests/test_gate_suite.py`

**Interfaces:**
- Produces (`agentgates.gate.suite`): `SuiteCase(id: str, task: str, constraints: list[str] = [], expect_contains: list[str] = [], runs: int | None = None)`, `Suite(name: str = "suite", runs_per_case: int = 1, threshold: float | None = None, cases: list[SuiteCase])`, `load_suite(path: Path) -> Suite`.

- [ ] **Step 1: failing tests** — `tests/test_gate_suite.py`:

```python
import pytest
from pydantic import ValidationError

from agentgates.gate.suite import Suite, load_suite

SUITE_TOML = """
[suite]
name = "booking agent"
runs_per_case = 2
threshold = 0.8

[[case]]
id = "cheap-flight"
task = "Find the cheapest flight under $500"
constraints = ["under $500"]
expect_contains = ["$420"]

[[case]]
id = "refund"
task = "Process a refund"
runs = 3
"""


def test_load_suite(tmp_path):
    path = tmp_path / "suite.toml"
    path.write_text(SUITE_TOML)
    suite = load_suite(path)
    assert suite.name == "booking agent"
    assert suite.runs_per_case == 2
    assert suite.threshold == 0.8
    assert len(suite.cases) == 2
    assert suite.cases[0].expect_contains == ["$420"]
    assert suite.cases[0].runs is None
    assert suite.cases[1].runs == 3
    assert suite.cases[1].expect_contains == []


def test_defaults(tmp_path):
    path = tmp_path / "suite.toml"
    path.write_text('[[case]]\nid = "a"\ntask = "do a"\n')
    suite = load_suite(path)
    assert suite.name == "suite"
    assert suite.runs_per_case == 1
    assert suite.threshold is None


def test_case_requires_id_and_task():
    with pytest.raises(ValidationError):
        Suite(cases=[{"id": "x"}])
```

- [ ] **Step 2: verify fail** — ModuleNotFoundError
- [ ] **Step 3: implement** — `src/agentgates/gate/__init__.py`:

```python
"""Gate layer: replayable regression suites with a reliability score."""
```

`src/agentgates/gate/suite.py`:

```python
"""TOML replay-suite definition."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class SuiteCase(BaseModel):
    id: str
    task: str
    constraints: list[str] = Field(default_factory=list)
    expect_contains: list[str] = Field(default_factory=list)
    runs: int | None = None


class Suite(BaseModel):
    name: str = "suite"
    runs_per_case: int = 1
    threshold: float | None = None
    cases: list[SuiteCase]


def load_suite(path: Path) -> Suite:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    meta = data.get("suite", {})
    return Suite(**meta, cases=data.get("case", []))
```

- [ ] **Step 4: verify pass**; **Step 5: commit** — `feat: TOML replay suite definition`

---

### Task 2: Scoring models

**Files:**
- Create: `src/agentgates/gate/score.py`
- Test: `tests/test_gate_score.py`

**Interfaces:**
- Produces: `RunResult(success: bool, findings: list[Finding] = [], error: str | None = None)`; `CaseResult(case_id: str, runs: list[RunResult])` with properties `success_rate` (0.0 for no runs) and `error_finding_count` (total error-severity findings across runs); `SuiteResult(suite_name: str, cases: list[CaseResult])` with property `reliability_score` per the Global Constraints formula and `total_runs`.

- [ ] **Step 1: failing tests** — `tests/test_gate_score.py`:

```python
from agentgates.detect import Finding
from agentgates.gate.score import CaseResult, RunResult, SuiteResult


def err(msg="e"):
    return Finding(detector="loop", severity="error", message=msg)


def warn():
    return Finding(detector="loop", severity="warning", message="w")


def test_case_success_rate():
    case = CaseResult(case_id="a", runs=[RunResult(success=True), RunResult(success=False)])
    assert case.success_rate == 0.5


def test_perfect_suite_scores_one():
    result = SuiteResult(
        suite_name="s",
        cases=[CaseResult(case_id="a", runs=[RunResult(success=True)])],
    )
    assert result.reliability_score == 1.0


def test_error_findings_penalize():
    # 2 cases: one perfect, one failed with 2 error findings in its single run.
    result = SuiteResult(
        suite_name="s",
        cases=[
            CaseResult(case_id="a", runs=[RunResult(success=True)]),
            CaseResult(case_id="b", runs=[RunResult(success=False, findings=[err(), err()])]),
        ],
    )
    # success mean 0.5; errors/run = 2/2 = 1.0; penalty 0.1 -> 0.5 * 0.9
    assert abs(result.reliability_score - 0.45) < 1e-9


def test_warnings_do_not_penalize():
    result = SuiteResult(
        suite_name="s",
        cases=[CaseResult(case_id="a", runs=[RunResult(success=True, findings=[warn()])])],
    )
    assert result.reliability_score == 1.0


def test_penalty_capped_at_half():
    findings = [err() for _ in range(20)]
    result = SuiteResult(
        suite_name="s",
        cases=[CaseResult(case_id="a", runs=[RunResult(success=True, findings=findings)])],
    )
    assert abs(result.reliability_score - 0.5) < 1e-9


def test_empty_suite_scores_zero():
    assert SuiteResult(suite_name="s", cases=[]).reliability_score == 0.0
```

- [ ] **Step 2: verify fail**; **Step 3: implement** — `src/agentgates/gate/score.py`:

```python
"""Outcome scoring: pass@k-style success rates + detector penalties."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentgates.detect import Finding


class RunResult(BaseModel):
    success: bool
    findings: list[Finding] = Field(default_factory=list)
    error: str | None = None


class CaseResult(BaseModel):
    case_id: str
    runs: list[RunResult]

    @property
    def success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.success) / len(self.runs)

    @property
    def error_finding_count(self) -> int:
        return sum(
            1 for r in self.runs for f in r.findings if f.severity == "error"
        )


class SuiteResult(BaseModel):
    suite_name: str
    cases: list[CaseResult]

    @property
    def total_runs(self) -> int:
        return sum(len(c.runs) for c in self.cases)

    @property
    def reliability_score(self) -> float:
        if not self.cases:
            return 0.0
        success_mean = sum(c.success_rate for c in self.cases) / len(self.cases)
        total_runs = self.total_runs
        errors_per_run = (
            sum(c.error_finding_count for c in self.cases) / total_runs
            if total_runs
            else 0.0
        )
        penalty = min(0.5, 0.1 * errors_per_run)
        return success_mean * (1 - penalty)
```

- [ ] **Step 4: verify pass**; **Step 5: commit** — `feat: reliability scoring models`

---

### Task 3: Runner resolution + suite execution

**Files:**
- Create: `src/agentgates/gate/runner.py`
- Create: `tests/fake_runner.py` (test fixture module, not a test file)
- Test: `tests/test_gate_runner.py`

**Interfaces:**
- Produces (`agentgates.gate.runner`):
  - `resolve_runner(spec: str) -> Callable[[str], AgentTrace]` — `"module:function"` (imports module, with cwd prepended to `sys.path` if missing) or `"path/to/file.py:function"` (spec_from_file_location). `ValueError` on missing `:`.
  - `run_suite(suite: Suite, runner, detectors: list | None = None) -> SuiteResult` — detectors default `default_detectors()`; runner exceptions → `RunResult(success=False, error="Type: msg")`; outcome text = `str(trace.metadata["result"])` if truthy else last `LLMCallStep.response` else `""`; success = every `expect_contains` substring in outcome.

- [ ] **Step 1: fixture + failing tests** — `tests/fake_runner.py`:

```python
"""Importable fake agent runners for gate tests (not a test module)."""

from agentgates.recorder import TraceRecorder


class _NullStore:
    def save(self, trace):
        pass


def good_runner(task: str):
    rec = TraceRecorder(task=task, framework="fake", store=_NullStore())
    rec.record_llm_call(model="m", prompt=task, response=f"done: {task} SUCCESS_TOKEN")
    return rec.finish()


def loopy_runner(task: str):
    rec = TraceRecorder(task=task, framework="fake", store=_NullStore())
    for _ in range(3):
        rec.record_tool_call(tool_name="poll", arguments={"t": task}, error="timeout")
    rec.record_llm_call(model="m", prompt="", response="gave up")
    return rec.finish()


def metadata_runner(task: str):
    rec = TraceRecorder(task=task, framework="fake", store=_NullStore())
    rec.trace.metadata["result"] = "final answer: 42"
    return rec.finish()


def crashing_runner(task: str):
    raise RuntimeError("agent exploded")
```

`tests/test_gate_runner.py`:

```python
import pytest

from agentgates.gate.runner import resolve_runner, run_suite
from agentgates.gate.suite import Suite, SuiteCase


def suite_of(case: SuiteCase, runs_per_case: int = 1) -> Suite:
    return Suite(name="t", runs_per_case=runs_per_case, cases=[case])


def test_resolve_runner_module_spec():
    fn = resolve_runner("tests.fake_runner:good_runner")
    assert fn("x").steps[0].response.endswith("SUCCESS_TOKEN")


def test_resolve_runner_file_spec(tmp_path):
    script = tmp_path / "my_runner.py"
    script.write_text(
        "from tests.fake_runner import good_runner\n"
        "run = good_runner\n"
    )
    fn = resolve_runner(f"{script}:run")
    assert fn("x").steps


def test_resolve_runner_bad_spec():
    with pytest.raises(ValueError):
        resolve_runner("no_colon_here")


def test_run_suite_success():
    suite = suite_of(SuiteCase(id="a", task="t", expect_contains=["SUCCESS_TOKEN"]))
    result = run_suite(suite, resolve_runner("tests.fake_runner:good_runner"))
    assert result.reliability_score == 1.0
    assert result.cases[0].success_rate == 1.0


def test_run_suite_missing_expectation_fails():
    suite = suite_of(SuiteCase(id="a", task="t", expect_contains=["NOPE"]))
    result = run_suite(suite, resolve_runner("tests.fake_runner:good_runner"))
    assert result.cases[0].success_rate == 0.0


def test_run_suite_metadata_result_wins():
    suite = suite_of(SuiteCase(id="a", task="t", expect_contains=["42"]))
    result = run_suite(suite, resolve_runner("tests.fake_runner:metadata_runner"))
    assert result.cases[0].success_rate == 1.0


def test_run_suite_detector_findings_attached():
    suite = suite_of(SuiteCase(id="a", task="t"))
    result = run_suite(suite, resolve_runner("tests.fake_runner:loopy_runner"))
    case = result.cases[0]
    assert case.success_rate == 1.0  # no expectations -> outcome passes
    assert case.error_finding_count >= 1  # blind retries flagged
    assert result.reliability_score < 1.0


def test_run_suite_crash_is_failed_run_not_exception():
    suite = suite_of(SuiteCase(id="a", task="t"))
    result = run_suite(suite, resolve_runner("tests.fake_runner:crashing_runner"))
    run = result.cases[0].runs[0]
    assert run.success is False
    assert "RuntimeError" in run.error


def test_runs_per_case_respected():
    suite = suite_of(SuiteCase(id="a", task="t"), runs_per_case=3)
    result = run_suite(suite, resolve_runner("tests.fake_runner:good_runner"))
    assert len(result.cases[0].runs) == 3


def test_case_runs_overrides_suite_default():
    suite = suite_of(SuiteCase(id="a", task="t", runs=2), runs_per_case=5)
    result = run_suite(suite, resolve_runner("tests.fake_runner:good_runner"))
    assert len(result.cases[0].runs) == 2
```

- [ ] **Step 2: verify fail**; **Step 3: implement** — `src/agentgates/gate/runner.py`:

```python
"""Resolve user agent runners and replay suites against them."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable

from agentgates.detect import default_detectors, run_detectors
from agentgates.gate.score import CaseResult, RunResult, SuiteResult
from agentgates.gate.suite import Suite
from agentgates.schema import AgentTrace, LLMCallStep

Runner = Callable[[str], AgentTrace]


def resolve_runner(spec: str) -> Runner:
    target, sep, attr = spec.rpartition(":")
    if not sep or not target:
        raise ValueError(f"runner must be 'module:function' or 'file.py:function', got {spec!r}")
    if target.endswith(".py"):
        path = Path(target)
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        module = importlib.import_module(target)
    return getattr(module, attr)


def _outcome_text(trace: AgentTrace) -> str:
    result = trace.metadata.get("result")
    if result:
        return str(result)
    for step in reversed(trace.steps):
        if isinstance(step, LLMCallStep):
            return step.response
    return ""


def run_suite(suite: Suite, runner: Runner, detectors: list | None = None) -> SuiteResult:
    if detectors is None:
        detectors = default_detectors()
    case_results: list[CaseResult] = []
    for case in suite.cases:
        runs: list[RunResult] = []
        for _ in range(case.runs or suite.runs_per_case):
            try:
                trace = runner(case.task)
            except Exception as exc:  # runner bugs must not kill the gate
                runs.append(
                    RunResult(success=False, error=f"{type(exc).__name__}: {exc}")
                )
                continue
            findings = run_detectors(trace, detectors)
            outcome = _outcome_text(trace)
            success = all(expected in outcome for expected in case.expect_contains)
            runs.append(RunResult(success=success, findings=findings))
        case_results.append(CaseResult(case_id=case.id, runs=runs))
    return SuiteResult(suite_name=suite.name, cases=case_results)
```

- [ ] **Step 4: verify pass** (10 tests); **Step 5: commit** — `feat: runner resolution and suite replay engine`

---

### Task 4: Baseline + reports (terminal & HTML)

**Files:**
- Create: `src/agentgates/gate/baseline.py`
- Create: `src/agentgates/report.py`
- Test: `tests/test_gate_baseline.py`, `tests/test_report.py`

**Interfaces:**
- `save_baseline(result: SuiteResult, path: Path) -> None` — JSON `{"reliability_score": float, "cases": {case_id: success_rate}}`; `load_baseline(path) -> dict`.
- `render_terminal(result: SuiteResult) -> str` — one line per case (`id  passed n/m  findings: e error/w warning`) + `reliability score: X.XX`.
- `render_html(result: SuiteResult) -> str` — self-contained page (`<html`, suite name, per-case rows, score, findings list, html-escaped).

- [ ] **Step 1: failing tests** — `tests/test_gate_baseline.py`:

```python
import json

from agentgates.gate.baseline import load_baseline, save_baseline
from agentgates.gate.score import CaseResult, RunResult, SuiteResult


def result_fixture() -> SuiteResult:
    return SuiteResult(
        suite_name="s",
        cases=[CaseResult(case_id="a", runs=[RunResult(success=True)])],
    )


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    save_baseline(result_fixture(), path)
    data = load_baseline(path)
    assert data["reliability_score"] == 1.0
    assert data["cases"] == {"a": 1.0}
    assert json.loads(path.read_text())  # valid json on disk
```

`tests/test_report.py`:

```python
from agentgates.detect import Finding
from agentgates.gate.score import CaseResult, RunResult, SuiteResult
from agentgates.report import render_html, render_terminal


def result_fixture() -> SuiteResult:
    finding = Finding(detector="loop", severity="error", message="<looped>", step_index=0)
    return SuiteResult(
        suite_name="my suite",
        cases=[
            CaseResult(case_id="good", runs=[RunResult(success=True)]),
            CaseResult(case_id="bad", runs=[RunResult(success=False, findings=[finding])]),
        ],
    )


def test_terminal_report():
    text = render_terminal(result_fixture())
    assert "good" in text
    assert "bad" in text
    assert "1/1" in text
    assert "reliability score" in text.lower()


def test_html_report_escapes_and_includes_score():
    html = render_html(result_fixture())
    assert html.startswith("<!doctype html>")
    assert "my suite" in html
    assert "&lt;looped&gt;" in html   # escaped finding message
    assert "<looped>" not in html
    assert "good" in html and "bad" in html
```

- [ ] **Step 2: verify fail**; **Step 3: implement** — `src/agentgates/gate/baseline.py`:

```python
"""Persisted reliability baselines for regression comparison."""

from __future__ import annotations

import json
from pathlib import Path

from agentgates.gate.score import SuiteResult


def save_baseline(result: SuiteResult, path: Path) -> None:
    payload = {
        "reliability_score": result.reliability_score,
        "cases": {c.case_id: c.success_rate for c in result.cases},
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_baseline(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
```

`src/agentgates/report.py`:

```python
"""Terminal and static-HTML reports for suite results (stdlib only)."""

from __future__ import annotations

from html import escape

from agentgates.gate.score import CaseResult, SuiteResult


def _finding_counts(case: CaseResult) -> tuple[int, int]:
    errors = warnings = 0
    for run in case.runs:
        for f in run.findings:
            if f.severity == "error":
                errors += 1
            elif f.severity == "warning":
                warnings += 1
    return errors, warnings


def render_terminal(result: SuiteResult) -> str:
    lines = [f"suite: {result.suite_name}"]
    for case in result.cases:
        passed = sum(1 for r in case.runs if r.success)
        errors, warnings = _finding_counts(case)
        lines.append(
            f"  {case.case_id:<24} passed {passed}/{len(case.runs)}"
            f"  findings: {errors} error / {warnings} warning"
        )
    lines.append(f"reliability score: {result.reliability_score:.2f}")
    return "\n".join(lines)


def render_html(result: SuiteResult) -> str:
    rows = []
    for case in result.cases:
        passed = sum(1 for r in case.runs if r.success)
        errors, warnings = _finding_counts(case)
        findings_html = "".join(
            f"<li class='{escape(f.severity)}'>[{escape(f.severity)}] "
            f"step {f.step_index if f.step_index is not None else '-'} "
            f"{escape(f.detector)}: {escape(f.message)}</li>"
            for run in case.runs
            for f in run.findings
        )
        run_errors = "".join(
            f"<li class='error'>run error: {escape(r.error)}</li>"
            for r in case.runs
            if r.error
        )
        rows.append(
            f"<tr><td>{escape(case.case_id)}</td>"
            f"<td>{passed}/{len(case.runs)}</td>"
            f"<td>{errors} error / {warnings} warning</td>"
            f"<td><ul>{findings_html}{run_errors}</ul></td></tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AgentGates report: {escape(result.suite_name)}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: .5rem; text-align: left; vertical-align: top; }}
.score {{ font-size: 1.4rem; margin: 1rem 0; }}
li.error {{ color: #b00020; }}
li.warning {{ color: #8a6d00; }}
ul {{ margin: 0; padding-left: 1.2rem; }}
</style></head><body>
<h1>AgentGates &mdash; {escape(result.suite_name)}</h1>
<p class="score">reliability score: <strong>{result.reliability_score:.2f}</strong>
 ({result.total_runs} runs)</p>
<table><tr><th>case</th><th>passed</th><th>findings</th><th>detail</th></tr>
{''.join(rows)}
</table></body></html>
"""
```

- [ ] **Step 4: verify pass**; **Step 5: commit** — `feat: baselines and terminal/HTML reports`

---

### Task 5: CLI `run` + `ci`, GitHub Action, example suite

**Files:**
- Modify: `src/agentgates/cli.py`
- Create: `action.yml`
- Create: `examples/suite.toml`, `examples/suite_runner.py`
- Test: `tests/test_cli_gate.py`

**Interfaces:**
- `agentgates run --suite PATH --runner SPEC [--report out.html]` — prints terminal report; writes HTML when asked; exit 0.
- `agentgates ci --suite PATH --runner SPEC [--threshold F] [--baseline PATH] [--update-baseline] [--report out.html]` — threshold falls back to suite's `threshold`; fails (exit 1) when score < threshold or score < stored baseline score − 1e-9; `--update-baseline` writes the baseline after the run (and skips the compare).
- `action.yml` — composite action: inputs `suite`, `runner`, `threshold` (optional), `install` (default `.`), `python-version` (default `3.12`); pip-installs agentgates + the project, runs `agentgates ci`.
- `examples/suite.toml` + `examples/suite_runner.py` — runnable demo (`agentgates ci --suite examples/suite.toml --runner examples/suite_runner.py:run_agent --threshold 0.9` passes).

- [ ] **Step 1: failing tests** — `tests/test_cli_gate.py`:

```python
from typer.testing import CliRunner

from agentgates.cli import app

runner = CliRunner()

SUITE = """
[suite]
name = "demo"

[[case]]
id = "ok"
task = "do the thing"
expect_contains = ["SUCCESS_TOKEN"]
"""

FAILING_SUITE = SUITE.replace("SUCCESS_TOKEN", "NEVER_THERE")

RUNNER = "tests.fake_runner:good_runner"


def write_suite(tmp_path, content=SUITE):
    path = tmp_path / "suite.toml"
    path.write_text(content)
    return path


def test_run_prints_report(tmp_path):
    result = runner.invoke(app, ["run", "--suite", str(write_suite(tmp_path)), "--runner", RUNNER])
    assert result.exit_code == 0
    assert "reliability score: 1.00" in result.output


def test_run_writes_html_report(tmp_path):
    out = tmp_path / "report.html"
    result = runner.invoke(
        app,
        ["run", "--suite", str(write_suite(tmp_path)), "--runner", RUNNER, "--report", str(out)],
    )
    assert result.exit_code == 0
    assert out.exists()
    assert "<!doctype html>" in out.read_text()


def test_ci_passes_above_threshold(tmp_path):
    result = runner.invoke(
        app,
        ["ci", "--suite", str(write_suite(tmp_path)), "--runner", RUNNER, "--threshold", "0.9"],
    )
    assert result.exit_code == 0
    assert "gate passed" in result.output.lower()


def test_ci_fails_below_threshold(tmp_path):
    result = runner.invoke(
        app,
        ["ci", "--suite", str(write_suite(tmp_path, FAILING_SUITE)), "--runner", RUNNER, "--threshold", "0.9"],
    )
    assert result.exit_code == 1
    assert "below threshold" in result.output.lower()


def test_ci_uses_suite_threshold(tmp_path):
    content = '[suite]\nname = "demo"\nthreshold = 0.9\n' + FAILING_SUITE.split("[[case]]", 1)[1].join(["[[case]]", ""])
    path = tmp_path / "suite.toml"
    path.write_text('[suite]\nthreshold = 0.9\n\n[[case]]\nid = "ok"\ntask = "t"\nexpect_contains = ["NEVER"]\n')
    result = runner.invoke(app, ["ci", "--suite", str(path), "--runner", RUNNER])
    assert result.exit_code == 1


def test_ci_baseline_update_then_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    ok = write_suite(tmp_path)
    result = runner.invoke(
        app,
        ["ci", "--suite", str(ok), "--runner", RUNNER, "--baseline", str(baseline), "--update-baseline"],
    )
    assert result.exit_code == 0
    assert baseline.exists()

    bad = tmp_path / "bad.toml"
    bad.write_text(FAILING_SUITE)
    result = runner.invoke(
        app,
        ["ci", "--suite", str(bad), "--runner", RUNNER, "--baseline", str(baseline)],
    )
    assert result.exit_code == 1
    assert "baseline" in result.output.lower()
```

- [ ] **Step 2: verify fail** (usage error, exit 2)
- [ ] **Step 3: implement** — add to `src/agentgates/cli.py` (new imports: `load_suite`, `resolve_runner`, `run_suite`, `save_baseline`, `load_baseline`, `render_terminal`, `render_html`; `Optional` from typing if needed):

```python
def _run_gate(suite_path: Path, runner_spec: str):
    suite = load_suite(suite_path)
    fn = resolve_runner(runner_spec)
    return suite, run_suite(suite, fn)


@app.command("run")
def run_cmd(
    suite: Path = typer.Option(..., "--suite", help="Path to suite.toml"),
    runner: str = typer.Option(..., "--runner", help="module:function or file.py:function returning an AgentTrace"),
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
    threshold: float = typer.Option(None, "--threshold", help="Fail if reliability score is below this."),
    baseline: Path = typer.Option(None, "--baseline", help="Baseline JSON to compare/update."),
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
```

`action.yml`:

```yaml
name: "AgentGates CI"
description: "Replay a AgentGates suite against your agent and fail the build when the reliability score drops."
branding:
  icon: "shield"
  color: "orange"
inputs:
  suite:
    description: "Path to the suite TOML file"
    required: true
  runner:
    description: "Runner spec: module:function or file.py:function"
    required: true
  threshold:
    description: "Minimum reliability score (0-1)"
    required: false
    default: ""
  install:
    description: "pip install target for your agent project"
    required: false
    default: "."
  python-version:
    description: "Python version"
    required: false
    default: "3.12"
runs:
  using: "composite"
  steps:
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ inputs.python-version }}
    - run: pip install agentgates "${{ inputs.install }}"
      shell: bash
    - run: |
        ARGS=""
        if [ -n "${{ inputs.threshold }}" ]; then
          ARGS="--threshold ${{ inputs.threshold }}"
        fi
        agentgates ci --suite "${{ inputs.suite }}" --runner "${{ inputs.runner }}" $ARGS
      shell: bash
```

`examples/suite_runner.py`:

```python
"""Simulated agent runner for the example suite (no API key needed).

Run:  agentgates ci --suite examples/suite.toml --runner examples/suite_runner.py:run_agent --threshold 0.9
"""

from agentgates import TraceRecorder


class _NullStore:
    def save(self, trace):
        pass


def run_agent(task: str):
    rec = TraceRecorder(task=task, framework="demo", model="simulated", store=_NullStore())
    rec.record_llm_call(model="simulated", prompt=task, response="Searching for options.")
    rec.record_tool_call(
        tool_name="search",
        arguments={"q": task},
        result={"best": "UA at $420"},
        latency_ms=120.0,
    )
    rec.record_llm_call(model="simulated", prompt="", response=f"Task complete: {task}. Best option UA at $420.")
    return rec.finish()
```

`examples/suite.toml`:

```toml
[suite]
name = "demo agent suite"
runs_per_case = 2
threshold = 0.9

[[case]]
id = "cheap-flight"
task = "Find the cheapest flight SFO->NYC under $500"
constraints = ["under $500"]
expect_contains = ["$420"]

[[case]]
id = "task-completion"
task = "Book the best available option"
expect_contains = ["Task complete"]
```

- [ ] **Step 4: verify** — `venv/bin/pytest -q` all green, then end-to-end:
  `venv/bin/agentgates ci --suite examples/suite.toml --runner examples/suite_runner.py:run_agent` → "gate passed: score 1.00", exit 0.
- [ ] **Step 5: commit** — `feat: agentgates run/ci commands, GitHub Action, example suite`
