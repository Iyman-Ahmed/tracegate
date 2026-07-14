# AgentGates Phase 2 (Detectors) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build AgentGates's Detect layer: typed findings, deterministic loop + tool-misuse detectors, LLM-judge goal-drift + ungrounded-assumption detectors behind a model-agnostic cached Judge interface, a labeled failure-injection eval corpus with precision/recall metrics, and CLI `detect` + `eval` commands.

**Architecture:** Detectors implement `detect(trace) -> list[Finding]` with `.name`. Deterministic detectors run by default (zero cost); judge detectors take a `Judge` (protocol: `complete(prompt) -> str`) so tests use fakes and production uses `ClaudeJudge` (optional `anthropic` extra) wrapped in `CachedJudge` (prompt-hash cache, disk-persisted so CI reruns are cheap). The eval corpus is generated in code (`agentgates.evals`) — "eval the evaluator" per the proposal.

**Tech Stack:** Existing Phase 1 stack. New optional extra: `anthropic>=0.40` under `[project.optional-dependencies] judge`. Judge default model `claude-opus-4-8` via `client.messages.create` (no sampling params — they 400 on Opus 4.8).

## Global Constraints

- No new required runtime deps; `anthropic` is optional (`pip install "agentgates[judge]"`), imported lazily inside `ClaudeJudge.__init__` with a helpful ImportError.
- No network in tests: judge detectors tested with fakes; `ClaudeJudge` has no unit test (thin wrapper, exercised only when a real key is present).
- Work on branch `phases-2-5`; run `venv/bin/pytest` from the project root; commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Detector contract: attribute `name: str`, method `detect(trace: AgentTrace) -> list[Finding]`.

---

### Task 1: Findings model + loop detector

**Files:**
- Create: `src/agentgates/detect/__init__.py`
- Create: `src/agentgates/detect/findings.py`
- Create: `src/agentgates/detect/loop.py`
- Test: `tests/test_detect_loop.py`

**Interfaces:**
- Produces:
  - `Finding(detector: str, severity: Literal["info","warning","error"], message: str, step_index: int | None = None, confidence: float = 1.0)` — Pydantic model in `agentgates.detect.findings`; also exported from `agentgates.detect`.
  - `LoopDetector(threshold: int = 3)` with `name = "loop"`; error finding for >= threshold identical *consecutive* steps; warning for >= threshold identical tool calls anywhere (skipped if already reported consecutively).
  - `run_detectors(trace, detectors) -> list[Finding]` (sorted by step_index, None first) and `default_detectors() -> list` (loop + tool_misuse; tool_misuse added in Task 2 — in this task return `[LoopDetector()]`).
  - Step fingerprint helper `agentgates.detect.loop._fingerprint(step) -> str` (tool: name + sorted-args JSON; llm: response text).

- [ ] **Step 1: Write the failing tests** — `tests/test_detect_loop.py`:

```python
from agentgates.detect import Finding, run_detectors
from agentgates.detect.loop import LoopDetector
from agentgates.schema import AgentInfo, AgentTrace, LLMCallStep, TaskSpec, ToolCallStep


def make_trace(steps) -> AgentTrace:
    trace = AgentTrace(task=TaskSpec(description="t"), agent=AgentInfo(framework="test"))
    trace.steps = steps
    return trace


def tool(i, name="search", args=None):
    return ToolCallStep(index=i, tool_name=name, arguments=args or {"q": "x"}, result="ok")


def llm(i, response="thinking"):
    return LLMCallStep(index=i, model="m", prompt="", response=response)


def test_clean_trace_no_findings():
    trace = make_trace([llm(0, "a"), tool(1, "search"), llm(2, "b"), tool(3, "book", {"id": 1})])
    assert LoopDetector().detect(trace) == []


def test_consecutive_identical_steps_flagged_as_error():
    trace = make_trace([tool(0), tool(1), tool(2), llm(3, "done")])
    findings = LoopDetector(threshold=3).detect(trace)
    assert len(findings) == 1
    f = findings[0]
    assert f.detector == "loop"
    assert f.severity == "error"
    assert f.step_index == 0
    assert "3" in f.message


def test_below_threshold_not_flagged():
    trace = make_trace([tool(0), tool(1), llm(2, "done")])
    assert LoopDetector(threshold=3).detect(trace) == []


def test_identical_llm_responses_count_as_loop():
    trace = make_trace([llm(0, "same"), llm(1, "same"), llm(2, "same")])
    findings = LoopDetector().detect(trace)
    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_nonconsecutive_repeated_tool_calls_warned():
    trace = make_trace([tool(0), llm(1, "a"), tool(2), llm(3, "b"), tool(4)])
    findings = LoopDetector().detect(trace)
    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].detector == "loop"


def test_different_args_not_a_loop():
    trace = make_trace([tool(0, args={"q": 1}), tool(1, args={"q": 2}), tool(2, args={"q": 3})])
    assert LoopDetector().detect(trace) == []


def test_run_detectors_sorts_by_step_index():
    class A:
        name = "a"
        def detect(self, trace):
            return [Finding(detector="a", severity="info", message="x", step_index=5)]
    class B:
        name = "b"
        def detect(self, trace):
            return [Finding(detector="b", severity="info", message="y", step_index=1)]
    trace = make_trace([])
    findings = run_detectors(trace, [A(), B()])
    assert [f.step_index for f in findings] == [1, 5]
```

- [ ] **Step 2: Run to verify failure** — `venv/bin/pytest tests/test_detect_loop.py` → `ModuleNotFoundError: No module named 'agentgates.detect'`

- [ ] **Step 3: Implement** — `src/agentgates/detect/findings.py`:

```python
"""Typed findings emitted by detectors."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Severity = Literal["info", "warning", "error"]


class Finding(BaseModel):
    detector: str
    severity: Severity
    message: str
    step_index: int | None = None
    confidence: float = 1.0
```

`src/agentgates/detect/loop.py`:

```python
"""Deterministic loop/stall detector: repeated identical steps."""

from __future__ import annotations

import json
from collections import Counter

from agentgates.detect.findings import Finding
from agentgates.schema import AgentTrace, LLMCallStep, ToolCallStep


def _fingerprint(step) -> str:
    if isinstance(step, ToolCallStep):
        args = json.dumps(step.arguments, sort_keys=True, default=str)
        return f"tool:{step.tool_name}:{args}"
    if isinstance(step, LLMCallStep):
        return f"llm:{step.response}"
    return f"{step.type}:{step.step_id}"


class LoopDetector:
    name = "loop"

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold

    def detect(self, trace: AgentTrace) -> list[Finding]:
        findings: list[Finding] = []
        prints = [_fingerprint(s) for s in trace.steps]
        reported: set[str] = set()

        i = 0
        while i < len(prints):
            j = i
            while j < len(prints) and prints[j] == prints[i]:
                j += 1
            if j - i >= self.threshold:
                reported.add(prints[i])
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="error",
                        step_index=trace.steps[i].index,
                        message=(
                            f"identical step repeated {j - i}x consecutively:"
                            f" {prints[i][:80]}"
                        ),
                    )
                )
            i = j

        counts = Counter(p for p in prints if p.startswith("tool:"))
        for fp, n in counts.items():
            if n >= self.threshold and fp not in reported:
                first = next(
                    s.index for s, p in zip(trace.steps, prints) if p == fp
                )
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="warning",
                        step_index=first,
                        message=f"tool call repeated {n}x with identical arguments: {fp[:80]}",
                    )
                )
        return findings
```

`src/agentgates/detect/__init__.py`:

```python
"""Detect layer: analyzers that flag silent failures in an AgentTrace."""

from __future__ import annotations

from agentgates.detect.findings import Finding, Severity
from agentgates.detect.loop import LoopDetector
from agentgates.schema import AgentTrace

__all__ = ["Finding", "Severity", "LoopDetector", "default_detectors", "run_detectors"]


def default_detectors() -> list:
    return [LoopDetector()]


def run_detectors(trace: AgentTrace, detectors: list) -> list[Finding]:
    findings: list[Finding] = []
    for detector in detectors:
        findings.extend(detector.detect(trace))
    return sorted(
        findings,
        key=lambda f: f.step_index if f.step_index is not None else -1,
    )
```

- [ ] **Step 4: Run to verify pass** — `venv/bin/pytest tests/test_detect_loop.py -v` → 7 passed
- [ ] **Step 5: Commit** — `git add src/agentgates/detect/ tests/test_detect_loop.py && git commit -m "feat: Finding model and deterministic loop detector"` (+ trailer)

---

### Task 2: Tool-misuse detector

**Files:**
- Create: `src/agentgates/detect/tool_misuse.py`
- Modify: `src/agentgates/detect/__init__.py` (add to exports + `default_detectors`)
- Test: `tests/test_detect_tool_misuse.py`

**Interfaces:**
- Produces: `ToolMisuseDetector()` with `name = "tool_misuse"`. Warning per errored tool call; error when a call with identical (tool_name, arguments) errors again after already erroring (retry-without-change). `default_detectors()` returns `[LoopDetector(), ToolMisuseDetector()]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_detect_tool_misuse.py`:

```python
from agentgates.detect import default_detectors
from agentgates.detect.tool_misuse import ToolMisuseDetector
from agentgates.schema import AgentInfo, AgentTrace, TaskSpec, ToolCallStep


def make_trace(steps) -> AgentTrace:
    trace = AgentTrace(task=TaskSpec(description="t"), agent=AgentInfo(framework="test"))
    trace.steps = steps
    return trace


def tool(i, name="fetch", args=None, error=None, result="ok"):
    return ToolCallStep(
        index=i,
        tool_name=name,
        arguments=args or {},
        result=None if error else result,
        error=error,
    )


def test_clean_trace_no_findings():
    trace = make_trace([tool(0), tool(1, name="save")])
    assert ToolMisuseDetector().detect(trace) == []


def test_tool_error_is_warning():
    trace = make_trace([tool(0, error="timeout")])
    findings = ToolMisuseDetector().detect(trace)
    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].step_index == 0
    assert "timeout" in findings[0].message


def test_identical_retry_after_error_is_error():
    trace = make_trace([
        tool(0, args={"url": "x"}, error="500"),
        tool(1, args={"url": "x"}, error="500"),
    ])
    findings = ToolMisuseDetector().detect(trace)
    severities = [f.severity for f in findings]
    assert severities == ["warning", "error"]
    assert findings[1].step_index == 1
    assert "identical arguments" in findings[1].message


def test_retry_with_changed_args_is_only_warnings():
    trace = make_trace([
        tool(0, args={"url": "x"}, error="500"),
        tool(1, args={"url": "y"}, error="500"),
    ])
    findings = ToolMisuseDetector().detect(trace)
    assert [f.severity for f in findings] == ["warning", "warning"]


def test_successful_retry_not_flagged_as_misuse():
    trace = make_trace([
        tool(0, args={"url": "x"}, error="500"),
        tool(1, args={"url": "x"}),
    ])
    findings = ToolMisuseDetector().detect(trace)
    assert [f.severity for f in findings] == ["warning"]


def test_in_default_detectors():
    names = [d.name for d in default_detectors()]
    assert names == ["loop", "tool_misuse"]
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: No module named 'agentgates.detect.tool_misuse'`
- [ ] **Step 3: Implement** — `src/agentgates/detect/tool_misuse.py`:

```python
"""Deterministic tool-misuse detector: errors and blind retries."""

from __future__ import annotations

import json

from agentgates.detect.findings import Finding
from agentgates.schema import AgentTrace, ToolCallStep


class ToolMisuseDetector:
    name = "tool_misuse"

    def detect(self, trace: AgentTrace) -> list[Finding]:
        findings: list[Finding] = []
        prior_error_fps: set[str] = set()
        for step in trace.steps:
            if not isinstance(step, ToolCallStep):
                continue
            args = json.dumps(step.arguments, sort_keys=True, default=str)
            fp = f"{step.tool_name}:{args}"
            if step.error is None:
                continue
            if fp in prior_error_fps:
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="error",
                        step_index=step.index,
                        message=(
                            f"tool '{step.tool_name}' retried with identical"
                            f" arguments after an error and failed again"
                        ),
                    )
                )
            else:
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="warning",
                        step_index=step.index,
                        message=f"tool '{step.tool_name}' returned an error: {step.error}",
                    )
                )
                prior_error_fps.add(fp)
        return findings
```

In `src/agentgates/detect/__init__.py`: import `ToolMisuseDetector`, add to `__all__`, and change `default_detectors` to `return [LoopDetector(), ToolMisuseDetector()]`.

- [ ] **Step 4: Run to verify pass** — `venv/bin/pytest tests/test_detect_tool_misuse.py tests/test_detect_loop.py -v` → all pass
- [ ] **Step 5: Commit** — `feat: deterministic tool-misuse detector`

---

### Task 3: Judge interface (protocol, JSON extraction, cache, Claude judge)

**Files:**
- Create: `src/agentgates/detect/judge.py`
- Modify: `pyproject.toml` (add `[project.optional-dependencies] judge = ["anthropic>=0.40"]`)
- Test: `tests/test_detect_judge.py`

**Interfaces:**
- Produces (`agentgates.detect.judge`):
  - `Judge` — `typing.Protocol` with `complete(self, prompt: str) -> str`.
  - `extract_json(text: str) -> dict` — parses the first `{...}` JSON object in text; `{}` on failure.
  - `CachedJudge(inner: Judge, cache_path: Path | None = None)` — memoizes by sha256(prompt); persists to `cache_path` JSON when given.
  - `ClaudeJudge(model: str = "claude-opus-4-8")` — lazy `import anthropic` (ImportError message mentions `pip install "agentgates[judge]"`); `complete` calls `client.messages.create(model, max_tokens=1024, messages=[{"role": "user", "content": prompt}])` and joins `text`-type blocks.

- [ ] **Step 1: Write the failing tests** — `tests/test_detect_judge.py`:

```python
import json

from agentgates.detect.judge import CachedJudge, extract_json


class CountingJudge:
    def __init__(self, response='{"ok": true}'):
        self.calls = 0
        self.response = response

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self.response


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_surrounding_text():
    text = 'Here is my verdict:\n```json\n{"drifted": false, "x": [1, 2]}\n```\nDone.'
    assert extract_json(text) == {"drifted": False, "x": [1, 2]}


def test_extract_json_garbage_returns_empty():
    assert extract_json("no json here") == {}
    assert extract_json("{broken") == {}


def test_cached_judge_memoizes():
    inner = CountingJudge()
    judge = CachedJudge(inner)
    assert judge.complete("p") == '{"ok": true}'
    assert judge.complete("p") == '{"ok": true}'
    assert inner.calls == 1
    judge.complete("different")
    assert inner.calls == 2


def test_cached_judge_persists_to_disk(tmp_path):
    path = tmp_path / "cache.json"
    inner = CountingJudge()
    CachedJudge(inner, cache_path=path).complete("p")
    assert path.exists()
    assert len(json.loads(path.read_text())) == 1

    inner2 = CountingJudge(response="never used")
    judge2 = CachedJudge(inner2, cache_path=path)
    assert judge2.complete("p") == '{"ok": true}'
    assert inner2.calls == 0
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: No module named 'agentgates.detect.judge'`
- [ ] **Step 3: Implement** — `src/agentgates/detect/judge.py`:

```python
"""LLM-judge interface: model-agnostic, cached by prompt hash.

Deterministic checks run first everywhere in AgentGates; judges are only for
semantics (drift, groundedness). CachedJudge keys by prompt content hash so
CI reruns are cheap and deterministic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol


class Judge(Protocol):
    def complete(self, prompt: str) -> str: ...


def extract_json(text: str) -> dict:
    """Parse the first JSON object found in text; {} if none parses."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return {}


class CachedJudge:
    def __init__(self, inner: Judge, cache_path: Path | None = None) -> None:
        self._inner = inner
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, str] = {}
        if self._cache_path and self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))

    def complete(self, prompt: str) -> str:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._inner.complete(prompt)
            if self._cache_path:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._cache_path.write_text(
                    json.dumps(self._cache), encoding="utf-8"
                )
        return self._cache[key]


class ClaudeJudge:
    """Judge backed by the Claude API (requires the 'judge' extra)."""

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "ClaudeJudge requires the anthropic package:"
                ' pip install "agentgates[judge]"'
            ) from e
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
```

Add to `pyproject.toml` under `[project.optional-dependencies]`: `judge = ["anthropic>=0.40"]`.

- [ ] **Step 4: Run to verify pass** — 5 passed
- [ ] **Step 5: Commit** — `feat: model-agnostic Judge protocol with disk cache and Claude backend`

---

### Task 4: Goal-drift detector (judge-based)

**Files:**
- Create: `src/agentgates/detect/goal_drift.py`
- Create: `src/agentgates/detect/_summary.py` (shared step summarizer)
- Test: `tests/test_detect_goal_drift.py`

**Interfaces:**
- Produces:
  - `agentgates.detect._summary.summarize_step(step) -> str` — `[i] llm: <response[:200]>` / `[i] tool <name>({args}) -> <error or result, [:200]>`.
  - `GoalDriftDetector(judge: Judge)` with `name = "goal_drift"`. One judge call per trace; prompt includes task description, constraints, numbered steps; expects JSON `{"drifted": bool, "step_index": int|null, "dropped_constraints": [...], "reasoning": str}`. Emits one error finding (confidence 0.8) when drifted, else [].

- [ ] **Step 1: Write the failing tests** — `tests/test_detect_goal_drift.py`:

```python
from agentgates.detect.goal_drift import GoalDriftDetector
from agentgates.schema import AgentInfo, AgentTrace, LLMCallStep, TaskSpec


class FakeJudge:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def make_trace() -> AgentTrace:
    trace = AgentTrace(
        task=TaskSpec(description="book flight", constraints=["under $500"]),
        agent=AgentInfo(framework="test"),
    )
    trace.steps.append(LLMCallStep(index=0, model="m", prompt="", response="booking $900 flight"))
    return trace


def test_no_drift_no_findings():
    judge = FakeJudge('{"drifted": false, "step_index": null, "dropped_constraints": [], "reasoning": "on task"}')
    assert GoalDriftDetector(judge).detect(make_trace()) == []


def test_drift_emits_error_with_step():
    judge = FakeJudge('{"drifted": true, "step_index": 0, "dropped_constraints": ["under $500"], "reasoning": "budget ignored"}')
    findings = GoalDriftDetector(judge).detect(make_trace())
    assert len(findings) == 1
    f = findings[0]
    assert f.detector == "goal_drift"
    assert f.severity == "error"
    assert f.step_index == 0
    assert "under $500" in f.message
    assert f.confidence == 0.8


def test_prompt_contains_task_and_steps():
    judge = FakeJudge('{"drifted": false}')
    GoalDriftDetector(judge).detect(make_trace())
    prompt = judge.prompts[0]
    assert "book flight" in prompt
    assert "under $500" in prompt
    assert "booking $900 flight" in prompt


def test_unparseable_judge_output_means_no_findings():
    judge = FakeJudge("I cannot answer")
    assert GoalDriftDetector(judge).detect(make_trace()) == []


def test_empty_trace_skips_judge():
    trace = AgentTrace(task=TaskSpec(description="t"), agent=AgentInfo(framework="test"))
    judge = FakeJudge('{"drifted": true}')
    assert GoalDriftDetector(judge).detect(trace) == []
    assert judge.prompts == []
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError
- [ ] **Step 3: Implement** — `src/agentgates/detect/_summary.py`:

```python
"""Shared plain-text step summaries for judge prompts."""

from __future__ import annotations

from agentgates.schema import LLMCallStep, ToolCallStep


def summarize_step(step) -> str:
    if isinstance(step, LLMCallStep):
        return f"[{step.index}] llm: {step.response[:200]}"
    if isinstance(step, ToolCallStep):
        outcome = f"ERROR {step.error}" if step.error else str(step.result)[:200]
        return f"[{step.index}] tool {step.tool_name}({step.arguments}) -> {outcome}"
    return f"[{step.index}] {step.type}"
```

`src/agentgates/detect/goal_drift.py`:

```python
"""Judge-based goal-drift detector: dropped constraints, diverging objective."""

from __future__ import annotations

from agentgates.detect._summary import summarize_step
from agentgates.detect.findings import Finding
from agentgates.detect.judge import Judge, extract_json
from agentgates.schema import AgentTrace

_PROMPT = """You are auditing an AI agent's execution trace for goal drift.

Original task: {task}
Constraints: {constraints}

Agent steps:
{steps}

Did the agent's working objective diverge from the original task, or did it \
drop/violate any constraint? Respond with ONLY a JSON object:
{{"drifted": true/false, "step_index": <int index of the first drifting step, or null>, \
"dropped_constraints": ["..."], "reasoning": "one sentence"}}"""


class GoalDriftDetector:
    name = "goal_drift"

    def __init__(self, judge: Judge) -> None:
        self._judge = judge

    def detect(self, trace: AgentTrace) -> list[Finding]:
        if not trace.steps:
            return []
        prompt = _PROMPT.format(
            task=trace.task.description,
            constraints=", ".join(trace.task.constraints) or "(none)",
            steps="\n".join(summarize_step(s) for s in trace.steps),
        )
        data = extract_json(self._judge.complete(prompt))
        if not data.get("drifted"):
            return []
        dropped = ", ".join(data.get("dropped_constraints") or [])
        detail = dropped or data.get("reasoning", "objective diverged from task")
        step_index = data.get("step_index")
        return [
            Finding(
                detector=self.name,
                severity="error",
                step_index=step_index if isinstance(step_index, int) else None,
                message=f"goal drift: {detail}",
                confidence=0.8,
            )
        ]
```

- [ ] **Step 4: Run to verify pass** — 5 passed
- [ ] **Step 5: Commit** — `feat: judge-based goal-drift detector`

---

### Task 5: Ungrounded-assumption detector (judge-based)

**Files:**
- Create: `src/agentgates/detect/ungrounded.py`
- Modify: `src/agentgates/detect/__init__.py` (export judge detectors + `judge_detectors(judge)` helper)
- Test: `tests/test_detect_ungrounded.py`

**Interfaces:**
- Produces:
  - `UngroundedAssumptionDetector(judge)` with `name = "ungrounded_assumption"`. Walks steps in order accumulating evidence (task description + each tool result/error); for each `LLMCallStep` makes one judge call with evidence-so-far + the message; expects JSON `{"ungrounded_claims": ["..."], "reasoning": str}`; one error finding per claim (confidence 0.8) at that step.
  - `agentgates.detect.judge_detectors(judge) -> list` returning `[GoalDriftDetector(judge), UngroundedAssumptionDetector(judge)]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_detect_ungrounded.py`:

```python
from agentgates.detect import judge_detectors
from agentgates.detect.ungrounded import UngroundedAssumptionDetector
from agentgates.schema import AgentInfo, AgentTrace, LLMCallStep, TaskSpec, ToolCallStep


class FakeJudge:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def make_trace() -> AgentTrace:
    trace = AgentTrace(task=TaskSpec(description="find flights"), agent=AgentInfo(framework="test"))
    trace.steps.append(ToolCallStep(index=0, tool_name="search", arguments={}, result={"flights": ["UA 420"]}))
    trace.steps.append(LLMCallStep(index=1, model="m", prompt="", response="UA 420 has free wifi"))
    return trace


def test_grounded_no_findings():
    judge = FakeJudge(['{"ungrounded_claims": [], "reasoning": "all grounded"}'])
    assert UngroundedAssumptionDetector(judge).detect(make_trace()) == []


def test_ungrounded_claim_flagged_at_step():
    judge = FakeJudge(['{"ungrounded_claims": ["UA 420 has free wifi"], "reasoning": "wifi never mentioned"}'])
    findings = UngroundedAssumptionDetector(judge).detect(make_trace())
    assert len(findings) == 1
    f = findings[0]
    assert f.detector == "ungrounded_assumption"
    assert f.severity == "error"
    assert f.step_index == 1
    assert "free wifi" in f.message


def test_one_judge_call_per_llm_step_with_prior_evidence():
    trace = make_trace()
    trace.steps.append(LLMCallStep(index=2, model="m", prompt="", response="booking it"))
    judge = FakeJudge(['{"ungrounded_claims": []}', '{"ungrounded_claims": []}'])
    UngroundedAssumptionDetector(judge).detect(trace)
    assert len(judge.prompts) == 2
    assert "UA 420" in judge.prompts[0]      # tool evidence present
    assert "find flights" in judge.prompts[0]  # task present
    assert "free wifi" in judge.prompts[1]   # prior llm message becomes evidence


def test_tool_only_trace_makes_no_judge_calls():
    trace = AgentTrace(task=TaskSpec(description="t"), agent=AgentInfo(framework="test"))
    trace.steps.append(ToolCallStep(index=0, tool_name="x", arguments={}, result="ok"))
    judge = FakeJudge([])
    assert UngroundedAssumptionDetector(judge).detect(trace) == []


def test_judge_detectors_factory():
    judge = FakeJudge([])
    names = [d.name for d in judge_detectors(judge)]
    assert names == ["goal_drift", "ungrounded_assumption"]
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError
- [ ] **Step 3: Implement** — `src/agentgates/detect/ungrounded.py`:

```python
"""Judge-based silent-hallucination detector: claims with no supporting evidence."""

from __future__ import annotations

from agentgates.detect._summary import summarize_step
from agentgates.detect.findings import Finding
from agentgates.detect.judge import Judge, extract_json
from agentgates.schema import AgentTrace, LLMCallStep

_PROMPT = """You are auditing one message from an AI agent for ungrounded assumptions \
(silent hallucinations): factual claims that appear in none of the evidence available \
to the agent.

Evidence available to the agent so far:
{evidence}

Agent message to audit:
{message}

List factual claims in the message that are NOT supported by the evidence. Ignore \
plans, intentions, and hedged statements. Respond with ONLY a JSON object:
{{"ungrounded_claims": ["exact claim", ...], "reasoning": "one sentence"}}"""


class UngroundedAssumptionDetector:
    name = "ungrounded_assumption"

    def __init__(self, judge: Judge) -> None:
        self._judge = judge

    def detect(self, trace: AgentTrace) -> list[Finding]:
        findings: list[Finding] = []
        evidence: list[str] = [f"task: {trace.task.description}"]
        for step in trace.steps:
            if isinstance(step, LLMCallStep):
                prompt = _PROMPT.format(
                    evidence="\n".join(evidence), message=step.response
                )
                data = extract_json(self._judge.complete(prompt))
                for claim in data.get("ungrounded_claims") or []:
                    findings.append(
                        Finding(
                            detector=self.name,
                            severity="error",
                            step_index=step.index,
                            message=f"ungrounded assumption: {claim}",
                            confidence=0.8,
                        )
                    )
            evidence.append(summarize_step(step))
        return findings
```

In `src/agentgates/detect/__init__.py`: import `GoalDriftDetector`, `UngroundedAssumptionDetector`, `Judge`; add:

```python
def judge_detectors(judge) -> list:
    return [GoalDriftDetector(judge), UngroundedAssumptionDetector(judge)]
```

and extend `__all__` accordingly.

- [ ] **Step 4: Run to verify pass** — 5 passed; full suite green
- [ ] **Step 5: Commit** — `feat: judge-based ungrounded-assumption detector`

---

### Task 6: Failure-injection eval corpus + metrics

**Files:**
- Create: `src/agentgates/evals/__init__.py`
- Create: `src/agentgates/evals/corpus.py`
- Create: `src/agentgates/evals/metrics.py`
- Test: `tests/test_evals.py`

**Interfaces:**
- Produces:
  - `LabeledTrace` (dataclass): `trace: AgentTrace`, `expected_detectors: set[str]`.
  - `build_corpus() -> list[LabeledTrace]` — 12 deterministic traces: 4 clean, 3 loop-injected, 3 tool-misuse-injected, 2 both.
  - `DetectorScore` (Pydantic): `detector, tp, fp, fn` + properties `precision`, `recall` (1.0 when denominator 0).
  - `evaluate_detector(detector, corpus) -> DetectorScore` — trace-level: detector "fires" if it emits ≥1 finding.

- [ ] **Step 1: Write the failing tests** — `tests/test_evals.py`:

```python
from agentgates.detect import LoopDetector, ToolMisuseDetector
from agentgates.evals.corpus import build_corpus
from agentgates.evals.metrics import DetectorScore, evaluate_detector


def test_corpus_composition():
    corpus = build_corpus()
    assert len(corpus) == 12
    assert sum(1 for c in corpus if not c.expected_detectors) == 4
    assert sum(1 for c in corpus if "loop" in c.expected_detectors) == 5
    assert sum(1 for c in corpus if "tool_misuse" in c.expected_detectors) == 5


def test_deterministic_detectors_perfect_on_corpus():
    corpus = build_corpus()
    for detector in (LoopDetector(), ToolMisuseDetector()):
        score = evaluate_detector(detector, corpus)
        assert score.precision == 1.0, f"{detector.name} fp={score.fp}"
        assert score.recall == 1.0, f"{detector.name} fn={score.fn}"


def test_score_math():
    score = DetectorScore(detector="d", tp=3, fp=1, fn=2)
    assert score.precision == 0.75
    assert score.recall == 0.6


def test_score_zero_denominators():
    score = DetectorScore(detector="d", tp=0, fp=0, fn=0)
    assert score.precision == 1.0
    assert score.recall == 1.0
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError
- [ ] **Step 3: Implement** — `src/agentgates/evals/__init__.py`:

```python
"""Eval the evaluator: labeled failure-injection corpus + detector metrics."""

from agentgates.evals.corpus import LabeledTrace, build_corpus
from agentgates.evals.metrics import DetectorScore, evaluate_detector

__all__ = ["LabeledTrace", "build_corpus", "DetectorScore", "evaluate_detector"]
```

`src/agentgates/evals/corpus.py`:

```python
"""Programmatically generated labeled corpus with injected failures."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentgates.schema import AgentInfo, AgentTrace, LLMCallStep, TaskSpec, ToolCallStep


@dataclass
class LabeledTrace:
    trace: AgentTrace
    expected_detectors: set[str] = field(default_factory=set)


def _trace(desc: str, steps: list) -> AgentTrace:
    trace = AgentTrace(task=TaskSpec(description=desc), agent=AgentInfo(framework="corpus"))
    trace.steps = steps
    return trace


def _llm(i: int, text: str) -> LLMCallStep:
    return LLMCallStep(index=i, model="corpus", prompt="", response=text)


def _tool(i: int, name: str, args: dict, result=None, error=None) -> ToolCallStep:
    return ToolCallStep(index=i, tool_name=name, arguments=args, result=result, error=error)


def _clean(n: int) -> AgentTrace:
    return _trace(
        f"clean task {n}",
        [
            _llm(0, f"planning approach {n}"),
            _tool(1, "search", {"q": f"query {n}"}, result=f"result {n}"),
            _llm(2, f"found answer {n}"),
            _tool(3, "save", {"value": n}, result="saved"),
            _llm(4, f"done with {n}"),
        ],
    )


def _loop_consecutive() -> AgentTrace:
    return _trace(
        "loop: same call repeated",
        [_tool(i, "fetch", {"url": "same"}, result="pending") for i in range(4)],
    )


def _loop_llm_repeat() -> AgentTrace:
    return _trace(
        "loop: agent restates itself",
        [_llm(i, "I will try again") for i in range(3)],
    )


def _loop_scattered() -> AgentTrace:
    return _trace(
        "loop: same call scattered",
        [
            _tool(0, "check", {"id": 7}, result="no"),
            _llm(1, "checking once more"),
            _tool(2, "check", {"id": 7}, result="no"),
            _llm(3, "one more look"),
            _tool(4, "check", {"id": 7}, result="no"),
        ],
    )


def _misuse_error_ignored() -> AgentTrace:
    return _trace(
        "misuse: error then unrelated continue",
        [
            _tool(0, "fetch", {"url": "a"}, error="404 not found"),
            _llm(1, "continuing anyway"),
        ],
    )


def _misuse_blind_retry() -> AgentTrace:
    return _trace(
        "misuse: identical failing retry",
        [
            _tool(0, "post", {"body": "x"}, error="500"),
            _tool(1, "post", {"body": "x"}, error="500"),
            _llm(2, "giving up"),
        ],
    )


def _misuse_two_errors() -> AgentTrace:
    return _trace(
        "misuse: two different tools error",
        [
            _tool(0, "read", {"path": "a"}, error="permission denied"),
            _tool(1, "write", {"path": "b"}, error="disk full"),
        ],
    )


def _combined(n: int) -> AgentTrace:
    return _trace(
        f"combined failure {n}",
        [
            _tool(0, "poll", {"job": n}, error="timeout"),
            _tool(1, "poll", {"job": n}, error="timeout"),
            _tool(2, "poll", {"job": n}, error="timeout"),
            _llm(3, "still waiting"),
        ],
    )


def build_corpus() -> list[LabeledTrace]:
    corpus = [LabeledTrace(_clean(n)) for n in range(1, 5)]
    corpus += [
        LabeledTrace(_loop_consecutive(), {"loop"}),
        LabeledTrace(_loop_llm_repeat(), {"loop"}),
        LabeledTrace(_loop_scattered(), {"loop"}),
        LabeledTrace(_misuse_error_ignored(), {"tool_misuse"}),
        LabeledTrace(_misuse_blind_retry(), {"tool_misuse"}),
        LabeledTrace(_misuse_two_errors(), {"tool_misuse"}),
        LabeledTrace(_combined(1), {"loop", "tool_misuse"}),
        LabeledTrace(_combined(2), {"loop", "tool_misuse"}),
    ]
    return corpus
```

`src/agentgates/evals/metrics.py`:

```python
"""Trace-level precision/recall for a detector over a labeled corpus."""

from __future__ import annotations

from pydantic import BaseModel

from agentgates.evals.corpus import LabeledTrace


class DetectorScore(BaseModel):
    detector: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0


def evaluate_detector(detector, corpus: list[LabeledTrace]) -> DetectorScore:
    tp = fp = fn = 0
    for labeled in corpus:
        fired = bool(detector.detect(labeled.trace))
        expected = detector.name in labeled.expected_detectors
        if fired and expected:
            tp += 1
        elif fired and not expected:
            fp += 1
        elif not fired and expected:
            fn += 1
    return DetectorScore(detector=detector.name, tp=tp, fp=fp, fn=fn)
```

- [ ] **Step 4: Run to verify pass** — 4 passed (adjust corpus if precision/recall != 1.0 — the corpus and detectors must agree; fix the corpus construction, not the assertion)
- [ ] **Step 5: Commit** — `feat: labeled failure-injection corpus and detector eval metrics`

---

### Task 7: CLI `detect` and `eval` commands

**Files:**
- Modify: `src/agentgates/cli.py`
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Produces:
  - `agentgates detect TRACE_ID [--store-dir PATH] [--judge]` — loads trace, runs `default_detectors()` (plus `judge_detectors(CachedJudge(ClaudeJudge(), <store-dir>/judge_cache.json))` when `--judge`), prints `[severity] step N detector: message` lines or "no findings"; exit 1 when trace missing; exit 0 otherwise (gating is Phase 3's `ci`).
  - `agentgates eval` — prints per-detector precision/recall table for deterministic detectors over the built-in corpus.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_cli.py`:

```python
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
    assert "1.00" in result.output
```

- [ ] **Step 2: Run to verify failure** — the four new tests fail (exit code 2, no such command)
- [ ] **Step 3: Implement** — add to `src/agentgates/cli.py`:

```python
from agentgates.detect import default_detectors, judge_detectors, run_detectors
from agentgates.detect.judge import CachedJudge, ClaudeJudge
from agentgates.evals import build_corpus, evaluate_detector


@app.command()
def detect(
    trace_id: str,
    store_dir: Path = StoreDirOption,
    judge: bool = typer.Option(False, "--judge", help="Also run LLM-judge detectors (needs ANTHROPIC_API_KEY and the 'judge' extra)."),
) -> None:
    """Run failure detectors over one recorded trace."""
    try:
        trace = _store(store_dir).load(trace_id)
    except KeyError:
        typer.echo(f"trace not found: {trace_id}")
        raise typer.Exit(code=1)
    detectors = default_detectors()
    if judge:
        cached = CachedJudge(ClaudeJudge(), cache_path=store_dir / "judge_cache.json")
        detectors += judge_detectors(cached)
    findings = run_detectors(trace, detectors)
    if not findings:
        typer.echo("no findings")
        return
    for f in findings:
        step = f"step {f.step_index}" if f.step_index is not None else "trace"
        typer.echo(f"[{f.severity}] {step} {f.detector}: {f.message}")


@app.command("eval")
def eval_cmd() -> None:
    """Score the deterministic detectors against the built-in labeled corpus."""
    corpus = build_corpus()
    typer.echo(f"corpus: {len(corpus)} labeled traces")
    typer.echo(f"{'detector':<16} {'precision':>9} {'recall':>7}")
    for detector in default_detectors():
        score = evaluate_detector(detector, corpus)
        typer.echo(f"{score.detector:<16} {score.precision:>9.2f} {score.recall:>7.2f}")
```

- [ ] **Step 4: Run to verify pass** — `venv/bin/pytest -q` all green
- [ ] **Step 5: Commit** — `feat: CLI detect and eval commands`
