# TraceGate Phase 1 (Core Spine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build TraceGate's Phase 1 core spine: the AgentTrace schema, a trace recorder with a Claude Agent SDK adapter, JSONL + SQLite local stores, and a minimal Typer CLI (`record`, `show`, `list`).

**Architecture:** A framework-agnostic Pydantic `AgentTrace` schema is the core product; a `TraceRecorder` builds traces in memory and persists them via a `TraceStore` protocol (JSONL default, SQLite optional). The Claude Agent SDK adapter translates the SDK's message stream into recorder calls without importing the SDK (duck-typed, so tests need no API key or network). The CLI reads/writes the local store only.

**Tech Stack:** Python 3.11+, Pydantic v2, Typer, SQLite (stdlib `sqlite3`), pytest. src layout, `pip install -e ".[dev]"` into `venv/`.

## Global Constraints

- Python `>=3.11`; dependencies limited to `pydantic>=2.7` and `typer>=0.12` (dev: `pytest>=8`). No other runtime deps in Phase 1.
- Local-first: no network calls anywhere in Phase 1 code or tests; the Claude adapter must not import `claude_agent_sdk` (duck-typed message handling only).
- Package name `tracegate`, console script `tracegate`, src layout (`src/tracegate/`).
- Default trace location: directory from env var `TRACEGATE_DIR`, falling back to `.tracegate/`; JSONL file name `traces.jsonl`.
- Never commit `CLAUDE.md`, `venv/`, `*.sqlite`, `.tracegate/` (gitignored).
- All work happens inside `/Users/iymanahmed/Documents/New project/tracegate` which gets its own git repo (Task 1). Run tests with `venv/bin/pytest` from the project root.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/tracegate/__init__.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_package.py`
- Modify: `.gitignore` (add `.tracegate/`, `*.egg-info/`, `.pytest_cache/`)

**Interfaces:**
- Consumes: nothing (first task)
- Produces: installable package `tracegate` with `tracegate.__version__: str = "0.1.0"`; a working `venv/bin/pytest`; a git repo with an initial commit.

- [ ] **Step 1: Init git repo and write packaging files**

`git init` in the project root (this makes `tracegate/` its own repo nested inside the parent folder — intended, it's a standalone portfolio project).

`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tracegate"
version = "0.1.0"
description = "The black box flight recorder for AI agents - record every step, detect silent failures, gate your deploys."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
tracegate = "tracegate.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/tracegate"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`src/tracegate/__init__.py`:

```python
"""TraceGate — the black box flight recorder for AI agents."""

__version__ = "0.1.0"
```

Append to `.gitignore`:

```
.tracegate/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 2: Write the failing test**

`tests/test_package.py`:

```python
import tracegate


def test_version():
    assert tracegate.__version__ == "0.1.0"
```

- [ ] **Step 3: Create venv, install, run test**

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev]"
venv/bin/pytest tests/test_package.py -v
```

Expected: 1 passed. (The `tracegate.cli:app` script target doesn't exist yet — that's fine; the entry point is only resolved when the `tracegate` command is run, which happens in Task 7.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/ tests/ .gitignore README.md PROPOSAL.md docs/
git commit -m "chore: scaffold tracegate package (src layout, hatchling, pytest)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: AgentTrace schema

**Files:**
- Create: `src/tracegate/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing
- Produces (all Pydantic v2 models in `tracegate.schema`):
  - `TokenUsage(input_tokens: int = 0, output_tokens: int = 0)`
  - `LLMCallStep(type="llm_call", step_id: str, index: int, timestamp: datetime, model: str, prompt: str, response: str, usage: TokenUsage)`
  - `ToolCallStep(type="tool_call", step_id: str, index: int, timestamp: datetime, tool_name: str, arguments: dict, result: Any = None, error: str | None = None, latency_ms: float | None = None)`
  - `Step` — discriminated union of the two, discriminator `type`
  - `TaskSpec(description: str, constraints: list[str] = [])`
  - `AgentInfo(framework: str, model: str | None = None)`
  - `AgentTrace(trace_id: str, task: TaskSpec, agent: AgentInfo, started_at: datetime, ended_at: datetime | None, steps: list[Step], metadata: dict)`
  - `step_id`/`trace_id` default to `uuid4().hex`; timestamps default to `datetime.now(timezone.utc)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_schema.py`:

```python
from tracegate.schema import (
    AgentInfo,
    AgentTrace,
    LLMCallStep,
    TaskSpec,
    ToolCallStep,
)


def make_trace() -> AgentTrace:
    trace = AgentTrace(
        task=TaskSpec(description="book a flight", constraints=["under $500"]),
        agent=AgentInfo(framework="claude-agent-sdk", model="claude-sonnet-5"),
    )
    trace.steps.append(
        LLMCallStep(index=0, model="claude-sonnet-5", prompt="hi", response="hello")
    )
    trace.steps.append(
        ToolCallStep(
            index=1,
            tool_name="search_flights",
            arguments={"to": "NYC"},
            result={"flights": 3},
            latency_ms=41.5,
        )
    )
    return trace


def test_trace_defaults():
    trace = make_trace()
    assert len(trace.trace_id) == 32  # uuid4 hex
    assert trace.started_at is not None
    assert trace.ended_at is None
    assert trace.metadata == {}


def test_step_discriminated_union_roundtrip():
    trace = make_trace()
    restored = AgentTrace.model_validate_json(trace.model_dump_json())
    assert isinstance(restored.steps[0], LLMCallStep)
    assert isinstance(restored.steps[1], ToolCallStep)
    assert restored.steps[1].tool_name == "search_flights"
    assert restored.steps[1].result == {"flights": 3}
    assert restored.trace_id == trace.trace_id


def test_tool_step_error_fields():
    step = ToolCallStep(index=0, tool_name="x", arguments={}, error="boom")
    assert step.result is None
    assert step.error == "boom"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracegate.schema'`

- [ ] **Step 3: Write the implementation**

`src/tracegate/schema.py`:

```python
"""The AgentTrace schema — TraceGate's core data model.

The trace schema is the real product: framework adapters normalize into
this shape, and every detector/report in later phases reads from it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class _BaseStep(BaseModel):
    step_id: str = Field(default_factory=_new_id)
    index: int
    timestamp: datetime = Field(default_factory=_now)


class LLMCallStep(_BaseStep):
    type: Literal["llm_call"] = "llm_call"
    model: str
    prompt: str
    response: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class ToolCallStep(_BaseStep):
    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    latency_ms: float | None = None


Step = Annotated[Union[LLMCallStep, ToolCallStep], Field(discriminator="type")]


class TaskSpec(BaseModel):
    description: str
    constraints: list[str] = Field(default_factory=list)


class AgentInfo(BaseModel):
    framework: str
    model: str | None = None


class AgentTrace(BaseModel):
    trace_id: str = Field(default_factory=_new_id)
    task: TaskSpec
    agent: AgentInfo
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    steps: list[Step] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_schema.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/tracegate/schema.py tests/test_schema.py
git commit -m "feat: AgentTrace schema with discriminated step union

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: JSONL trace store

**Files:**
- Create: `src/tracegate/store/__init__.py`
- Create: `src/tracegate/store/jsonl.py`
- Test: `tests/test_store_jsonl.py`

**Interfaces:**
- Consumes: `AgentTrace` from Task 2.
- Produces:
  - `tracegate.store.jsonl.JSONLTraceStore(path: Path)` with methods `save(trace: AgentTrace) -> None`, `load(trace_id: str) -> AgentTrace` (raises `KeyError` if missing), `list_traces() -> list[AgentTrace]` (oldest first).
  - `tracegate.store.default_store() -> JSONLTraceStore` — uses `$TRACEGATE_DIR` or `.tracegate/`, file `traces.jsonl`.

- [ ] **Step 1: Write the failing tests**

`tests/test_store_jsonl.py`:

```python
import pytest

from tracegate.schema import AgentInfo, AgentTrace, TaskSpec
from tracegate.store import default_store
from tracegate.store.jsonl import JSONLTraceStore


def make_trace(desc: str = "task") -> AgentTrace:
    return AgentTrace(
        task=TaskSpec(description=desc),
        agent=AgentInfo(framework="test"),
    )


def test_save_and_load_roundtrip(tmp_path):
    store = JSONLTraceStore(tmp_path / "traces.jsonl")
    trace = make_trace()
    store.save(trace)
    loaded = store.load(trace.trace_id)
    assert loaded.trace_id == trace.trace_id
    assert loaded.task.description == "task"


def test_load_missing_raises(tmp_path):
    store = JSONLTraceStore(tmp_path / "traces.jsonl")
    with pytest.raises(KeyError):
        store.load("nope")


def test_list_traces_ordered(tmp_path):
    store = JSONLTraceStore(tmp_path / "traces.jsonl")
    a, b = make_trace("a"), make_trace("b")
    store.save(a)
    store.save(b)
    listed = store.list_traces()
    assert [t.task.description for t in listed] == ["a", "b"]


def test_save_creates_parent_dir(tmp_path):
    store = JSONLTraceStore(tmp_path / "deep" / "traces.jsonl")
    store.save(make_trace())
    assert (tmp_path / "deep" / "traces.jsonl").exists()


def test_default_store_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEGATE_DIR", str(tmp_path / "custom"))
    store = default_store()
    assert store.path == tmp_path / "custom" / "traces.jsonl"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_store_jsonl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracegate.store'`

- [ ] **Step 3: Write the implementation**

`src/tracegate/store/jsonl.py`:

```python
"""Append-only JSONL trace store — the zero-config default."""

from __future__ import annotations

from pathlib import Path

from tracegate.schema import AgentTrace


class JSONLTraceStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def save(self, trace: AgentTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")

    def load(self, trace_id: str) -> AgentTrace:
        for trace in self.list_traces():
            if trace.trace_id == trace_id:
                return trace
        raise KeyError(f"trace not found: {trace_id}")

    def list_traces(self) -> list[AgentTrace]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [
                AgentTrace.model_validate_json(line)
                for line in f
                if line.strip()
            ]
```

`src/tracegate/store/__init__.py`:

```python
"""Local-first trace stores."""

from __future__ import annotations

import os
from pathlib import Path

from tracegate.store.jsonl import JSONLTraceStore

TRACES_FILENAME = "traces.jsonl"


def default_store() -> JSONLTraceStore:
    root = Path(os.environ.get("TRACEGATE_DIR", ".tracegate"))
    return JSONLTraceStore(root / TRACES_FILENAME)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_store_jsonl.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/tracegate/store/ tests/test_store_jsonl.py
git commit -m "feat: JSONL trace store with env-configurable default location

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: SQLite trace store

**Files:**
- Create: `src/tracegate/store/sqlite.py`
- Test: `tests/test_store_sqlite.py`

**Interfaces:**
- Consumes: `AgentTrace` from Task 2.
- Produces: `tracegate.store.sqlite.SQLiteTraceStore(path: Path)` with the same three methods as `JSONLTraceStore`: `save(trace) -> None` (upsert by `trace_id`), `load(trace_id) -> AgentTrace` (raises `KeyError`), `list_traces() -> list[AgentTrace]` (ordered by `started_at`).

- [ ] **Step 1: Write the failing tests**

`tests/test_store_sqlite.py`:

```python
import pytest

from tracegate.schema import AgentInfo, AgentTrace, TaskSpec
from tracegate.store.sqlite import SQLiteTraceStore


def make_trace(desc: str = "task") -> AgentTrace:
    return AgentTrace(
        task=TaskSpec(description=desc),
        agent=AgentInfo(framework="test"),
    )


def test_save_and_load_roundtrip(tmp_path):
    store = SQLiteTraceStore(tmp_path / "traces.sqlite")
    trace = make_trace()
    store.save(trace)
    loaded = store.load(trace.trace_id)
    assert loaded.trace_id == trace.trace_id
    assert loaded.task.description == "task"


def test_load_missing_raises(tmp_path):
    store = SQLiteTraceStore(tmp_path / "traces.sqlite")
    with pytest.raises(KeyError):
        store.load("nope")


def test_save_is_upsert(tmp_path):
    store = SQLiteTraceStore(tmp_path / "traces.sqlite")
    trace = make_trace("v1")
    store.save(trace)
    trace.task.description = "v2"
    store.save(trace)
    assert store.load(trace.trace_id).task.description == "v2"
    assert len(store.list_traces()) == 1


def test_list_traces_ordered(tmp_path):
    store = SQLiteTraceStore(tmp_path / "traces.sqlite")
    a, b = make_trace("a"), make_trace("b")
    b.started_at = b.started_at.replace(year=b.started_at.year + 1)
    store.save(b)
    store.save(a)
    listed = store.list_traces()
    assert [t.task.description for t in listed] == ["a", "b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_store_sqlite.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracegate.store.sqlite'`

- [ ] **Step 3: Write the implementation**

`src/tracegate/store/sqlite.py`:

```python
"""SQLite trace store — same interface as JSONL, queryable at scale."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tracegate.schema import AgentTrace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id   TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    payload    TEXT NOT NULL
)
"""


class SQLiteTraceStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def save(self, trace: AgentTrace) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces (trace_id, started_at, payload)"
                " VALUES (?, ?, ?)",
                (
                    trace.trace_id,
                    trace.started_at.isoformat(),
                    trace.model_dump_json(),
                ),
            )

    def load(self, trace_id: str) -> AgentTrace:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"trace not found: {trace_id}")
        return AgentTrace.model_validate_json(row[0])

    def list_traces(self) -> list[AgentTrace]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM traces ORDER BY started_at"
            ).fetchall()
        return [AgentTrace.model_validate_json(r[0]) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_store_sqlite.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/tracegate/store/sqlite.py tests/test_store_sqlite.py
git commit -m "feat: SQLite trace store

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: TraceRecorder

**Files:**
- Create: `src/tracegate/recorder.py`
- Modify: `src/tracegate/__init__.py` (re-export public API)
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: `AgentTrace`, `TaskSpec`, `AgentInfo`, `LLMCallStep`, `ToolCallStep`, `TokenUsage` (Task 2); `default_store()` (Task 3).
- Produces: `tracegate.recorder.TraceRecorder`:
  - `TraceRecorder(task: str | TaskSpec, framework: str, model: str | None = None, store: object | None = None)` — `store` is any object with `save(trace)`; `None` means `default_store()`.
  - attribute `trace: AgentTrace`
  - `record_llm_call(*, model: str, prompt: str, response: str, input_tokens: int = 0, output_tokens: int = 0) -> LLMCallStep`
  - `record_tool_call(*, tool_name: str, arguments: dict, result=None, error: str | None = None, latency_ms: float | None = None) -> ToolCallStep`
  - `finish() -> AgentTrace` — sets `ended_at`, saves to store, idempotent (second call does not re-save)
  - context manager: `__enter__` returns self, `__exit__` calls `finish()`
  - Also re-exported as `tracegate.TraceRecorder`.

- [ ] **Step 1: Write the failing tests**

`tests/test_recorder.py`:

```python
from tracegate.recorder import TraceRecorder
from tracegate.schema import TaskSpec


class FakeStore:
    def __init__(self):
        self.saved = []

    def save(self, trace):
        self.saved.append(trace)


def test_records_steps_with_increasing_index():
    rec = TraceRecorder(task="do things", framework="test", store=FakeStore())
    rec.record_llm_call(model="m", prompt="p", response="r")
    rec.record_tool_call(tool_name="t", arguments={"a": 1}, result="ok")
    assert [s.index for s in rec.trace.steps] == [0, 1]
    assert rec.trace.steps[1].tool_name == "t"


def test_task_string_becomes_taskspec():
    rec = TraceRecorder(task="hello", framework="test", store=FakeStore())
    assert rec.trace.task == TaskSpec(description="hello")


def test_finish_sets_ended_at_and_saves_once():
    store = FakeStore()
    rec = TraceRecorder(task="x", framework="test", store=store)
    trace = rec.finish()
    rec.finish()
    assert trace.ended_at is not None
    assert len(store.saved) == 1


def test_context_manager_finishes():
    store = FakeStore()
    with TraceRecorder(task="x", framework="test", store=store) as rec:
        rec.record_llm_call(model="m", prompt="p", response="r")
    assert len(store.saved) == 1
    assert store.saved[0].ended_at is not None


def test_default_store_used_when_none(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEGATE_DIR", str(tmp_path))
    with TraceRecorder(task="x", framework="test"):
        pass
    assert (tmp_path / "traces.jsonl").exists()


def test_public_api_reexport():
    import tracegate

    assert tracegate.TraceRecorder is TraceRecorder
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_recorder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracegate.recorder'`

- [ ] **Step 3: Write the implementation**

`src/tracegate/recorder.py`:

```python
"""TraceRecorder — builds an AgentTrace in memory and persists it on finish."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tracegate.schema import (
    AgentInfo,
    AgentTrace,
    LLMCallStep,
    TaskSpec,
    TokenUsage,
    ToolCallStep,
)
from tracegate.store import default_store


class TraceRecorder:
    def __init__(
        self,
        task: str | TaskSpec,
        framework: str,
        model: str | None = None,
        store: Any | None = None,
    ) -> None:
        if isinstance(task, str):
            task = TaskSpec(description=task)
        self.trace = AgentTrace(
            task=task, agent=AgentInfo(framework=framework, model=model)
        )
        self._store = store if store is not None else default_store()
        self._finished = False

    def record_llm_call(
        self,
        *,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> LLMCallStep:
        step = LLMCallStep(
            index=len(self.trace.steps),
            model=model,
            prompt=prompt,
            response=response,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )
        self.trace.steps.append(step)
        return step

    def record_tool_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: str | None = None,
        latency_ms: float | None = None,
    ) -> ToolCallStep:
        step = ToolCallStep(
            index=len(self.trace.steps),
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            error=error,
            latency_ms=latency_ms,
        )
        self.trace.steps.append(step)
        return step

    def finish(self) -> AgentTrace:
        if not self._finished:
            self._finished = True
            self.trace.ended_at = datetime.now(timezone.utc)
            self._store.save(self.trace)
        return self.trace

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()
```

Replace `src/tracegate/__init__.py` with:

```python
"""TraceGate — the black box flight recorder for AI agents."""

from tracegate.recorder import TraceRecorder
from tracegate.schema import (
    AgentInfo,
    AgentTrace,
    LLMCallStep,
    TaskSpec,
    TokenUsage,
    ToolCallStep,
)

__version__ = "0.1.0"

__all__ = [
    "AgentInfo",
    "AgentTrace",
    "LLMCallStep",
    "TaskSpec",
    "TokenUsage",
    "ToolCallStep",
    "TraceRecorder",
    "__version__",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_recorder.py -v`
Expected: 6 passed. Also run full suite: `venv/bin/pytest -v` — all passed.

- [ ] **Step 5: Commit**

```bash
git add src/tracegate/recorder.py src/tracegate/__init__.py tests/test_recorder.py
git commit -m "feat: TraceRecorder with context-manager finish and default store

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Claude Agent SDK adapter

**Files:**
- Create: `src/tracegate/adapters/__init__.py` (empty)
- Create: `src/tracegate/adapters/claude_agent_sdk.py`
- Test: `tests/test_adapter_claude.py`

**Interfaces:**
- Consumes: `TraceRecorder` (Task 5).
- Produces (`tracegate.adapters.claude_agent_sdk`):
  - `ClaudeAgentAdapter(recorder: TraceRecorder)` with `handle_message(message: Any) -> None`. Dispatch is on `type(message).__name__` (`AssistantMessage`, `UserMessage`, `ResultMessage`) and duck-typed block attributes — the module must NOT import `claude_agent_sdk`.
  - `record_stream(stream: AsyncIterator, recorder: TraceRecorder) -> AsyncIterator` — async generator that feeds every message through an adapter, yields it unchanged, and calls `recorder.finish()` when the stream ends.
- Message semantics: `AssistantMessage.content` is a list of blocks — a block with `.text` is assistant text (recorded as one `LLMCallStep` per assistant message, `prompt=""`, model from `message.model` else recorder's model else `"unknown"`); a block with `.name` and `.input` is a tool use (held pending by `block.id`). `UserMessage.content` blocks with `.tool_use_id` are tool results — matched to pending tool uses and recorded as `ToolCallStep` (`block.is_error` truthy → `error=str(block.content)`, else `result=block.content`). `ResultMessage` contributes `metadata["result"]` and `metadata["total_cost_usd"]` when present.

- [ ] **Step 1: Write the failing tests**

`tests/test_adapter_claude.py`:

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

from tracegate.adapters.claude_agent_sdk import ClaudeAgentAdapter, record_stream
from tracegate.recorder import TraceRecorder
from tracegate.schema import LLMCallStep, ToolCallStep


# Stubs mirroring claude-agent-sdk message/block shapes (name-based dispatch).
@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: Any = None
    is_error: bool = False


@dataclass
class AssistantMessage:
    content: list
    model: str = "claude-sonnet-5"


@dataclass
class UserMessage:
    content: Any = None


@dataclass
class ResultMessage:
    result: str = "done"
    total_cost_usd: float = 0.01


class FakeStore:
    def __init__(self):
        self.saved = []

    def save(self, trace):
        self.saved.append(trace)


def make_recorder() -> TraceRecorder:
    return TraceRecorder(task="t", framework="claude-agent-sdk", store=FakeStore())


def test_assistant_text_becomes_llm_step():
    rec = make_recorder()
    ClaudeAgentAdapter(rec).handle_message(
        AssistantMessage(content=[TextBlock("hello"), TextBlock("world")])
    )
    assert len(rec.trace.steps) == 1
    step = rec.trace.steps[0]
    assert isinstance(step, LLMCallStep)
    assert step.response == "hello\nworld"
    assert step.model == "claude-sonnet-5"


def test_tool_use_then_result_becomes_tool_step():
    rec = make_recorder()
    adapter = ClaudeAgentAdapter(rec)
    adapter.handle_message(
        AssistantMessage(content=[ToolUseBlock(id="tu1", name="calc", input={"x": 2})])
    )
    assert rec.trace.steps == []  # pending until result arrives
    adapter.handle_message(
        UserMessage(content=[ToolResultBlock(tool_use_id="tu1", content="4")])
    )
    step = rec.trace.steps[0]
    assert isinstance(step, ToolCallStep)
    assert step.tool_name == "calc"
    assert step.arguments == {"x": 2}
    assert step.result == "4"
    assert step.error is None


def test_tool_error_recorded():
    rec = make_recorder()
    adapter = ClaudeAgentAdapter(rec)
    adapter.handle_message(
        AssistantMessage(content=[ToolUseBlock(id="tu1", name="calc", input={})])
    )
    adapter.handle_message(
        UserMessage(
            content=[ToolResultBlock(tool_use_id="tu1", content="boom", is_error=True)]
        )
    )
    step = rec.trace.steps[0]
    assert step.error == "boom"
    assert step.result is None


def test_unmatched_tool_result_ignored():
    rec = make_recorder()
    ClaudeAgentAdapter(rec).handle_message(
        UserMessage(content=[ToolResultBlock(tool_use_id="ghost")])
    )
    assert rec.trace.steps == []


def test_non_list_user_content_ignored():
    rec = make_recorder()
    ClaudeAgentAdapter(rec).handle_message(UserMessage(content="plain string"))
    assert rec.trace.steps == []


def test_result_message_fills_metadata():
    rec = make_recorder()
    ClaudeAgentAdapter(rec).handle_message(ResultMessage())
    assert rec.trace.metadata["result"] == "done"
    assert rec.trace.metadata["total_cost_usd"] == 0.01


def test_record_stream_yields_and_finishes():
    store = FakeStore()
    rec = TraceRecorder(task="t", framework="claude-agent-sdk", store=store)

    async def fake_query():
        yield AssistantMessage(content=[TextBlock("hi")])
        yield ResultMessage()

    async def consume():
        return [m async for m in record_stream(fake_query(), rec)]

    messages = asyncio.run(consume())
    assert len(messages) == 2
    assert len(store.saved) == 1
    assert store.saved[0].ended_at is not None
    assert len(store.saved[0].steps) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_adapter_claude.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracegate.adapters'`

- [ ] **Step 3: Write the implementation**

`src/tracegate/adapters/__init__.py`:

```python
"""Framework adapters that normalize agent events into AgentTrace steps."""
```

`src/tracegate/adapters/claude_agent_sdk.py`:

```python
"""Adapter for the Claude Agent SDK message stream.

Deliberately does not import claude_agent_sdk: dispatch is by message class
name and duck-typed block attributes, so tracing adds no hard dependency and
tests run without the SDK, an API key, or network access.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from tracegate.recorder import TraceRecorder


class ClaudeAgentAdapter:
    def __init__(self, recorder: TraceRecorder) -> None:
        self._recorder = recorder
        self._pending_tools: dict[str, dict[str, Any]] = {}

    def handle_message(self, message: Any) -> None:
        handler = {
            "AssistantMessage": self._handle_assistant,
            "UserMessage": self._handle_user,
            "ResultMessage": self._handle_result,
        }.get(type(message).__name__)
        if handler is not None:
            handler(message)

    def _handle_assistant(self, message: Any) -> None:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return
        texts: list[str] = []
        for block in content:
            if hasattr(block, "text"):
                texts.append(block.text)
            elif hasattr(block, "name") and hasattr(block, "input"):
                block_id = getattr(block, "id", f"pending_{len(self._pending_tools)}")
                self._pending_tools[block_id] = {
                    "tool_name": block.name,
                    "arguments": dict(block.input or {}),
                    "started": time.monotonic(),
                }
        if texts:
            model = (
                getattr(message, "model", None)
                or self._recorder.trace.agent.model
                or "unknown"
            )
            self._recorder.record_llm_call(
                model=model, prompt="", response="\n".join(texts)
            )

    def _handle_user(self, message: Any) -> None:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return
        for block in content:
            tool_use_id = getattr(block, "tool_use_id", None)
            if tool_use_id is None:
                continue
            pending = self._pending_tools.pop(tool_use_id, None)
            if pending is None:
                continue
            latency_ms = (time.monotonic() - pending["started"]) * 1000
            is_error = bool(getattr(block, "is_error", False))
            result = getattr(block, "content", None)
            self._recorder.record_tool_call(
                tool_name=pending["tool_name"],
                arguments=pending["arguments"],
                result=None if is_error else result,
                error=str(result) if is_error else None,
                latency_ms=latency_ms,
            )

    def _handle_result(self, message: Any) -> None:
        result = getattr(message, "result", None)
        if result is not None:
            self._recorder.trace.metadata["result"] = result
        cost = getattr(message, "total_cost_usd", None)
        if cost is not None:
            self._recorder.trace.metadata["total_cost_usd"] = cost


async def record_stream(
    stream: AsyncIterator[Any], recorder: TraceRecorder
) -> AsyncIterator[Any]:
    """Wrap a claude_agent_sdk.query() stream, recording as messages flow.

    Usage:
        recorder = TraceRecorder(task=prompt, framework="claude-agent-sdk")
        async for message in record_stream(query(prompt=prompt), recorder):
            ...  # use the message exactly as before
    """
    adapter = ClaudeAgentAdapter(recorder)
    try:
        async for message in stream:
            adapter.handle_message(message)
            yield message
    finally:
        recorder.finish()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_adapter_claude.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/tracegate/adapters/ tests/test_adapter_claude.py
git commit -m "feat: Claude Agent SDK adapter (duck-typed, no hard dependency)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: CLI — `list` and `show`

**Files:**
- Create: `src/tracegate/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `default_store()`, `JSONLTraceStore` (Task 3); `AgentTrace` (Task 2).
- Produces: Typer app `tracegate.cli.app` (referenced by the `tracegate` console script from Task 1) with commands:
  - `tracegate list [--store-dir PATH]` — one line per trace: `trace_id  started_at  n_steps  task description`. `--store-dir` defaults to env `TRACEGATE_DIR` else `.tracegate`.
  - `tracegate show TRACE_ID [--store-dir PATH]` — header (trace id, task, framework/model, started/ended) plus one line per step; exits code 1 with `trace not found: <id>` on stderr-style output for missing ids.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py` (the `record` test lands in Task 8; start with these):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracegate.cli'`

- [ ] **Step 3: Write the implementation**

`src/tracegate/cli.py`:

```python
"""TraceGate CLI: record, show, list."""

from __future__ import annotations

from pathlib import Path

import typer

from tracegate.schema import AgentTrace, LLMCallStep, ToolCallStep
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_cli.py -v`
Expected: 4 passed. Also sanity-check the console script: `venv/bin/tracegate --help` prints the app help with `list` and `show`.

- [ ] **Step 5: Commit**

```bash
git add src/tracegate/cli.py tests/test_cli.py
git commit -m "feat: CLI list and show commands

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: CLI `record` + demo example + README quickstart

**Files:**
- Modify: `src/tracegate/cli.py` (add `record` command)
- Create: `examples/demo_agent.py`
- Modify: `README.md` (quickstart)
- Test: `tests/test_cli.py` (append tests)

**Interfaces:**
- Consumes: `app`, `_store` (Task 7); `TraceRecorder` (Task 5); env-var contract `TRACEGATE_DIR` (Task 3).
- Produces: `tracegate record SCRIPT [ARGS...] [--store-dir PATH]` — runs `sys.executable SCRIPT ARGS...` as a subprocess with `TRACEGATE_DIR` set to the store dir, propagates the script's exit code, and on success prints how many traces the store now holds. `examples/demo_agent.py` — runnable simulated agent (no API key needed) that records a 5-step trace via `TraceRecorder`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_record_runs_script_and_stores_trace(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text(
        "from tracegate import TraceRecorder\n"
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_cli.py -v`
Expected: the two new tests FAIL (`record` command does not exist → exit code 2 / usage error); the four Task 7 tests still pass.

- [ ] **Step 3: Write the implementation**

Add to `src/tracegate/cli.py` (imports `os`, `subprocess`, `sys` at top of file alongside existing imports):

```python
import os
import subprocess
import sys


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
```

`examples/demo_agent.py`:

```python
"""Simulated 5-step agent run — records a trace without any API key.

Run:  tracegate record examples/demo_agent.py
Then: tracegate list && tracegate show <trace_id>
"""

from tracegate import TraceRecorder

with TraceRecorder(
    task="Find the cheapest flight SFO->NYC under $500 next Friday",
    framework="demo",
    model="simulated",
) as rec:
    rec.record_llm_call(
        model="simulated",
        prompt="Find the cheapest flight SFO->NYC under $500 next Friday",
        response="I'll search for flights, then filter by price.",
    )
    rec.record_tool_call(
        tool_name="search_flights",
        arguments={"from": "SFO", "to": "NYC", "date": "next Friday"},
        result=[{"airline": "UA", "price": 420}, {"airline": "DL", "price": 510}],
        latency_ms=132.0,
    )
    rec.record_llm_call(
        model="simulated",
        prompt="",
        response="UA at $420 fits the budget. Checking seat availability.",
    )
    rec.record_tool_call(
        tool_name="check_seats",
        arguments={"airline": "UA", "price": 420},
        result={"available": True},
        latency_ms=87.0,
    )
    rec.record_llm_call(
        model="simulated",
        prompt="",
        response="Done: UA flight at $420, seats available.",
    )

print(f"recorded trace {rec.trace.trace_id} with {len(rec.trace.steps)} steps")
```

Replace `README.md` with:

```markdown
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
```

- [ ] **Step 4: Run the full suite and the demo end-to-end**

```bash
venv/bin/pytest -v
```
Expected: all tests pass (package 1 + schema 3 + jsonl 5 + sqlite 4 + recorder 6 + adapter 7 + cli 6 = 32).

```bash
venv/bin/tracegate record examples/demo_agent.py
venv/bin/tracegate list
```
Expected: demo prints its trace id; `list` shows 1 trace with "Find the cheapest flight". Then `venv/bin/tracegate show <that id>` renders 5 steps. Clean up: `rm -rf .tracegate`.

- [ ] **Step 5: Commit**

```bash
git add src/tracegate/cli.py examples/ README.md tests/test_cli.py
git commit -m "feat: CLI record command, demo agent, README quickstart

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
