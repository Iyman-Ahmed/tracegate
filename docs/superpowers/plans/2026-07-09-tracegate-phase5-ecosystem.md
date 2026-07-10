# TraceGate Phase 5 (Ecosystem) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Framework reach: a LangGraph adapter, an OpenAI Agents SDK adapter, and a `pytest` plugin exposing a `trace_recorder` fixture — all duck-typed with zero hard dependencies, mirroring the Claude adapter's pattern.

**Architecture:** Both adapters dispatch on `type(obj).__name__` and duck-typed attributes so neither `langchain`/`langgraph` nor `openai-agents` is imported. LangGraph: `AIMessage` (text + `tool_calls` list of `{name, args, id}` dicts) and `ToolMessage` (`content`, `tool_call_id`, `status`), consumed either per-message or via `record_langgraph_stream` over LangGraph's `stream()` update dicts (`{node: {"messages": [...]}}`). OpenAI Agents: `RunItem` classes `MessageOutputItem` / `ToolCallItem` / `ToolCallOutputItem` via `record_run(result, recorder)` over `result.new_items` + `result.final_output`. The pytest plugin registers via the `pytest11` entry point and yields a finished-on-teardown `TraceRecorder` named after the test.

## Global Constraints

- No imports of langgraph/langchain/openai-agents anywhere; tests use stub classes with matching names/attributes.
- `pytest11` entry point requires re-running `pip install -e ".[dev]"` after editing pyproject.
- `tests/conftest.py` sets `pytest_plugins = "pytester"` for plugin tests.
- Branch `phases-2-5`; commit trailer as before.

### Task 1: LangGraph adapter — `src/tracegate/adapters/langgraph.py`, tests `tests/test_adapter_langgraph.py`
Produces `LangGraphAdapter(recorder)` with `handle_message(msg)` (AIMessage text→LLMCallStep; AIMessage.tool_calls→pending by id; ToolMessage→ToolCallStep, `status=="error"`→error) and `record_langgraph_stream(stream, recorder)` sync generator yielding updates unchanged, `finish()` in `finally`.

### Task 2: OpenAI Agents adapter — `src/tracegate/adapters/openai_agents.py`, tests `tests/test_adapter_openai_agents.py`
Produces `OpenAIAgentsAdapter(recorder)` with `handle_item(item)` and `record_run(result, recorder) -> AgentTrace` (iterates `result.new_items`, sets `metadata["result"] = str(result.final_output)`, finishes). ToolCallItem raw args may be a JSON string → parse, fallback `{"raw": s}`.

### Task 3: pytest plugin — `src/tracegate/pytest_plugin.py`, entry point in pyproject, tests `tests/test_pytest_plugin.py` (+ `tests/conftest.py`)
Fixture `trace_recorder`: `TraceRecorder(task=<test name>, framework="pytest")`, finished on teardown, saved to the default store (`TRACEGATE_DIR`). Verified via `pytester.runpytest_inprocess()` with `TRACEGATE_DIR` monkeypatched.

### Task 4: README ecosystem section + full-suite verification + commit per task.
