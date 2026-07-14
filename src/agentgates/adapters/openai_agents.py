"""Adapter for the OpenAI Agents SDK run items.

Deliberately does not import the openai-agents package: dispatch is by run-item
class name and duck-typed attributes, so tracing adds no hard dependency and
tests run without the SDK installed.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agentgates.recorder import TraceRecorder
from agentgates.schema import AgentTrace


def _parse_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"raw": raw}
    return {}


def _call_id_of(raw_item: Any) -> Any:
    if isinstance(raw_item, dict):
        return raw_item.get("call_id")
    return getattr(raw_item, "call_id", None)


class OpenAIAgentsAdapter:
    def __init__(self, recorder: TraceRecorder) -> None:
        self._recorder = recorder
        self._pending_tools: dict[Any, dict[str, Any]] = {}

    def handle_item(self, item: Any) -> None:
        handler = {
            "MessageOutputItem": self._handle_message,
            "ToolCallItem": self._handle_tool_call,
            "ToolCallOutputItem": self._handle_tool_output,
        }.get(type(item).__name__)
        if handler is not None:
            handler(item)

    def _handle_message(self, item: Any) -> None:
        raw = getattr(item, "raw_item", None)
        texts = [
            block.text
            for block in getattr(raw, "content", None) or []
            if getattr(block, "text", None)
        ]
        if texts:
            model = self._recorder.trace.agent.model or "unknown"
            self._recorder.record_llm_call(
                model=model, prompt="", response="\n".join(texts)
            )

    def _handle_tool_call(self, item: Any) -> None:
        raw = getattr(item, "raw_item", None)
        call_id = _call_id_of(raw) or f"pending_{len(self._pending_tools)}"
        self._pending_tools[call_id] = {
            "tool_name": getattr(raw, "name", "unknown"),
            "arguments": _parse_arguments(getattr(raw, "arguments", None)),
            "started": time.monotonic(),
        }

    def _handle_tool_output(self, item: Any) -> None:
        call_id = _call_id_of(getattr(item, "raw_item", None))
        pending = self._pending_tools.pop(call_id, None)
        if pending is None:
            return
        latency_ms = (time.monotonic() - pending["started"]) * 1000
        self._recorder.record_tool_call(
            tool_name=pending["tool_name"],
            arguments=pending["arguments"],
            result=getattr(item, "output", None),
            latency_ms=latency_ms,
        )


def record_run(result: Any, recorder: TraceRecorder) -> AgentTrace:
    """Record a completed Runner.run(...) result into the recorder's trace.

    Usage:
        recorder = TraceRecorder(task=task, framework="openai-agents")
        result = await Runner.run(agent, task)
        record_run(result, recorder)
    """
    adapter = OpenAIAgentsAdapter(recorder)
    for item in getattr(result, "new_items", None) or []:
        adapter.handle_item(item)
    final_output = getattr(result, "final_output", None)
    if final_output is not None:
        recorder.trace.metadata["result"] = str(final_output)
    return recorder.finish()
