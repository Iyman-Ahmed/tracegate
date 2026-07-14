"""Adapter for LangGraph / LangChain message streams.

Deliberately does not import langgraph or langchain: dispatch is by message
class name and duck-typed attributes, so tracing adds no hard dependency and
tests run without either framework installed.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Iterator

from agentgates.recorder import TraceRecorder

_AI_MESSAGE_NAMES = {"AIMessage", "AIMessageChunk"}


class LangGraphAdapter:
    def __init__(self, recorder: TraceRecorder) -> None:
        self._recorder = recorder
        self._pending_tools: dict[str, dict[str, Any]] = {}

    def handle_message(self, message: Any) -> None:
        name = type(message).__name__
        if name in _AI_MESSAGE_NAMES:
            self._handle_ai(message)
        elif name == "ToolMessage":
            self._handle_tool_result(message)

    def _handle_ai(self, message: Any) -> None:
        for call in getattr(message, "tool_calls", None) or []:
            call_id = call.get("id") or f"pending_{len(self._pending_tools)}"
            self._pending_tools[call_id] = {
                "tool_name": call.get("name", "unknown"),
                "arguments": dict(call.get("args") or {}),
                "started": time.monotonic(),
            }
        content = getattr(message, "content", None)
        if content:
            model = (
                getattr(message, "response_metadata", {}).get("model_name")
                if isinstance(getattr(message, "response_metadata", None), dict)
                else None
            ) or self._recorder.trace.agent.model or "unknown"
            self._recorder.record_llm_call(
                model=model, prompt="", response=str(content)
            )

    def _handle_tool_result(self, message: Any) -> None:
        call_id = getattr(message, "tool_call_id", None)
        pending = self._pending_tools.pop(call_id, None)
        if pending is None:
            return
        latency_ms = (time.monotonic() - pending["started"]) * 1000
        is_error = getattr(message, "status", "success") == "error"
        content = getattr(message, "content", None)
        self._recorder.record_tool_call(
            tool_name=pending["tool_name"],
            arguments=pending["arguments"],
            result=None if is_error else content,
            error=str(content) if is_error else None,
            latency_ms=latency_ms,
        )


def record_langgraph_stream(
    stream: Iterable[Any], recorder: TraceRecorder
) -> Iterator[Any]:
    """Wrap a LangGraph graph.stream() iterator, recording as updates flow.

    Usage:
        recorder = TraceRecorder(task=task, framework="langgraph")
        for update in record_langgraph_stream(graph.stream(inputs), recorder):
            ...  # use the update exactly as before
    """
    adapter = LangGraphAdapter(recorder)
    try:
        for update in stream:
            if isinstance(update, dict):
                for node_payload in update.values():
                    if isinstance(node_payload, dict):
                        for message in node_payload.get("messages") or []:
                            adapter.handle_message(message)
            yield update
    finally:
        recorder.finish()
