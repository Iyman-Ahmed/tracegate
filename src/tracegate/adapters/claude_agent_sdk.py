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
