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
