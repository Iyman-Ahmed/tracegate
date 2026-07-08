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
