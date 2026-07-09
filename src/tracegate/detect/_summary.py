"""Shared plain-text step summaries for judge prompts."""

from __future__ import annotations

from tracegate.schema import LLMCallStep, ToolCallStep


def summarize_step(step) -> str:
    if isinstance(step, LLMCallStep):
        return f"[{step.index}] llm: {step.response[:200]}"
    if isinstance(step, ToolCallStep):
        outcome = f"ERROR {step.error}" if step.error else str(step.result)[:200]
        return f"[{step.index}] tool {step.tool_name}({step.arguments}) -> {outcome}"
    return f"[{step.index}] {step.type}"
