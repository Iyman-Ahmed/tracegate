import asyncio
from dataclasses import dataclass
from typing import Any

from agentgates.adapters.claude_agent_sdk import ClaudeAgentAdapter, record_stream
from agentgates.recorder import TraceRecorder
from agentgates.schema import LLMCallStep, ToolCallStep


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
