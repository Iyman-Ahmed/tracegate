from dataclasses import dataclass, field
from typing import Any

from agentgates.adapters.langgraph import LangGraphAdapter, record_langgraph_stream
from agentgates.recorder import TraceRecorder
from agentgates.schema import LLMCallStep, ToolCallStep


# Stubs mirroring langchain-core message shapes (name-based dispatch).
@dataclass
class AIMessage:
    content: Any = ""
    tool_calls: list = field(default_factory=list)


@dataclass
class ToolMessage:
    content: Any = None
    tool_call_id: str = ""
    status: str = "success"


@dataclass
class HumanMessage:
    content: str = ""


class FakeStore:
    def __init__(self):
        self.saved = []

    def save(self, trace):
        self.saved.append(trace)


def make_recorder(store=None) -> TraceRecorder:
    return TraceRecorder(task="t", framework="langgraph", model="gpt-x", store=store or FakeStore())


def test_ai_text_becomes_llm_step():
    rec = make_recorder()
    LangGraphAdapter(rec).handle_message(AIMessage(content="hello there"))
    step = rec.trace.steps[0]
    assert isinstance(step, LLMCallStep)
    assert step.response == "hello there"
    assert step.model == "gpt-x"  # falls back to recorder's model


def test_tool_call_then_tool_message():
    rec = make_recorder()
    adapter = LangGraphAdapter(rec)
    adapter.handle_message(
        AIMessage(tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "tc1"}])
    )
    assert rec.trace.steps == []  # pending until result
    adapter.handle_message(ToolMessage(content="found it", tool_call_id="tc1"))
    step = rec.trace.steps[0]
    assert isinstance(step, ToolCallStep)
    assert step.tool_name == "search"
    assert step.arguments == {"q": "x"}
    assert step.result == "found it"
    assert step.error is None


def test_tool_error_status():
    rec = make_recorder()
    adapter = LangGraphAdapter(rec)
    adapter.handle_message(AIMessage(tool_calls=[{"name": "s", "args": {}, "id": "tc1"}]))
    adapter.handle_message(ToolMessage(content="boom", tool_call_id="tc1", status="error"))
    step = rec.trace.steps[0]
    assert step.error == "boom"
    assert step.result is None


def test_unknown_and_human_messages_ignored():
    rec = make_recorder()
    adapter = LangGraphAdapter(rec)
    adapter.handle_message(HumanMessage(content="hi"))
    adapter.handle_message(ToolMessage(content="orphan", tool_call_id="ghost"))
    assert rec.trace.steps == []


def test_record_langgraph_stream():
    store = FakeStore()
    rec = make_recorder(store)

    updates = [
        {"agent": {"messages": [AIMessage(content="thinking", tool_calls=[{"name": "s", "args": {}, "id": "tc1"}])]}},
        {"tools": {"messages": [ToolMessage(content="ok", tool_call_id="tc1")]}},
        {"agent": {"messages": [AIMessage(content="done")]}},
    ]
    seen = list(record_langgraph_stream(iter(updates), rec))
    assert seen == updates  # passthrough
    assert len(store.saved) == 1
    trace = store.saved[0]
    assert trace.ended_at is not None
    types = [s.type for s in trace.steps]
    assert types == ["llm_call", "tool_call", "llm_call"]
