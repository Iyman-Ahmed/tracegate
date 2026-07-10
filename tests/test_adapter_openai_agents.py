from dataclasses import dataclass, field
from typing import Any

from tracegate.adapters.openai_agents import OpenAIAgentsAdapter, record_run
from tracegate.recorder import TraceRecorder
from tracegate.schema import LLMCallStep, ToolCallStep


# Stubs mirroring openai-agents RunItem shapes (name-based dispatch).
@dataclass
class _TextBlock:
    text: str


@dataclass
class _RawMessage:
    content: list


@dataclass
class _RawToolCall:
    name: str
    arguments: str  # the SDK serializes args as a JSON string
    call_id: str


@dataclass
class MessageOutputItem:
    raw_item: Any


@dataclass
class ToolCallItem:
    raw_item: Any


@dataclass
class ToolCallOutputItem:
    raw_item: Any
    output: Any = None


@dataclass
class FakeRunResult:
    new_items: list = field(default_factory=list)
    final_output: Any = None


class FakeStore:
    def __init__(self):
        self.saved = []

    def save(self, trace):
        self.saved.append(trace)


def make_recorder(store=None) -> TraceRecorder:
    return TraceRecorder(
        task="t", framework="openai-agents", model="gpt-x", store=store or FakeStore()
    )


def test_message_output_becomes_llm_step():
    rec = make_recorder()
    item = MessageOutputItem(raw_item=_RawMessage(content=[_TextBlock("hello"), _TextBlock("world")]))
    OpenAIAgentsAdapter(rec).handle_item(item)
    step = rec.trace.steps[0]
    assert isinstance(step, LLMCallStep)
    assert step.response == "hello\nworld"
    assert step.model == "gpt-x"


def test_tool_call_then_output():
    rec = make_recorder()
    adapter = OpenAIAgentsAdapter(rec)
    adapter.handle_item(
        ToolCallItem(raw_item=_RawToolCall(name="search", arguments='{"q": "x"}', call_id="c1"))
    )
    assert rec.trace.steps == []  # pending until output
    adapter.handle_item(
        ToolCallOutputItem(raw_item={"call_id": "c1"}, output="found it")
    )
    step = rec.trace.steps[0]
    assert isinstance(step, ToolCallStep)
    assert step.tool_name == "search"
    assert step.arguments == {"q": "x"}
    assert step.result == "found it"


def test_unparseable_arguments_kept_raw():
    rec = make_recorder()
    adapter = OpenAIAgentsAdapter(rec)
    adapter.handle_item(
        ToolCallItem(raw_item=_RawToolCall(name="s", arguments="not json", call_id="c1"))
    )
    adapter.handle_item(ToolCallOutputItem(raw_item={"call_id": "c1"}, output="ok"))
    assert rec.trace.steps[0].arguments == {"raw": "not json"}


def test_orphan_output_ignored():
    rec = make_recorder()
    OpenAIAgentsAdapter(rec).handle_item(
        ToolCallOutputItem(raw_item={"call_id": "ghost"}, output="x")
    )
    assert rec.trace.steps == []


def test_record_run_processes_items_and_finishes():
    store = FakeStore()
    rec = make_recorder(store)
    result = FakeRunResult(
        new_items=[
            ToolCallItem(raw_item=_RawToolCall(name="s", arguments="{}", call_id="c1")),
            ToolCallOutputItem(raw_item={"call_id": "c1"}, output="data"),
            MessageOutputItem(raw_item=_RawMessage(content=[_TextBlock("done")])),
        ],
        final_output="the answer is 42",
    )
    trace = record_run(result, rec)
    assert len(store.saved) == 1
    assert trace.metadata["result"] == "the answer is 42"
    assert [s.type for s in trace.steps] == ["tool_call", "llm_call"]
    assert trace.ended_at is not None
