from agentgates.schema import (
    AgentInfo,
    AgentTrace,
    LLMCallStep,
    TaskSpec,
    ToolCallStep,
)


def make_trace() -> AgentTrace:
    trace = AgentTrace(
        task=TaskSpec(description="book a flight", constraints=["under $500"]),
        agent=AgentInfo(framework="claude-agent-sdk", model="claude-sonnet-5"),
    )
    trace.steps.append(
        LLMCallStep(index=0, model="claude-sonnet-5", prompt="hi", response="hello")
    )
    trace.steps.append(
        ToolCallStep(
            index=1,
            tool_name="search_flights",
            arguments={"to": "NYC"},
            result={"flights": 3},
            latency_ms=41.5,
        )
    )
    return trace


def test_trace_defaults():
    trace = make_trace()
    assert len(trace.trace_id) == 32  # uuid4 hex
    assert trace.started_at is not None
    assert trace.ended_at is None
    assert trace.metadata == {}


def test_step_discriminated_union_roundtrip():
    trace = make_trace()
    restored = AgentTrace.model_validate_json(trace.model_dump_json())
    assert isinstance(restored.steps[0], LLMCallStep)
    assert isinstance(restored.steps[1], ToolCallStep)
    assert restored.steps[1].tool_name == "search_flights"
    assert restored.steps[1].result == {"flights": 3}
    assert restored.trace_id == trace.trace_id


def test_tool_step_error_fields():
    step = ToolCallStep(index=0, tool_name="x", arguments={}, error="boom")
    assert step.result is None
    assert step.error == "boom"
