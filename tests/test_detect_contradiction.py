from tracegate.detect import default_detectors
from tracegate.detect.contradiction import ContradictionDetector
from tracegate.schema import AgentInfo, AgentTrace, LLMCallStep, TaskSpec, ToolCallStep


def make_trace(steps) -> AgentTrace:
    trace = AgentTrace(task=TaskSpec(description="t"), agent=AgentInfo(framework="test"))
    trace.steps = steps
    return trace


def tool(i, name="get_price", args=None, result=None, error=None):
    return ToolCallStep(
        index=i,
        tool_name=name,
        arguments=args or {"id": "UA420"},
        result=result,
        error=error,
    )


def llm(i, response="thinking"):
    return LLMCallStep(index=i, model="m", prompt="", response=response)


def test_clean_trace_no_findings():
    trace = make_trace([tool(0, result=420), llm(1, "booked at $420")])
    assert ContradictionDetector().detect(trace) == []


def test_identical_results_no_finding():
    trace = make_trace([tool(0, result=420), tool(1, result=420)])
    assert ContradictionDetector().detect(trace) == []


def test_different_args_no_finding():
    trace = make_trace([
        tool(0, args={"id": "UA420"}, result=420),
        tool(1, args={"id": "DL999"}, result=580),
    ])
    assert ContradictionDetector().detect(trace) == []


def test_conflicting_results_warn_at_later_step():
    trace = make_trace([tool(0, result=420), tool(1, result=580)])
    findings = ContradictionDetector().detect(trace)
    assert len(findings) == 1
    f = findings[0]
    assert f.detector == "contradiction"
    assert f.severity == "warning"
    assert f.step_index == 1
    assert "420" in f.message and "580" in f.message


def test_errored_calls_ignored():
    trace = make_trace([
        tool(0, result=420),
        tool(1, error="timeout"),
        tool(2, result=420),
    ])
    assert ContradictionDetector().detect(trace) == []


def test_acting_on_superseded_value_is_error():
    trace = make_trace([
        tool(0, result=420),
        tool(1, result=580),
        llm(2, "Booked! Total charged: $420."),
    ])
    findings = ContradictionDetector().detect(trace)
    assert [f.severity for f in findings] == ["warning", "error"]
    error = findings[1]
    assert error.step_index == 2
    assert "420" in error.message
    assert "superseded" in error.message.lower()


def test_quoting_fresh_value_only_warns():
    trace = make_trace([
        tool(0, result=420),
        tool(1, result=580),
        llm(2, "The fare rose to $580. Confirming."),
    ])
    findings = ContradictionDetector().detect(trace)
    assert [f.severity for f in findings] == ["warning"]


def test_llm_before_the_conflict_is_not_flagged():
    trace = make_trace([
        tool(0, result=420),
        llm(1, "the fare is $420"),  # true at this point in the run
        tool(2, result=580),
    ])
    findings = ContradictionDetector().detect(trace)
    assert [f.severity for f in findings] == ["warning"]


def test_non_scalar_results_warn_without_escalation():
    trace = make_trace([
        tool(0, result={"price": 420}),
        tool(1, result={"price": 580}),
        llm(2, "still going with 420"),
    ])
    findings = ContradictionDetector().detect(trace)
    assert [f.severity for f in findings] == ["warning"]


def test_in_default_detectors():
    names = [d.name for d in default_detectors()]
    assert names == ["loop", "tool_misuse", "contradiction"]
