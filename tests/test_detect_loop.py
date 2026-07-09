from tracegate.detect import Finding, run_detectors
from tracegate.detect.loop import LoopDetector
from tracegate.schema import AgentInfo, AgentTrace, LLMCallStep, TaskSpec, ToolCallStep


def make_trace(steps) -> AgentTrace:
    trace = AgentTrace(task=TaskSpec(description="t"), agent=AgentInfo(framework="test"))
    trace.steps = steps
    return trace


def tool(i, name="search", args=None):
    return ToolCallStep(index=i, tool_name=name, arguments=args or {"q": "x"}, result="ok")


def llm(i, response="thinking"):
    return LLMCallStep(index=i, model="m", prompt="", response=response)


def test_clean_trace_no_findings():
    trace = make_trace([llm(0, "a"), tool(1, "search"), llm(2, "b"), tool(3, "book", {"id": 1})])
    assert LoopDetector().detect(trace) == []


def test_consecutive_identical_steps_flagged_as_error():
    trace = make_trace([tool(0), tool(1), tool(2), llm(3, "done")])
    findings = LoopDetector(threshold=3).detect(trace)
    assert len(findings) == 1
    f = findings[0]
    assert f.detector == "loop"
    assert f.severity == "error"
    assert f.step_index == 0
    assert "3" in f.message


def test_below_threshold_not_flagged():
    trace = make_trace([tool(0), tool(1), llm(2, "done")])
    assert LoopDetector(threshold=3).detect(trace) == []


def test_identical_llm_responses_count_as_loop():
    trace = make_trace([llm(0, "same"), llm(1, "same"), llm(2, "same")])
    findings = LoopDetector().detect(trace)
    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_nonconsecutive_repeated_tool_calls_warned():
    trace = make_trace([tool(0), llm(1, "a"), tool(2), llm(3, "b"), tool(4)])
    findings = LoopDetector().detect(trace)
    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].detector == "loop"


def test_different_args_not_a_loop():
    trace = make_trace([tool(0, args={"q": 1}), tool(1, args={"q": 2}), tool(2, args={"q": 3})])
    assert LoopDetector().detect(trace) == []


def test_run_detectors_sorts_by_step_index():
    class A:
        name = "a"

        def detect(self, trace):
            return [Finding(detector="a", severity="info", message="x", step_index=5)]

    class B:
        name = "b"

        def detect(self, trace):
            return [Finding(detector="b", severity="info", message="y", step_index=1)]

    trace = make_trace([])
    findings = run_detectors(trace, [A(), B()])
    assert [f.step_index for f in findings] == [1, 5]
