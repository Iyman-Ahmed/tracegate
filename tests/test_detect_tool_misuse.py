from tracegate.detect import default_detectors
from tracegate.detect.tool_misuse import ToolMisuseDetector
from tracegate.schema import AgentInfo, AgentTrace, TaskSpec, ToolCallStep


def make_trace(steps) -> AgentTrace:
    trace = AgentTrace(task=TaskSpec(description="t"), agent=AgentInfo(framework="test"))
    trace.steps = steps
    return trace


def tool(i, name="fetch", args=None, error=None, result="ok"):
    return ToolCallStep(
        index=i,
        tool_name=name,
        arguments=args or {},
        result=None if error else result,
        error=error,
    )


def test_clean_trace_no_findings():
    trace = make_trace([tool(0), tool(1, name="save")])
    assert ToolMisuseDetector().detect(trace) == []


def test_tool_error_is_warning():
    trace = make_trace([tool(0, error="timeout")])
    findings = ToolMisuseDetector().detect(trace)
    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].step_index == 0
    assert "timeout" in findings[0].message


def test_identical_retry_after_error_is_error():
    trace = make_trace([
        tool(0, args={"url": "x"}, error="500"),
        tool(1, args={"url": "x"}, error="500"),
    ])
    findings = ToolMisuseDetector().detect(trace)
    severities = [f.severity for f in findings]
    assert severities == ["warning", "error"]
    assert findings[1].step_index == 1
    assert "identical arguments" in findings[1].message


def test_retry_with_changed_args_is_only_warnings():
    trace = make_trace([
        tool(0, args={"url": "x"}, error="500"),
        tool(1, args={"url": "y"}, error="500"),
    ])
    findings = ToolMisuseDetector().detect(trace)
    assert [f.severity for f in findings] == ["warning", "warning"]


def test_successful_retry_not_flagged_as_misuse():
    trace = make_trace([
        tool(0, args={"url": "x"}, error="500"),
        tool(1, args={"url": "x"}),
    ])
    findings = ToolMisuseDetector().detect(trace)
    assert [f.severity for f in findings] == ["warning"]


def test_in_default_detectors():
    names = [d.name for d in default_detectors()]
    assert "tool_misuse" in names
