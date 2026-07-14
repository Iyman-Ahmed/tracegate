from agentgates.detect import Finding
from agentgates.gate.score import CaseResult, RunResult, SuiteResult
from agentgates.report import render_html, render_terminal


def result_fixture() -> SuiteResult:
    finding = Finding(detector="loop", severity="error", message="<looped>", step_index=0)
    return SuiteResult(
        suite_name="my suite",
        cases=[
            CaseResult(case_id="good", runs=[RunResult(success=True)]),
            CaseResult(case_id="bad", runs=[RunResult(success=False, findings=[finding])]),
        ],
    )


def test_terminal_report():
    text = render_terminal(result_fixture())
    assert "good" in text
    assert "bad" in text
    assert "1/1" in text
    assert "reliability score" in text.lower()


def test_html_report_escapes_and_includes_score():
    html = render_html(result_fixture())
    assert html.startswith("<!doctype html>")
    assert "my suite" in html
    assert "&lt;looped&gt;" in html   # escaped finding message
    assert "<looped>" not in html
    assert "good" in html and "bad" in html
