"""Terminal and static-HTML reports for suite results (stdlib only)."""

from __future__ import annotations

from html import escape

from tracegate.gate.score import CaseResult, SuiteResult


def _finding_counts(case: CaseResult) -> tuple[int, int]:
    errors = warnings = 0
    for run in case.runs:
        for f in run.findings:
            if f.severity == "error":
                errors += 1
            elif f.severity == "warning":
                warnings += 1
    return errors, warnings


def render_terminal(result: SuiteResult) -> str:
    lines = [f"suite: {result.suite_name}"]
    for case in result.cases:
        passed = sum(1 for r in case.runs if r.success)
        errors, warnings = _finding_counts(case)
        lines.append(
            f"  {case.case_id:<24} passed {passed}/{len(case.runs)}"
            f"  findings: {errors} error / {warnings} warning"
        )
    lines.append(f"reliability score: {result.reliability_score:.2f}")
    return "\n".join(lines)


def render_html(result: SuiteResult) -> str:
    rows = []
    for case in result.cases:
        passed = sum(1 for r in case.runs if r.success)
        errors, warnings = _finding_counts(case)
        findings_html = "".join(
            f"<li class='{escape(f.severity)}'>[{escape(f.severity)}] "
            f"step {f.step_index if f.step_index is not None else '-'} "
            f"{escape(f.detector)}: {escape(f.message)}</li>"
            for run in case.runs
            for f in run.findings
        )
        run_errors = "".join(
            f"<li class='error'>run error: {escape(r.error)}</li>"
            for r in case.runs
            if r.error
        )
        rows.append(
            f"<tr><td>{escape(case.case_id)}</td>"
            f"<td>{passed}/{len(case.runs)}</td>"
            f"<td>{errors} error / {warnings} warning</td>"
            f"<td><ul>{findings_html}{run_errors}</ul></td></tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TraceGate report: {escape(result.suite_name)}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: .5rem; text-align: left; vertical-align: top; }}
.score {{ font-size: 1.4rem; margin: 1rem 0; }}
li.error {{ color: #b00020; }}
li.warning {{ color: #8a6d00; }}
ul {{ margin: 0; padding-left: 1.2rem; }}
</style></head><body>
<h1>TraceGate &mdash; {escape(result.suite_name)}</h1>
<p class="score">reliability score: <strong>{result.reliability_score:.2f}</strong>
 ({result.total_runs} runs)</p>
<table><tr><th>case</th><th>passed</th><th>findings</th><th>detail</th></tr>
{''.join(rows)}
</table></body></html>
"""
