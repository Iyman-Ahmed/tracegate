from agentgates.detect import Finding
from agentgates.gate.score import CaseResult, RunResult, SuiteResult


def err(msg="e"):
    return Finding(detector="loop", severity="error", message=msg)


def warn():
    return Finding(detector="loop", severity="warning", message="w")


def test_case_success_rate():
    case = CaseResult(case_id="a", runs=[RunResult(success=True), RunResult(success=False)])
    assert case.success_rate == 0.5


def test_perfect_suite_scores_one():
    result = SuiteResult(
        suite_name="s",
        cases=[CaseResult(case_id="a", runs=[RunResult(success=True)])],
    )
    assert result.reliability_score == 1.0


def test_error_findings_penalize():
    # 2 cases: one perfect, one failed with 2 error findings in its single run.
    result = SuiteResult(
        suite_name="s",
        cases=[
            CaseResult(case_id="a", runs=[RunResult(success=True)]),
            CaseResult(case_id="b", runs=[RunResult(success=False, findings=[err(), err()])]),
        ],
    )
    # success mean 0.5; errors/run = 2/2 = 1.0; penalty 0.1 -> 0.5 * 0.9
    assert abs(result.reliability_score - 0.45) < 1e-9


def test_warnings_do_not_penalize():
    result = SuiteResult(
        suite_name="s",
        cases=[CaseResult(case_id="a", runs=[RunResult(success=True, findings=[warn()])])],
    )
    assert result.reliability_score == 1.0


def test_penalty_capped_at_half():
    findings = [err() for _ in range(20)]
    result = SuiteResult(
        suite_name="s",
        cases=[CaseResult(case_id="a", runs=[RunResult(success=True, findings=findings)])],
    )
    assert abs(result.reliability_score - 0.5) < 1e-9


def test_empty_suite_scores_zero():
    assert SuiteResult(suite_name="s", cases=[]).reliability_score == 0.0
