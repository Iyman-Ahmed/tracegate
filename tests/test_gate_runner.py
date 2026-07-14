import pytest

from agentgates.gate.runner import resolve_runner, run_suite
from agentgates.gate.suite import Suite, SuiteCase


def suite_of(case: SuiteCase, runs_per_case: int = 1) -> Suite:
    return Suite(name="t", runs_per_case=runs_per_case, cases=[case])


def test_resolve_runner_module_spec():
    fn = resolve_runner("tests.fake_runner:good_runner")
    assert fn("x").steps[0].response.endswith("SUCCESS_TOKEN")


def test_resolve_runner_file_spec(tmp_path):
    script = tmp_path / "my_runner.py"
    script.write_text(
        "from tests.fake_runner import good_runner\n"
        "run = good_runner\n"
    )
    fn = resolve_runner(f"{script}:run")
    assert fn("x").steps


def test_resolve_runner_bad_spec():
    with pytest.raises(ValueError):
        resolve_runner("no_colon_here")


def test_run_suite_success():
    suite = suite_of(SuiteCase(id="a", task="t", expect_contains=["SUCCESS_TOKEN"]))
    result = run_suite(suite, resolve_runner("tests.fake_runner:good_runner"))
    assert result.reliability_score == 1.0
    assert result.cases[0].success_rate == 1.0


def test_run_suite_missing_expectation_fails():
    suite = suite_of(SuiteCase(id="a", task="t", expect_contains=["NOPE"]))
    result = run_suite(suite, resolve_runner("tests.fake_runner:good_runner"))
    assert result.cases[0].success_rate == 0.0


def test_run_suite_metadata_result_wins():
    suite = suite_of(SuiteCase(id="a", task="t", expect_contains=["42"]))
    result = run_suite(suite, resolve_runner("tests.fake_runner:metadata_runner"))
    assert result.cases[0].success_rate == 1.0


def test_run_suite_detector_findings_attached():
    suite = suite_of(SuiteCase(id="a", task="t"))
    result = run_suite(suite, resolve_runner("tests.fake_runner:loopy_runner"))
    case = result.cases[0]
    assert case.success_rate == 1.0  # no expectations -> outcome passes
    assert case.error_finding_count >= 1  # blind retries flagged
    assert result.reliability_score < 1.0


def test_run_suite_crash_is_failed_run_not_exception():
    suite = suite_of(SuiteCase(id="a", task="t"))
    result = run_suite(suite, resolve_runner("tests.fake_runner:crashing_runner"))
    run = result.cases[0].runs[0]
    assert run.success is False
    assert "RuntimeError" in run.error


def test_runs_per_case_respected():
    suite = suite_of(SuiteCase(id="a", task="t"), runs_per_case=3)
    result = run_suite(suite, resolve_runner("tests.fake_runner:good_runner"))
    assert len(result.cases[0].runs) == 3


def test_case_runs_overrides_suite_default():
    suite = suite_of(SuiteCase(id="a", task="t", runs=2), runs_per_case=5)
    result = run_suite(suite, resolve_runner("tests.fake_runner:good_runner"))
    assert len(result.cases[0].runs) == 2
