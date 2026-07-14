import json

from agentgates.gate.baseline import load_baseline, save_baseline
from agentgates.gate.score import CaseResult, RunResult, SuiteResult


def result_fixture() -> SuiteResult:
    return SuiteResult(
        suite_name="s",
        cases=[CaseResult(case_id="a", runs=[RunResult(success=True)])],
    )


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    save_baseline(result_fixture(), path)
    data = load_baseline(path)
    assert data["reliability_score"] == 1.0
    assert data["cases"] == {"a": 1.0}
    assert json.loads(path.read_text())  # valid json on disk
