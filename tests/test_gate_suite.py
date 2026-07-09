import pytest
from pydantic import ValidationError

from tracegate.gate.suite import Suite, load_suite

SUITE_TOML = """
[suite]
name = "booking agent"
runs_per_case = 2
threshold = 0.8

[[case]]
id = "cheap-flight"
task = "Find the cheapest flight under $500"
constraints = ["under $500"]
expect_contains = ["$420"]

[[case]]
id = "refund"
task = "Process a refund"
runs = 3
"""


def test_load_suite(tmp_path):
    path = tmp_path / "suite.toml"
    path.write_text(SUITE_TOML)
    suite = load_suite(path)
    assert suite.name == "booking agent"
    assert suite.runs_per_case == 2
    assert suite.threshold == 0.8
    assert len(suite.cases) == 2
    assert suite.cases[0].expect_contains == ["$420"]
    assert suite.cases[0].runs is None
    assert suite.cases[1].runs == 3
    assert suite.cases[1].expect_contains == []


def test_defaults(tmp_path):
    path = tmp_path / "suite.toml"
    path.write_text('[[case]]\nid = "a"\ntask = "do a"\n')
    suite = load_suite(path)
    assert suite.name == "suite"
    assert suite.runs_per_case == 1
    assert suite.threshold is None


def test_case_requires_id_and_task():
    with pytest.raises(ValidationError):
        Suite(cases=[{"id": "x"}])
