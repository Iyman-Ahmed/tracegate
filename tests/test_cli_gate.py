from typer.testing import CliRunner

from tracegate.cli import app

runner = CliRunner()

SUITE = """
[suite]
name = "demo"

[[case]]
id = "ok"
task = "do the thing"
expect_contains = ["SUCCESS_TOKEN"]
"""

FAILING_SUITE = SUITE.replace("SUCCESS_TOKEN", "NEVER_THERE")

RUNNER = "tests.fake_runner:good_runner"


def write_suite(tmp_path, content=SUITE):
    path = tmp_path / "suite.toml"
    path.write_text(content)
    return path


def test_run_prints_report(tmp_path):
    result = runner.invoke(app, ["run", "--suite", str(write_suite(tmp_path)), "--runner", RUNNER])
    assert result.exit_code == 0
    assert "reliability score: 1.00" in result.output


def test_run_writes_html_report(tmp_path):
    out = tmp_path / "report.html"
    result = runner.invoke(
        app,
        ["run", "--suite", str(write_suite(tmp_path)), "--runner", RUNNER, "--report", str(out)],
    )
    assert result.exit_code == 0
    assert out.exists()
    assert "<!doctype html>" in out.read_text()


def test_ci_passes_above_threshold(tmp_path):
    result = runner.invoke(
        app,
        ["ci", "--suite", str(write_suite(tmp_path)), "--runner", RUNNER, "--threshold", "0.9"],
    )
    assert result.exit_code == 0
    assert "gate passed" in result.output.lower()


def test_ci_fails_below_threshold(tmp_path):
    result = runner.invoke(
        app,
        ["ci", "--suite", str(write_suite(tmp_path, FAILING_SUITE)), "--runner", RUNNER, "--threshold", "0.9"],
    )
    assert result.exit_code == 1
    assert "below threshold" in result.output.lower()


def test_ci_uses_suite_threshold(tmp_path):
    path = tmp_path / "suite.toml"
    path.write_text(
        '[suite]\nthreshold = 0.9\n\n'
        '[[case]]\nid = "ok"\ntask = "t"\nexpect_contains = ["NEVER"]\n'
    )
    result = runner.invoke(app, ["ci", "--suite", str(path), "--runner", RUNNER])
    assert result.exit_code == 1


def test_ci_baseline_update_then_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    ok = write_suite(tmp_path)
    result = runner.invoke(
        app,
        ["ci", "--suite", str(ok), "--runner", RUNNER, "--baseline", str(baseline), "--update-baseline"],
    )
    assert result.exit_code == 0
    assert baseline.exists()

    bad = tmp_path / "bad.toml"
    bad.write_text(FAILING_SUITE)
    result = runner.invoke(
        app,
        ["ci", "--suite", str(bad), "--runner", RUNNER, "--baseline", str(baseline)],
    )
    assert result.exit_code == 1
    assert "baseline" in result.output.lower()
