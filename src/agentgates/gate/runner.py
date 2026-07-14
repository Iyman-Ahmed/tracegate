"""Resolve user agent runners and replay suites against them."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable

from agentgates.detect import default_detectors, run_detectors
from agentgates.gate.score import CaseResult, RunResult, SuiteResult
from agentgates.gate.suite import Suite
from agentgates.schema import AgentTrace, LLMCallStep

Runner = Callable[[str], AgentTrace]


def resolve_runner(spec: str) -> Runner:
    target, sep, attr = spec.rpartition(":")
    if not sep or not target:
        raise ValueError(
            f"runner must be 'module:function' or 'file.py:function', got {spec!r}"
        )
    if target.endswith(".py"):
        path = Path(target)
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        module = importlib.import_module(target)
    return getattr(module, attr)


def _outcome_text(trace: AgentTrace) -> str:
    result = trace.metadata.get("result")
    if result:
        return str(result)
    for step in reversed(trace.steps):
        if isinstance(step, LLMCallStep):
            return step.response
    return ""


def run_suite(suite: Suite, runner: Runner, detectors: list | None = None) -> SuiteResult:
    if detectors is None:
        detectors = default_detectors()
    case_results: list[CaseResult] = []
    for case in suite.cases:
        runs: list[RunResult] = []
        for _ in range(case.runs or suite.runs_per_case):
            try:
                trace = runner(case.task)
            except Exception as exc:  # runner bugs must not kill the gate
                runs.append(
                    RunResult(success=False, error=f"{type(exc).__name__}: {exc}")
                )
                continue
            findings = run_detectors(trace, detectors)
            outcome = _outcome_text(trace)
            success = all(expected in outcome for expected in case.expect_contains)
            runs.append(RunResult(success=success, findings=findings))
        case_results.append(CaseResult(case_id=case.id, runs=runs))
    return SuiteResult(suite_name=suite.name, cases=case_results)
