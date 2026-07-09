"""Detect layer: analyzers that flag silent failures in an AgentTrace."""

from __future__ import annotations

from tracegate.detect.findings import Finding, Severity
from tracegate.detect.goal_drift import GoalDriftDetector
from tracegate.detect.judge import Judge
from tracegate.detect.loop import LoopDetector
from tracegate.detect.tool_misuse import ToolMisuseDetector
from tracegate.detect.ungrounded import UngroundedAssumptionDetector
from tracegate.schema import AgentTrace

__all__ = [
    "Finding",
    "GoalDriftDetector",
    "Judge",
    "LoopDetector",
    "Severity",
    "ToolMisuseDetector",
    "UngroundedAssumptionDetector",
    "default_detectors",
    "judge_detectors",
    "run_detectors",
]


def default_detectors() -> list:
    return [LoopDetector(), ToolMisuseDetector()]


def judge_detectors(judge: Judge) -> list:
    return [GoalDriftDetector(judge), UngroundedAssumptionDetector(judge)]


def run_detectors(trace: AgentTrace, detectors: list) -> list[Finding]:
    findings: list[Finding] = []
    for detector in detectors:
        findings.extend(detector.detect(trace))
    return sorted(
        findings,
        key=lambda f: f.step_index if f.step_index is not None else -1,
    )
