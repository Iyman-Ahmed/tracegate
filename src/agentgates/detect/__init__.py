"""Detect layer: analyzers that flag silent failures in an AgentTrace."""

from __future__ import annotations

from agentgates.detect.contradiction import ContradictionDetector
from agentgates.detect.findings import Finding, Severity
from agentgates.detect.goal_drift import GoalDriftDetector
from agentgates.detect.judge import Judge
from agentgates.detect.loop import LoopDetector
from agentgates.detect.tool_misuse import ToolMisuseDetector
from agentgates.detect.ungrounded import UngroundedAssumptionDetector
from agentgates.schema import AgentTrace

__all__ = [
    "ContradictionDetector",
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
    return [LoopDetector(), ToolMisuseDetector(), ContradictionDetector()]


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
