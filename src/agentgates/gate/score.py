"""Outcome scoring: pass@k-style success rates + detector penalties."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentgates.detect import Finding


class RunResult(BaseModel):
    success: bool
    findings: list[Finding] = Field(default_factory=list)
    error: str | None = None


class CaseResult(BaseModel):
    case_id: str
    runs: list[RunResult]

    @property
    def success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.success) / len(self.runs)

    @property
    def error_finding_count(self) -> int:
        return sum(
            1 for r in self.runs for f in r.findings if f.severity == "error"
        )


class SuiteResult(BaseModel):
    suite_name: str
    cases: list[CaseResult]

    @property
    def total_runs(self) -> int:
        return sum(len(c.runs) for c in self.cases)

    @property
    def reliability_score(self) -> float:
        if not self.cases:
            return 0.0
        success_mean = sum(c.success_rate for c in self.cases) / len(self.cases)
        total_runs = self.total_runs
        errors_per_run = (
            sum(c.error_finding_count for c in self.cases) / total_runs
            if total_runs
            else 0.0
        )
        penalty = min(0.5, 0.1 * errors_per_run)
        return success_mean * (1 - penalty)
