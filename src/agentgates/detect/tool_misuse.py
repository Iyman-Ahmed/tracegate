"""Deterministic tool-misuse detector: errors and blind retries."""

from __future__ import annotations

import json

from agentgates.detect.findings import Finding
from agentgates.schema import AgentTrace, ToolCallStep


class ToolMisuseDetector:
    name = "tool_misuse"

    def detect(self, trace: AgentTrace) -> list[Finding]:
        findings: list[Finding] = []
        prior_error_fps: set[str] = set()
        for step in trace.steps:
            if not isinstance(step, ToolCallStep):
                continue
            args = json.dumps(step.arguments, sort_keys=True, default=str)
            fp = f"{step.tool_name}:{args}"
            if step.error is None:
                continue
            if fp in prior_error_fps:
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="error",
                        step_index=step.index,
                        message=(
                            f"tool '{step.tool_name}' retried with identical"
                            f" arguments after an error and failed again"
                        ),
                    )
                )
            else:
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="warning",
                        step_index=step.index,
                        message=f"tool '{step.tool_name}' returned an error: {step.error}",
                    )
                )
                prior_error_fps.add(fp)
        return findings
