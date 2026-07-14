"""Deterministic loop/stall detector: repeated identical steps."""

from __future__ import annotations

import json
from collections import Counter

from agentgates.detect.findings import Finding
from agentgates.schema import AgentTrace, LLMCallStep, ToolCallStep


def _fingerprint(step) -> str:
    if isinstance(step, ToolCallStep):
        args = json.dumps(step.arguments, sort_keys=True, default=str)
        return f"tool:{step.tool_name}:{args}"
    if isinstance(step, LLMCallStep):
        return f"llm:{step.response}"
    return f"{step.type}:{step.step_id}"


class LoopDetector:
    name = "loop"

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold

    def detect(self, trace: AgentTrace) -> list[Finding]:
        findings: list[Finding] = []
        prints = [_fingerprint(s) for s in trace.steps]
        reported: set[str] = set()

        i = 0
        while i < len(prints):
            j = i
            while j < len(prints) and prints[j] == prints[i]:
                j += 1
            if j - i >= self.threshold:
                reported.add(prints[i])
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="error",
                        step_index=trace.steps[i].index,
                        message=(
                            f"identical step repeated {j - i}x consecutively:"
                            f" {prints[i][:80]}"
                        ),
                    )
                )
            i = j

        counts = Counter(p for p in prints if p.startswith("tool:"))
        for fp, n in counts.items():
            if n >= self.threshold and fp not in reported:
                first = next(
                    s.index for s, p in zip(trace.steps, prints) if p == fp
                )
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="warning",
                        step_index=first,
                        message=f"tool call repeated {n}x with identical arguments: {fp[:80]}",
                    )
                )
        return findings
