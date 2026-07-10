"""Deterministic contradiction detector: conflicting evidence, stale beliefs.

The same tool called with the same arguments returning two different results
means the agent is holding conflicting evidence. That alone is a warning. If a
later step still acts on the value that was superseded, that is an error --
the agent's answer is built on information it had already been told was wrong.

Exact-value conflicts only. Semantic contradictions ("the flight is cheap" vs.
a $900 fare) are judge territory; this detector costs nothing to run.
"""

from __future__ import annotations

import json

from tracegate.detect.findings import Finding
from tracegate.schema import AgentTrace, LLMCallStep, ToolCallStep


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


class ContradictionDetector:
    name = "contradiction"

    def detect(self, trace: AgentTrace) -> list[Finding]:
        findings: list[Finding] = []
        seen: dict[str, tuple[int, object]] = {}
        # Scalar values that a later step must no longer rely on.
        superseded: list[str] = []

        for step in trace.steps:
            if isinstance(step, ToolCallStep):
                if step.error is not None:
                    continue
                fingerprint = (
                    f"{step.tool_name}:{_canonical(step.arguments)}"
                )
                previous = seen.get(fingerprint)
                if previous is None:
                    seen[fingerprint] = (step.index, step.result)
                    continue
                prev_index, prev_result = previous
                if _canonical(prev_result) == _canonical(step.result):
                    continue
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="warning",
                        step_index=step.index,
                        message=(
                            f"tool '{step.tool_name}' returned {step.result!r} at step"
                            f" {step.index} but {prev_result!r} at step {prev_index}"
                            f" for identical arguments"
                        ),
                    )
                )
                seen[fingerprint] = (step.index, step.result)
                if isinstance(prev_result, (str, int, float)) and not isinstance(
                    prev_result, bool
                ):
                    superseded.append(str(prev_result))

            elif isinstance(step, LLMCallStep) and superseded:
                for stale in superseded:
                    if stale in step.response:
                        findings.append(
                            Finding(
                                detector=self.name,
                                severity="error",
                                step_index=step.index,
                                message=(
                                    f"step relies on superseded value {stale!r},"
                                    f" which a later tool result contradicted"
                                ),
                            )
                        )
        return findings
