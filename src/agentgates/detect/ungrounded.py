"""Judge-based silent-hallucination detector: claims with no supporting evidence."""

from __future__ import annotations

from agentgates.detect._summary import summarize_step
from agentgates.detect.findings import Finding
from agentgates.detect.judge import Judge, extract_json
from agentgates.schema import AgentTrace, LLMCallStep

_PROMPT = """You are auditing one message from an AI agent for ungrounded assumptions \
(silent hallucinations): factual claims that appear in none of the evidence available \
to the agent.

Evidence available to the agent so far:
{evidence}

Agent message to audit:
{message}

List factual claims in the message that are NOT supported by the evidence. Ignore \
plans, intentions, and hedged statements. Respond with ONLY a JSON object:
{{"ungrounded_claims": ["exact claim", ...], "reasoning": "one sentence"}}"""


class UngroundedAssumptionDetector:
    name = "ungrounded_assumption"

    def __init__(self, judge: Judge) -> None:
        self._judge = judge

    def detect(self, trace: AgentTrace) -> list[Finding]:
        findings: list[Finding] = []
        evidence: list[str] = [f"task: {trace.task.description}"]
        for step in trace.steps:
            if isinstance(step, LLMCallStep):
                prompt = _PROMPT.format(
                    evidence="\n".join(evidence), message=step.response
                )
                data = extract_json(self._judge.complete(prompt))
                for claim in data.get("ungrounded_claims") or []:
                    findings.append(
                        Finding(
                            detector=self.name,
                            severity="error",
                            step_index=step.index,
                            message=f"ungrounded assumption: {claim}",
                            confidence=0.8,
                        )
                    )
            evidence.append(summarize_step(step))
        return findings
