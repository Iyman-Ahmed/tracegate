from tracegate.detect import judge_detectors
from tracegate.detect.ungrounded import UngroundedAssumptionDetector
from tracegate.schema import AgentInfo, AgentTrace, LLMCallStep, TaskSpec, ToolCallStep


class FakeJudge:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def make_trace() -> AgentTrace:
    trace = AgentTrace(task=TaskSpec(description="find flights"), agent=AgentInfo(framework="test"))
    trace.steps.append(ToolCallStep(index=0, tool_name="search", arguments={}, result={"flights": ["UA 420"]}))
    trace.steps.append(LLMCallStep(index=1, model="m", prompt="", response="UA 420 has free wifi"))
    return trace


def test_grounded_no_findings():
    judge = FakeJudge(['{"ungrounded_claims": [], "reasoning": "all grounded"}'])
    assert UngroundedAssumptionDetector(judge).detect(make_trace()) == []


def test_ungrounded_claim_flagged_at_step():
    judge = FakeJudge(['{"ungrounded_claims": ["UA 420 has free wifi"], "reasoning": "wifi never mentioned"}'])
    findings = UngroundedAssumptionDetector(judge).detect(make_trace())
    assert len(findings) == 1
    f = findings[0]
    assert f.detector == "ungrounded_assumption"
    assert f.severity == "error"
    assert f.step_index == 1
    assert "free wifi" in f.message


def test_one_judge_call_per_llm_step_with_prior_evidence():
    trace = make_trace()
    trace.steps.append(LLMCallStep(index=2, model="m", prompt="", response="booking it"))
    judge = FakeJudge(['{"ungrounded_claims": []}', '{"ungrounded_claims": []}'])
    UngroundedAssumptionDetector(judge).detect(trace)
    assert len(judge.prompts) == 2
    assert "UA 420" in judge.prompts[0]        # tool evidence present
    assert "find flights" in judge.prompts[0]  # task present
    assert "free wifi" in judge.prompts[1]     # prior llm message becomes evidence


def test_tool_only_trace_makes_no_judge_calls():
    trace = AgentTrace(task=TaskSpec(description="t"), agent=AgentInfo(framework="test"))
    trace.steps.append(ToolCallStep(index=0, tool_name="x", arguments={}, result="ok"))
    judge = FakeJudge([])
    assert UngroundedAssumptionDetector(judge).detect(trace) == []


def test_judge_detectors_factory():
    judge = FakeJudge([])
    names = [d.name for d in judge_detectors(judge)]
    assert names == ["goal_drift", "ungrounded_assumption"]
