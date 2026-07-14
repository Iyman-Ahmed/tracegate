from agentgates.detect.goal_drift import GoalDriftDetector
from agentgates.schema import AgentInfo, AgentTrace, LLMCallStep, TaskSpec


class FakeJudge:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def make_trace() -> AgentTrace:
    trace = AgentTrace(
        task=TaskSpec(description="book flight", constraints=["under $500"]),
        agent=AgentInfo(framework="test"),
    )
    trace.steps.append(LLMCallStep(index=0, model="m", prompt="", response="booking $900 flight"))
    return trace


def test_no_drift_no_findings():
    judge = FakeJudge('{"drifted": false, "step_index": null, "dropped_constraints": [], "reasoning": "on task"}')
    assert GoalDriftDetector(judge).detect(make_trace()) == []


def test_drift_emits_error_with_step():
    judge = FakeJudge('{"drifted": true, "step_index": 0, "dropped_constraints": ["under $500"], "reasoning": "budget ignored"}')
    findings = GoalDriftDetector(judge).detect(make_trace())
    assert len(findings) == 1
    f = findings[0]
    assert f.detector == "goal_drift"
    assert f.severity == "error"
    assert f.step_index == 0
    assert "under $500" in f.message
    assert f.confidence == 0.8


def test_prompt_contains_task_and_steps():
    judge = FakeJudge('{"drifted": false}')
    GoalDriftDetector(judge).detect(make_trace())
    prompt = judge.prompts[0]
    assert "book flight" in prompt
    assert "under $500" in prompt
    assert "booking $900 flight" in prompt


def test_unparseable_judge_output_means_no_findings():
    judge = FakeJudge("I cannot answer")
    assert GoalDriftDetector(judge).detect(make_trace()) == []


def test_empty_trace_skips_judge():
    trace = AgentTrace(task=TaskSpec(description="t"), agent=AgentInfo(framework="test"))
    judge = FakeJudge('{"drifted": true}')
    assert GoalDriftDetector(judge).detect(trace) == []
    assert judge.prompts == []
