"""The judge detectors must be measured too, or 'eval the evaluator' is a lie.

These tests validate the eval *harness* over judge-based detectors using
scripted judges — deterministic, no network. Real precision/recall numbers
require a live model (`agentgates eval --judge`); the harness that produces
them is what's exercised here.
"""

import json

from agentgates.detect import judge_detectors
from agentgates.detect.goal_drift import GoalDriftDetector
from agentgates.detect.ungrounded import UngroundedAssumptionDetector
from agentgates.evals import build_judge_corpus, evaluate_detector


class ScriptedJudge:
    """A perfect oracle for the judge corpus, keyed on prompt content.

    The goal-drift and ungrounded prompts are disjoint (one says "goal drift",
    the other "ungrounded assumptions"), and every corpus trace plants a
    distinctive token, so a content-keyed judge can answer both correctly.
    """

    def complete(self, prompt: str) -> str:
        lower = prompt.lower()
        if "goal drift" in lower:
            drifted = "$900" in prompt or "subscription plan" in prompt
            return json.dumps({"drifted": drifted, "step_index": 0, "dropped_constraints": []})
        claims: list[str] = []
        if "free wifi" in prompt:
            claims = ["UA420 includes free wifi"]
        elif "$1,240" in prompt:
            claims = ["The total is $1,240"]
        return json.dumps({"ungrounded_claims": claims})


class MissesPriceJudge(ScriptedJudge):
    """Imperfect: never flags the ungrounded price claim (a false negative)."""

    def complete(self, prompt: str) -> str:
        if "$1,240" in prompt and "ungrounded" in prompt.lower():
            return json.dumps({"ungrounded_claims": []})
        return super().complete(prompt)


def test_judge_corpus_composition():
    corpus = build_judge_corpus()
    assert len(corpus) == 6
    assert sum(1 for c in corpus if "goal_drift" in c.expected_detectors) == 2
    assert sum(1 for c in corpus if "ungrounded_assumption" in c.expected_detectors) == 2
    assert sum(1 for c in corpus if not c.expected_detectors) == 2


def test_perfect_judge_scores_both_detectors_perfectly():
    corpus = build_judge_corpus()
    judge = ScriptedJudge()
    for detector in judge_detectors(judge):
        score = evaluate_detector(detector, corpus)
        assert score.precision == 1.0, f"{detector.name} fp={score.fp}"
        assert score.recall == 1.0, f"{detector.name} fn={score.fn}"


def test_metric_reflects_a_false_negative():
    corpus = build_judge_corpus()
    score = evaluate_detector(UngroundedAssumptionDetector(MissesPriceJudge()), corpus)
    # One ungrounded positive caught, one missed: perfect precision, half recall.
    assert score.precision == 1.0
    assert score.recall == 0.5


def test_grounded_traces_are_true_negatives():
    corpus = build_judge_corpus()
    grounded = [c for c in corpus if not c.expected_detectors]
    judge = ScriptedJudge()
    detector = GoalDriftDetector(judge)
    for labeled in grounded:
        assert detector.detect(labeled.trace) == []
