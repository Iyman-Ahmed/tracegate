"""Eval the evaluator: labeled failure-injection corpus + detector metrics."""

from agentgates.evals.corpus import LabeledTrace, build_corpus, build_judge_corpus
from agentgates.evals.metrics import DetectorScore, evaluate_detector

__all__ = [
    "LabeledTrace",
    "build_corpus",
    "build_judge_corpus",
    "DetectorScore",
    "evaluate_detector",
]
