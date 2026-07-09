"""Eval the evaluator: labeled failure-injection corpus + detector metrics."""

from tracegate.evals.corpus import LabeledTrace, build_corpus
from tracegate.evals.metrics import DetectorScore, evaluate_detector

__all__ = ["LabeledTrace", "build_corpus", "DetectorScore", "evaluate_detector"]
