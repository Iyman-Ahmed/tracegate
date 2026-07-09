"""Trace-level precision/recall for a detector over a labeled corpus."""

from __future__ import annotations

from pydantic import BaseModel

from tracegate.evals.corpus import LabeledTrace


class DetectorScore(BaseModel):
    detector: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0


def evaluate_detector(detector, corpus: list[LabeledTrace]) -> DetectorScore:
    tp = fp = fn = 0
    for labeled in corpus:
        fired = bool(detector.detect(labeled.trace))
        expected = detector.name in labeled.expected_detectors
        if fired and expected:
            tp += 1
        elif fired and not expected:
            fp += 1
        elif not fired and expected:
            fn += 1
    return DetectorScore(detector=detector.name, tp=tp, fp=fp, fn=fn)
