"""Persisted reliability baselines for regression comparison."""

from __future__ import annotations

import json
from pathlib import Path

from tracegate.gate.score import SuiteResult


def save_baseline(result: SuiteResult, path: Path) -> None:
    payload = {
        "reliability_score": result.reliability_score,
        "cases": {c.case_id: c.success_rate for c in result.cases},
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_baseline(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
