"""Typed findings emitted by detectors."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Severity = Literal["info", "warning", "error"]


class Finding(BaseModel):
    detector: str
    severity: Severity
    message: str
    step_index: int | None = None
    confidence: float = 1.0
