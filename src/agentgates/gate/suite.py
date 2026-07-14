"""TOML replay-suite definition."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class SuiteCase(BaseModel):
    id: str
    task: str
    constraints: list[str] = Field(default_factory=list)
    expect_contains: list[str] = Field(default_factory=list)
    runs: int | None = None


class Suite(BaseModel):
    name: str = "suite"
    runs_per_case: int = 1
    threshold: float | None = None
    cases: list[SuiteCase]


def load_suite(path: Path) -> Suite:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    meta = data.get("suite", {})
    return Suite(**meta, cases=data.get("case", []))
