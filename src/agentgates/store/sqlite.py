"""SQLite trace store — same interface as JSONL, queryable at scale."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agentgates.schema import AgentTrace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id   TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    payload    TEXT NOT NULL
)
"""


class SQLiteTraceStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def save(self, trace: AgentTrace) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces (trace_id, started_at, payload)"
                " VALUES (?, ?, ?)",
                (
                    trace.trace_id,
                    trace.started_at.isoformat(),
                    trace.model_dump_json(),
                ),
            )

    def load(self, trace_id: str) -> AgentTrace:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"trace not found: {trace_id}")
        return AgentTrace.model_validate_json(row[0])

    def list_traces(self) -> list[AgentTrace]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM traces ORDER BY started_at"
            ).fetchall()
        return [AgentTrace.model_validate_json(r[0]) for r in rows]
