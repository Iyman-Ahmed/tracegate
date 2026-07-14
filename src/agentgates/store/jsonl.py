"""Append-only JSONL trace store — the zero-config default."""

from __future__ import annotations

from pathlib import Path

from agentgates.schema import AgentTrace


class JSONLTraceStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def save(self, trace: AgentTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")

    def load(self, trace_id: str) -> AgentTrace:
        for trace in self.list_traces():
            if trace.trace_id == trace_id:
                return trace
        raise KeyError(f"trace not found: {trace_id}")

    def list_traces(self) -> list[AgentTrace]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [
                AgentTrace.model_validate_json(line)
                for line in f
                if line.strip()
            ]
