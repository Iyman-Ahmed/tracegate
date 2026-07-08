"""Local-first trace stores."""

from __future__ import annotations

import os
from pathlib import Path

from tracegate.store.jsonl import JSONLTraceStore

TRACES_FILENAME = "traces.jsonl"


def default_store() -> JSONLTraceStore:
    root = Path(os.environ.get("TRACEGATE_DIR", ".tracegate"))
    return JSONLTraceStore(root / TRACES_FILENAME)
