"""pytest plugin: a per-test TraceRecorder fixture.

Registered automatically via the `pytest11` entry point when tracegate is
installed. Traces save to the default store (TRACEGATE_DIR, else .tracegate/)
when the test finishes, so `tracegate list` / `show` / `detect` work on them.
"""

from __future__ import annotations

import pytest

from tracegate.recorder import TraceRecorder


@pytest.fixture
def trace_recorder(request):
    """A TraceRecorder named after the current test, saved on teardown."""
    recorder = TraceRecorder(task=request.node.name, framework="pytest")
    yield recorder
    recorder.finish()
