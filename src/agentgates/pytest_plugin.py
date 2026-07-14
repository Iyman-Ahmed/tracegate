"""pytest plugin: a per-test TraceRecorder fixture.

Registered automatically via the `pytest11` entry point when agentgates is
installed. Traces save to the default store (AGENTGATES_DIR, else .agentgates/)
when the test finishes, so `agentgates list` / `show` / `detect` work on them.
"""

from __future__ import annotations

import pytest

from agentgates.recorder import TraceRecorder


@pytest.fixture
def trace_recorder(request):
    """A TraceRecorder named after the current test, saved on teardown."""
    recorder = TraceRecorder(task=request.node.name, framework="pytest")
    yield recorder
    recorder.finish()
