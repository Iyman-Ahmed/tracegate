"""The compounding-error demo — TraceGate's flagship pitch, executable.

An agent that is 95% reliable per step is only ~59% reliable across a
10-step workflow. This script simulates exactly that: 10 runs of a 10-step
agent where each step silently fails 5% of the time. A failed step doesn't
throw — the agent blindly retries, gives up quietly, and still ends the run
with "Task complete." The output *looks* fine every time.

TraceGate's detectors read the trace and pinpoint the originating step.

Run:  python examples/compounding_demo.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from tracegate import TraceRecorder
from tracegate.detect import Finding, default_detectors, run_detectors
from tracegate.schema import AgentTrace

STEPS = 10
PER_STEP_RELIABILITY = 0.95


class _NullStore:
    def save(self, trace):
        pass


@dataclass
class RunOutcome:
    trace: AgentTrace
    injected_step: int | None
    findings: list[Finding] = field(default_factory=list)


def _run_agent(rng: random.Random, task: str) -> tuple[AgentTrace, int | None]:
    rec = TraceRecorder(task=task, framework="demo", model="simulated", store=_NullStore())
    injected: int | None = None
    for i in range(STEPS):
        step_ok = rng.random() < PER_STEP_RELIABILITY
        if step_ok or injected is not None:
            rec.record_tool_call(
                tool_name=f"step_{i}", arguments={"n": i}, result="ok"
            )
        else:
            # The silent failure: blind identical retries, then move on.
            injected = len(rec.trace.steps)
            for _ in range(3):
                rec.record_tool_call(
                    tool_name=f"step_{i}",
                    arguments={"n": i},
                    error="unexpected state",
                )
    rec.record_llm_call(
        model="simulated", prompt="", response="Task complete. Everything went smoothly."
    )
    return rec.finish(), injected


def simulate(n_runs: int = 10, seed: int = 7) -> list[RunOutcome]:
    rng = random.Random(seed)
    detectors = default_detectors()
    outcomes: list[RunOutcome] = []
    for n in range(n_runs):
        trace, injected = _run_agent(rng, f"complete workflow run {n + 1}")
        findings = run_detectors(trace, detectors)
        outcomes.append(RunOutcome(trace=trace, injected_step=injected, findings=findings))
    return outcomes


def main() -> None:
    outcomes = simulate()
    failed = sum(1 for o in outcomes if o.injected_step is not None)
    print(f"{STEPS}-step agent, {PER_STEP_RELIABILITY:.0%} reliable per step, {len(outcomes)} runs")
    print(f"expected failure rate: {1 - PER_STEP_RELIABILITY ** STEPS:.0%}\n")
    print(f"{'run':<5} {'agent says':<18} {'reality':<22} traceGate verdict")
    for n, o in enumerate(outcomes, 1):
        agent_says = '"Task complete."'
        if o.injected_step is None:
            print(f"{n:<5} {agent_says:<18} {'actually succeeded':<22} no findings")
        else:
            pinned = next(f for f in o.findings if f.step_index == o.injected_step)
            reality = f"silently died at step {o.injected_step}"
            print(f"{n:<5} {agent_says:<18} {reality:<22} [{pinned.severity}] step {pinned.step_index} {pinned.detector}: {pinned.message[:48]}")
    print(f"\n{failed}/{len(outcomes)} runs failed silently — every one pinned to its originating step.")


if __name__ == "__main__":
    main()
