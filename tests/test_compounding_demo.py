import importlib.util
import sys
from pathlib import Path

DEMO_PATH = Path(__file__).parent.parent / "examples" / "compounding_demo.py"


def load_demo():
    spec = importlib.util.spec_from_file_location("compounding_demo", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compounding_demo"] = module
    spec.loader.exec_module(module)
    return module


def test_simulation_is_deterministic_and_detectable():
    demo = load_demo()
    outcomes = demo.simulate(n_runs=10, seed=7)
    assert len(outcomes) == 10

    failed = [o for o in outcomes if o.injected_step is not None]
    clean = [o for o in outcomes if o.injected_step is None]
    assert len(failed) >= 2  # compounding math: ~4/10 runs should fail

    for outcome in failed:
        # every run still *looks* successful...
        assert "Task complete" in outcome.trace.steps[-1].response
        # ...but AgentGates pinpoints the originating step
        assert any(f.step_index == outcome.injected_step for f in outcome.findings)

    for outcome in clean:
        assert outcome.findings == []


def test_same_seed_same_outcome():
    demo = load_demo()
    a = demo.simulate(n_runs=10, seed=7)
    b = demo.simulate(n_runs=10, seed=7)
    assert [o.injected_step for o in a] == [o.injected_step for o in b]
