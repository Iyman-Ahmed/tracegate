# TraceGate Phase 4 (Launch Assets) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** In-repo launch assets: the compounding-error demo (flagship README asset), the silent-failure taxonomy document, and a README overhaul telling the three-layer story. No public posting — that's the owner's call.

**Architecture:** `examples/compounding_demo.py` exposes `simulate(n_runs=10, seed=7) -> list[RunOutcome]` — a seeded 10-step simulated agent at 95% per-step reliability. A failing step injects a *silent* failure (three identical failing retries, then the run continues and ends with "Task complete." — output looks fine). Deterministic detectors pinpoint the originating step. A pytest locks the behavior; `python examples/compounding_demo.py` prints the table for the README.

## Global Constraints

- Demo must be deterministic (seeded `random.Random`), no network, no deps.
- Test asserts: 10 runs; ≥2 failed runs; every failed run's findings include one at exactly the injected step; clean runs have zero findings.
- Taxonomy doc names all 5 modes (goal drift, ungrounded assumption, contradiction, loop/stall, tool misuse); contradiction is marked roadmap (no detector yet).
- README: pitch, compounding math, 3-layer quickstart (record → detect → gate), detector table with corpus eval numbers, GitHub Action snippet, taxonomy link.

### Task 1: Compounding-error demo (TDD)

**Files:** Create `examples/compounding_demo.py`; Test `tests/test_compounding_demo.py`.

**Interfaces:** `RunOutcome` dataclass: `trace: AgentTrace`, `injected_step: int | None`, `findings: list[Finding]`; `simulate(n_runs: int = 10, seed: int = 7) -> list[RunOutcome]`; `main()` prints the run table + summary.

- [ ] Step 1: failing test (see file content in repo)
- [ ] Step 2: verify fail — ModuleNotFoundError (examples not a package: test imports via `importlib.util.spec_from_file_location`)
- [ ] Step 3: implement demo
- [ ] Step 4: verify pass + run `venv/bin/python examples/compounding_demo.py`
- [ ] Step 5: commit `feat: compounding-error demo`

### Task 2: Taxonomy doc + README overhaul

**Files:** Create `docs/silent-failure-taxonomy.md`; rewrite `README.md`.

- [ ] Step 1: write taxonomy doc (5 modes, examples, detector mapping, contradiction = roadmap)
- [ ] Step 2: rewrite README with real `tracegate eval` output and demo numbers from Task 1
- [ ] Step 3: full suite green; commit `docs: silent-failure taxonomy and README overhaul`
