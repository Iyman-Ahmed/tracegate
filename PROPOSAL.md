# TraceGate — The Black Box Flight Recorder for AI Agents

> **Purpose of this document:** Full project context for future working sessions. If you are an AI assistant reading this in a new chat, this document is the single source of truth for what TraceGate is, why it exists, who it serves, how it works, and what to build next. Written 2026-07-08.

---

## 1. Executive Summary

TraceGate is an open-source reliability toolkit for AI agents built on three layers: **Record** (drop-in instrumentation that captures every step of an agent run), **Detect** (analyzers that flag silent failures in those traces), and **Gate** (replayable regression suites with a reliability score that fails CI when quality drops).

**One-line pitch:** "The black box flight recorder for AI agents — record every step, detect silent failures, gate your deploys."

**Why now (the whole thesis):**
- Agent reliability compounds badly: an agent that is 95% reliable per step is only ~59% reliable across a 10-step workflow. This math is the project's tagline-level hook.
- Gartner predicts **40%+ of agentic AI projects will be canceled by end of 2027**, primarily due to unreliable output and weak controls.
- Researchers have recently named a failure mode with essentially no tooling: **"silent hallucinations"** — false intermediate beliefs/assumptions that steer an agent's actions without ever being surfaced to the user (see OpenReview: "Tiny Silent Hallucinations in Agentic AI").
- RAG evals were 2024–25's solved problem (Ragas, promptfoo, etc.). Agent evals are the 2026 equivalent, and the space has no standard open-source tool yet — observability vendors (LangSmith, Langfuse, Maxim, Noveum) are hosted/commercial-first, leaving the open, CI-native niche open.

**Dual goal:** genuine open-source adoption AND flagship portfolio piece for the owner's GenAI/MLOps job search. The interview narrative: *"I built eval-driven CI for RAG (RAGOps); agents have the same problem but compounding — so I built the same discipline for agents."*

---

## 2. The Problem

Teams shipping multi-step agents (support automation, research agents, coding agents, workflow bots) face three gaps:

1. **No visibility:** when an agent run goes wrong, there is no structured record of *which step* introduced the error. Logs capture the LLM calls but not the semantic chain — what the agent believed, why it chose a tool, whether a tool result contradicted an earlier one.
2. **Silent failures:** the scariest bugs never throw. The agent quietly drops a constraint from the original request (goal drift), invents an assumption and acts on it (silent hallucination), keeps working from a tool result that a later result contradicted, or loops with slight variations. Output *looks* fine; it's wrong.
3. **No regression safety:** teams tweak a prompt or bump a model version and have no way to know whether agent reliability went up or down. There is no "pytest for agents" — no suite you can run in CI that says "this change dropped task success from 92% to 71%, blocked."

Existing tools each cover a fragment: observability platforms (LangSmith, Langfuse) record traces but detection is manual/hosted; eval libraries (Ragas, DeepEval) score single LLM calls, not multi-step workflows; agent frameworks ship minimal built-in eval. Nobody owns **record → detect → gate** as one open-source pipeline.

---

## 3. The Product

### 3.1 Layer 1 — Record (instrumentation SDK)

- Python library, integration in ≤5 lines. Wraps or hooks into: **Claude Agent SDK**, **LangGraph**, **OpenAI Agents SDK** (v1 targets; adapter interface for others).
- Captures a structured **AgentTrace**: every LLM call (prompt, response, model, tokens), every tool call (name, args, result, latency), retrievals, state transitions, and the original task specification.
- Emits **OpenTelemetry-compatible spans** (follows the emerging GenAI semantic conventions) so traces also flow into existing observability stacks — adoption-friendly, no rip-and-replace.
- Local-first: traces serialize to JSONL/SQLite on disk. No server required, no data leaves the machine. This is a deliberate wedge against hosted-first competitors.

### 3.2 Layer 2 — Detect (trace analyzers)

Analyzers run over an AgentTrace (post-hoc or streaming) and emit typed findings with severity + the exact step where the failure originated:

| Detector | What it catches |
|----------|----------------|
| **Goal drift** | Agent's working objective diverges from the original task (LLM-judge compares task spec vs. each step's intent; flags dropped constraints) |
| **Ungrounded assumption** | Agent states/acts on a "fact" that appears in no tool result, retrieval, or user input — the silent-hallucination detector, the project's signature feature |
| **Contradiction** | A step relies on information that a later (or earlier) tool result contradicts |
| **Loop/stall** | Semantic near-duplicate step sequences, budget burn without state progress |
| **Tool misuse** | Malformed/hallucinated arguments, ignored tool errors, fabricated tool output references |

- Detection uses a hybrid: cheap deterministic checks first (schema validation, exact contradiction, loop hashing), LLM-as-judge only where semantics are required (drift, groundedness). Keeps cost predictable.
- **The detectors themselves ship with an eval set** — a labeled corpus of traces with known injected failures, so detector precision/recall is measured and published. (Credibility: a reliability tool must prove its own reliability. This is the RAGOps methodology applied to the tool itself.)

### 3.3 Layer 3 — Gate (regression harness + CI)

- Recorded traces become **replayable test cases**: `tracegate record` during development → curate into a suite → `tracegate run` re-executes the agent against the same tasks and scores outcomes.
- Scoring: task success (LLM-judge vs. expected outcome + optional programmatic assertions), per-step reliability, detector findings count, cost/latency budgets.
- **`tracegate ci`** — one command that runs the suite and exits non-zero if the reliability score drops below a configured threshold or vs. a baseline. GitHub Action published to the marketplace. This is the direct lift of the RAGOps CI-quality-gate pattern.
- Output: terminal report + static HTML report (per-run drill-down: timeline of steps, findings pinned to steps, diff vs. baseline run).

### 3.4 What TraceGate is NOT (v1)
- Not a hosted dashboard/SaaS — local + CI first. (Hosted team dashboard is the *future* monetization layer, explicitly deferred.)
- Not an agent framework — it instruments others, never competes with them.
- Not a prompt-injection/security scanner — that's a separate project (MCP Shield idea); TraceGate covers *accidents*, not *attacks*.
- Not a generic LLM eval library — multi-step agent workflows only; single-call evals are Ragas/DeepEval territory.

---

## 4. How It Works (Technical Architecture)

```
Agent app (Claude Agent SDK / LangGraph / OpenAI Agents)
   │  (adapter hooks / OTel spans)
   ▼
tracegate-sdk  ──►  AgentTrace store (JSONL / SQLite, local)
                        │
        ┌───────────────┼──────────────────┐
        ▼               ▼                  ▼
   Detectors       Replay engine      Report generator
   (rules + LLM    (re-run suite,     (terminal + static
    judge)          score outcomes)    HTML)
        └───────────────┴──────────────────┘
                        ▼
                 `tracegate ci` → exit code + threshold gate
                 (GitHub Action)
```

**Key engineering principles:**
1. **Adapter pattern for frameworks** — core is framework-agnostic over a normalized AgentTrace schema; each framework gets a thin adapter. The trace schema is the real product; document it as a spec.
2. **Deterministic before LLM** — every detector tries rules first, LLM-judge second. Judge calls are cached by trace-content hash so CI reruns are cheap.
3. **Eval the evaluator** — labeled failure-injection corpus in-repo; detector precision/recall published in the README. CI gates TraceGate's own detectors.
4. **Local-first, zero-config start** — `pip install tracegate`, one decorator, traces appear. Friction kills dev-tool adoption.

**Stack:** Python 3.11+, Pydantic schemas, SQLite/JSONL storage, OpenTelemetry SDK, Claude API for judge calls (model-agnostic interface), Typer CLI, Jinja static HTML reports, GitHub Action (composite), pytest plugin (`pytest-tracegate`) as a stretch goal.

---

## 5. Adoption Strategy (Open-Source GTM)

### 5.1 Target user
- **Primary:** Python developers shipping agents to production at startups/mid-size companies — the person who got paged because the agent did something weird and has no trace to debug.
- **Secondary:** AI engineers evaluating "should we ship this agent at all"; eng leads needing a reliability number for stakeholders.

### 5.2 Launch & growth channels
1. **The compounding-error demo** — flagship README asset: a 10-step agent at 95%/step visibly failing 4 times out of 10, with TraceGate pinpointing the originating step each time. Show, don't tell.
2. **Show HN / r/LocalLLaMA / r/LangChain launch** — title in the shape of "TraceGate — pytest for AI agents: record every step, catch silent failures, gate your CI."
3. **Content: the silent-failure taxonomy** — a well-written post naming and demonstrating the 5 failure modes (goal drift, ungrounded assumption, contradiction, loop, tool misuse) can become the reference people link. Own the vocabulary, own the category.
4. **Framework ecosystem PRs/listings** — get into LangGraph/Claude SDK community integrations pages; answer agent-debugging questions in those Discords with traces.
5. **GitHub Action marketplace** — CI-native distribution; teams discover it where they gate deploys.
6. **Benchmark tie-in** — run TraceGate detectors over public agent benchmark traces (e.g., published GAIA/SWE-bench agent runs) and publish "X% of failing runs contained a silent hallucination at step N" findings. Research-flavored content earns citations.

### 5.3 Success metrics
- Launch month: 500 GitHub stars, 3 external issues/PRs, front page of Show HN.
- Month 6: 2,000 stars, 10+ external contributors, 3 companies confirmed using it in CI, the taxonomy post cited by others.
- **Career metric (explicit):** project referenced in interviews; at least one talk/blog invitation.
- Kill/pivot criteria: if by month 4 there's engagement on the *detectors* but not the *gate* (or vice versa), split the project and double down on the half that pulls.

### 5.4 Monetization (deferred, deliberate)
Open-core path only if traction warrants: hosted team dashboard (shared trace history, trend lines, alerts), SSO, retention. Never paywall the SDK, detectors, or CI gate — those are the community contract.

### 5.5 Competition & positioning
- **LangSmith / Langfuse / Braintrust:** trace observability, hosted-first; detection is manual or platform-locked. TraceGate = open, local, CI-native, with opinionated *automatic* failure detection.
- **Ragas / DeepEval / promptfoo:** single-call or RAG evals; no multi-step trace model, no step-level failure attribution.
- **AgentOps / Maxim / Noveum:** commercial agent monitoring; same hosted-first gap.
- **Positioning sentence:** "Observability tools show you the trace; TraceGate reads it, tells you which step went wrong, and blocks the deploy."

### 5.6 Risks
1. **LLM-judge reliability** — the detectors could themselves hallucinate. Mitigation: published precision/recall on the labeled corpus, deterministic-first design, confidence scores on findings.
2. **Framework churn** — agent SDKs change fast. Mitigation: adapters are thin; the normalized trace schema + OTel compatibility absorb change.
3. **Replay non-determinism** — agents don't rerun identically. Mitigation: score *outcomes*, not paths; N-run sampling with pass@k-style reliability scores rather than single-run pass/fail.
4. **Big player ships it natively** — LangChain/Anthropic could bundle equivalent tooling. Mitigation: speed, neutrality (works across all frameworks), and community ownership of the failure taxonomy.

---

## 6. Roadmap

| Phase | Timeline | Deliverable |
|-------|----------|-------------|
| 1 — Core spine | Weeks 1–3 | AgentTrace schema + SDK recorder for Claude Agent SDK; JSONL/SQLite store; minimal CLI (`record`, `show`) |
| 2 — Detectors | Weeks 4–6 | Loop + tool-misuse (deterministic), then ungrounded-assumption + goal-drift (judge); labeled failure-injection corpus + detector eval |
| 3 — Gate | Weeks 7–9 | Replay engine, reliability scoring, `tracegate ci`, GitHub Action, HTML report |
| 4 — Launch | Week 10 | Compounding-error demo, taxonomy blog post, Show HN + Reddit launch |
| 5 — Ecosystem | Months 3–5 | LangGraph + OpenAI Agents adapters, pytest plugin, benchmark-trace findings post, community triage |

**Build order rationale:** Record must exist before Detect has input; Detect before Gate so the gate has signal beyond task success. Launch waits for all three layers because the three-layer story IS the differentiation.

---

## 7. Context for Future Sessions

- **Owner:** Iyman Ahmed — GenAI/ML engineer; job search targeting GenAI/MLOps roles (available 2026-09-20). Relevant prior work: **RAGOps** (eval-driven RAG with CI quality gates, github.com/Iyman-Ahmed/ragops) — TraceGate deliberately extends that methodology and interview narrative from RAG to agents. Also: Clinical RAG POC.
- **Origin:** merged from two ideas explored 2026-07-08 — (a) an agent reliability eval harness with CI gating, and (b) silent-failure detection middleware. Working name was "FlightRecorder"; renamed TraceGate.
- **Sibling projects:** `compliact/` (EU AI Act compliance SaaS — business play) and `carbonproxy/` (SMB emissions reporting — business play). TraceGate is the open-source/portfolio play. A separate future idea, "MCP Shield" (agent tool-use security firewall), is complementary — TraceGate covers accidents, Shield covers attacks. Do not merge them.
- **Status as of 2026-07-08:** Proposal stage. No code. Next concrete step: Phase 1 — design the AgentTrace schema and build the Claude Agent SDK recorder.
- **Key sources behind this proposal:** OpenReview "Tiny Silent Hallucinations in Agentic AI" (openreview.net/forum?id=1KxDazvI6L), Noveum blog on production hallucination detection, Gartner agentic-AI cancellation prediction (via 2026 press coverage), LLM hallucination statistics roundups (sqmagazine.co.uk, suprmind.ai).
