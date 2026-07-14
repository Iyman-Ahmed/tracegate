# The Silent-Failure Taxonomy for AI Agents

The scariest agent bugs never throw. The run finishes, the output reads well,
and the agent reports success — but somewhere in the middle, a step went wrong
and everything after it was built on sand. We call these **silent failures**,
and after instrumenting multi-step agents we keep seeing the same five shapes.

Naming a failure mode is the first step to detecting it. This document defines
the taxonomy AgentGates's detectors are built around.

---

## 1. Goal drift

**What it is:** the agent's *working objective* diverges from the task it was
given. Most often this is a **dropped constraint**: asked for "the cheapest
flight under $500," the agent gets absorbed in comparing airlines and books a
$900 flight — confidently, helpfully, and wrong.

**Why it's silent:** every individual step is locally reasonable. Only the
relationship between the original task spec and the step's intent reveals the
problem, and nothing in the stack ever re-checks that relationship.

**Example trace:**

```
task: "Find the cheapest flight SFO->NYC under $500"
[0] llm:  I'll search for flights.
[1] tool: search_flights(...) -> [UA $420, DL $510, Polaris $890]
[2] llm:  The Polaris seats look far more comfortable. Booking Polaris at $890.
```

**AgentGates detector:** `goal_drift` (LLM-judge; compares the task spec +
constraints against each step's intent, flags the first drifting step and the
dropped constraint).

---

## 2. Ungrounded assumption (silent hallucination)

**What it is:** the agent states — or worse, *acts on* — a "fact" that appears
in no tool result, retrieval, or user input. It invented a belief and never
surfaced it as an assumption. Recent research calls these "tiny silent
hallucinations": false intermediate beliefs that steer the run without ever
being visible in the final answer.

**Why it's silent:** the fabricated fact usually sounds plausible and often
doesn't appear verbatim in the output — it just changes which branch the agent
takes.

**Example trace:**

```
[1] tool: search_flights(...) -> [UA 420 at $420]
[2] llm:  UA 420 includes free wifi and checked bags, so it's the best value. Booking it.
```

Nothing anywhere said wifi or bags. The *decision* was steered by invented
evidence.

**AgentGates detector:** `ungrounded_assumption` (LLM-judge; audits each agent
message against the evidence accumulated so far — task, tool results, prior
messages — and lists claims with no support). This is AgentGates's signature
detector.

---

## 3. Contradiction

**What it is:** the agent keeps working from information that another tool
result — earlier or later — contradicts. A price check at step 2 says $420; a
re-check at step 6 says $580; the agent quotes $420 in the final answer.

**Why it's silent:** both facts are individually grounded. The failure is in
*reconciliation* — the agent never noticed the conflict.

**Example trace:**

```
[2] tool: get_price(UA420) -> {"price": 420}
[6] tool: get_price(UA420) -> {"price": 580}   # fare changed
[7] llm:  Booked! Total charged: $420.
```

**AgentGates detector:** `contradiction` (deterministic; when the same tool with
the same arguments returns two different results, the agent holds conflicting
evidence — a warning. When a later step still acts on the value that was
superseded, that's an error). Semantic contradictions that aren't exact-value
conflicts remain judge territory.

---

## 4. Loop / stall

**What it is:** the agent repeats semantically near-identical steps — the same
tool with the same arguments, the same "let me try again" message — burning
budget without state progress. Runs that loop usually end by quietly giving up
and declaring success anyway.

**Why it's silent:** each retry looks like legitimate work in a log stream.
Only the pattern across steps reveals that nothing is advancing.

**Example trace:**

```
[3] tool: poll_job({"id": 7}) -> "pending"
[4] tool: poll_job({"id": 7}) -> "pending"
[5] tool: poll_job({"id": 7}) -> "pending"
[6] llm:  The job completed. Moving on.
```

**AgentGates detector:** `loop` (deterministic; fingerprints each step and
flags ≥N identical consecutive steps as errors and ≥N identical tool calls
anywhere as warnings — zero LLM cost).

---

## 5. Tool misuse

**What it is:** a family of tool-boundary failures: malformed or hallucinated
arguments, **ignored tool errors** (the tool said 404 and the agent carried
on), blind identical retries after an error, and referencing tool output that
was never produced.

**Why it's silent:** agent frameworks hand tool errors back to the model as
just another message. Nothing forces the model to *deal* with the error, and
often it simply… doesn't.

**Example trace:**

```
[0] tool: fetch_invoice({"id": "A17"}) -> ERROR 404 not found
[1] llm:  Invoice A17 retrieved. The total is $1,240. Marking as paid.
```

**AgentGates detector:** `tool_misuse` (deterministic; warns on every errored
tool call, escalates to error on identical-argument retries that fail again).
Fabricated references to tool output are covered by `ungrounded_assumption`.

---

## Why detection has to be a pipeline, not a dashboard

All five modes share a property: **you cannot see them in the output.** You
can only see them in the *trace* — the structured record of what the agent
believed, called, and received at every step. That's why AgentGates is built as
record → detect → gate:

1. **Record** every step into a normalized `AgentTrace` (local-first, JSONL/SQLite).
2. **Detect** with cheap deterministic checks first (loop, tool misuse) and
   LLM-judge only where semantics demand it (drift, groundedness) — with the
   detectors themselves scored against a labeled failure-injection corpus.
3. **Gate** deploys in CI: replay a suite, score outcomes, fail the build when
   the reliability score drops.

An agent that is 95% reliable per step is ~59% reliable across 10 steps.
Compounding is the enemy; step-level attribution is the weapon.
