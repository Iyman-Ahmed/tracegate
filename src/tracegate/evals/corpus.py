"""Programmatically generated labeled corpus with injected failures."""

from __future__ import annotations

from dataclasses import dataclass, field

from tracegate.schema import AgentInfo, AgentTrace, LLMCallStep, TaskSpec, ToolCallStep


@dataclass
class LabeledTrace:
    trace: AgentTrace
    expected_detectors: set[str] = field(default_factory=set)


def _trace(desc: str, steps: list) -> AgentTrace:
    trace = AgentTrace(task=TaskSpec(description=desc), agent=AgentInfo(framework="corpus"))
    trace.steps = steps
    return trace


def _llm(i: int, text: str) -> LLMCallStep:
    return LLMCallStep(index=i, model="corpus", prompt="", response=text)


def _tool(i: int, name: str, args: dict, result=None, error=None) -> ToolCallStep:
    return ToolCallStep(index=i, tool_name=name, arguments=args, result=result, error=error)


def _clean(n: int) -> AgentTrace:
    return _trace(
        f"clean task {n}",
        [
            _llm(0, f"planning approach {n}"),
            _tool(1, "search", {"q": f"query {n}"}, result=f"result {n}"),
            _llm(2, f"found answer {n}"),
            _tool(3, "save", {"value": n}, result="saved"),
            _llm(4, f"done with {n}"),
        ],
    )


def _loop_consecutive() -> AgentTrace:
    return _trace(
        "loop: same call repeated",
        [_tool(i, "fetch", {"url": "same"}, result="pending") for i in range(4)],
    )


def _loop_llm_repeat() -> AgentTrace:
    return _trace(
        "loop: agent restates itself",
        [_llm(i, "I will try again") for i in range(3)],
    )


def _loop_scattered() -> AgentTrace:
    return _trace(
        "loop: same call scattered",
        [
            _tool(0, "check", {"id": 7}, result="no"),
            _llm(1, "checking once more"),
            _tool(2, "check", {"id": 7}, result="no"),
            _llm(3, "one more look"),
            _tool(4, "check", {"id": 7}, result="no"),
        ],
    )


def _misuse_error_ignored() -> AgentTrace:
    return _trace(
        "misuse: error then unrelated continue",
        [
            _tool(0, "fetch", {"url": "a"}, error="404 not found"),
            _llm(1, "continuing anyway"),
        ],
    )


def _misuse_blind_retry() -> AgentTrace:
    return _trace(
        "misuse: identical failing retry",
        [
            _tool(0, "post", {"body": "x"}, error="500"),
            _tool(1, "post", {"body": "x"}, error="500"),
            _llm(2, "giving up"),
        ],
    )


def _misuse_two_errors() -> AgentTrace:
    return _trace(
        "misuse: two different tools error",
        [
            _tool(0, "read", {"path": "a"}, error="permission denied"),
            _tool(1, "write", {"path": "b"}, error="disk full"),
        ],
    )


def _contradiction_stale_price() -> AgentTrace:
    return _trace(
        "contradiction: books at a fare that changed",
        [
            _tool(0, "get_price", {"flight": "UA420"}, result=420),
            _llm(1, "That fare works. Checking seat availability."),
            _tool(2, "get_price", {"flight": "UA420"}, result=580),
            _llm(3, "Booked! Total charged: 420."),
        ],
    )


def _contradiction_stale_stock() -> AgentTrace:
    return _trace(
        "contradiction: promises stock that ran out",
        [
            _tool(0, "check_stock", {"sku": "A17"}, result=5),
            _tool(1, "check_stock", {"sku": "A17"}, result=0),
            _llm(2, "5 units are available, so I'll place the order."),
        ],
    )


def _combined(n: int) -> AgentTrace:
    return _trace(
        f"combined failure {n}",
        [
            _tool(0, "poll", {"job": n}, error="timeout"),
            _tool(1, "poll", {"job": n}, error="timeout"),
            _tool(2, "poll", {"job": n}, error="timeout"),
            _llm(3, "still waiting"),
        ],
    )


def build_corpus() -> list[LabeledTrace]:
    corpus = [LabeledTrace(_clean(n)) for n in range(1, 5)]
    corpus += [
        LabeledTrace(_loop_consecutive(), {"loop"}),
        LabeledTrace(_loop_llm_repeat(), {"loop"}),
        LabeledTrace(_loop_scattered(), {"loop"}),
        LabeledTrace(_misuse_error_ignored(), {"tool_misuse"}),
        LabeledTrace(_misuse_blind_retry(), {"tool_misuse"}),
        LabeledTrace(_misuse_two_errors(), {"tool_misuse"}),
        LabeledTrace(_contradiction_stale_price(), {"contradiction"}),
        LabeledTrace(_contradiction_stale_stock(), {"contradiction"}),
        LabeledTrace(_combined(1), {"loop", "tool_misuse"}),
        LabeledTrace(_combined(2), {"loop", "tool_misuse"}),
    ]
    return corpus


# --- Judge corpus -----------------------------------------------------------
# Semantic failures the deterministic detectors can't see. Scored by the judge
# detectors; each trace is single-LLM-step so the ungrounded detector makes at
# most one call, keeping the labeled outcome unambiguous.


def _drift_overspend() -> AgentTrace:
    trace = AgentTrace(
        task=TaskSpec(
            description="book the cheapest flight under $500",
            constraints=["under $500"],
        ),
        agent=AgentInfo(framework="corpus"),
    )
    trace.steps = [
        _llm(0, "The premium cabin at $900 is much nicer, so I'll book that.")
    ]
    return trace


def _drift_scope() -> AgentTrace:
    trace = AgentTrace(
        task=TaskSpec(description="reply to the customer's refund request only"),
        agent=AgentInfo(framework="corpus"),
    )
    trace.steps = [
        _llm(0, "While I'm here I'll also change their subscription plan and resubscribe them to marketing.")
    ]
    return trace


def _ungrounded_wifi() -> AgentTrace:
    return _trace(
        "find a flight for the customer",
        [
            _tool(0, "search_flights", {"to": "NYC"}, result={"flights": ["UA420"]}),
            _llm(1, "UA420 includes free wifi and lounge access, so it's the best pick."),
        ],
    )


def _ungrounded_price() -> AgentTrace:
    return _trace(
        "look up the order total",
        [
            _tool(0, "get_order", {"id": "A17"}, result={"items": 3}),
            _llm(1, "The total is $1,240, so I'll mark the order as paid."),
        ],
    )


def _grounded_summary() -> AgentTrace:
    return _trace(
        "summarize the search result",
        [
            _tool(0, "search", {"q": "flights"}, result={"summary": "three flights found"}),
            _llm(1, "Three flights were found."),
        ],
    )


def _grounded_plan() -> AgentTrace:
    return _trace(
        "plan the booking steps",
        [_llm(0, "I'll search flights, then filter by price, then book the cheapest.")],
    )


def build_judge_corpus() -> list[LabeledTrace]:
    return [
        LabeledTrace(_drift_overspend(), {"goal_drift"}),
        LabeledTrace(_drift_scope(), {"goal_drift"}),
        LabeledTrace(_ungrounded_wifi(), {"ungrounded_assumption"}),
        LabeledTrace(_ungrounded_price(), {"ungrounded_assumption"}),
        LabeledTrace(_grounded_summary()),
        LabeledTrace(_grounded_plan()),
    ]
