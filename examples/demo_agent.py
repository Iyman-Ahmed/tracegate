"""Simulated 5-step agent run — records a trace without any API key.

Run:  agentgates record examples/demo_agent.py
Then: agentgates list && agentgates show <trace_id>
"""

from agentgates import TraceRecorder

with TraceRecorder(
    task="Find the cheapest flight SFO->NYC under $500 next Friday",
    framework="demo",
    model="simulated",
) as rec:
    rec.record_llm_call(
        model="simulated",
        prompt="Find the cheapest flight SFO->NYC under $500 next Friday",
        response="I'll search for flights, then filter by price.",
    )
    rec.record_tool_call(
        tool_name="search_flights",
        arguments={"from": "SFO", "to": "NYC", "date": "next Friday"},
        result=[{"airline": "UA", "price": 420}, {"airline": "DL", "price": 510}],
        latency_ms=132.0,
    )
    rec.record_llm_call(
        model="simulated",
        prompt="",
        response="UA at $420 fits the budget. Checking seat availability.",
    )
    rec.record_tool_call(
        tool_name="check_seats",
        arguments={"airline": "UA", "price": 420},
        result={"available": True},
        latency_ms=87.0,
    )
    rec.record_llm_call(
        model="simulated",
        prompt="",
        response="Done: UA flight at $420, seats available.",
    )

print(f"recorded trace {rec.trace.trace_id} with {len(rec.trace.steps)} steps")
