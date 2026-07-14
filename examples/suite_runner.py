"""Simulated agent runner for the example suite (no API key needed).

Run:  agentgates ci --suite examples/suite.toml --runner examples/suite_runner.py:run_agent --threshold 0.9
"""

from agentgates import TraceRecorder


class _NullStore:
    def save(self, trace):
        pass


def run_agent(task: str):
    rec = TraceRecorder(task=task, framework="demo", model="simulated", store=_NullStore())
    rec.record_llm_call(model="simulated", prompt=task, response="Searching for options.")
    rec.record_tool_call(
        tool_name="search",
        arguments={"q": task},
        result={"best": "UA at $420"},
        latency_ms=120.0,
    )
    rec.record_llm_call(
        model="simulated",
        prompt="",
        response=f"Task complete: {task}. Best option UA at $420.",
    )
    return rec.finish()
