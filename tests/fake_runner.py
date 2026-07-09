"""Importable fake agent runners for gate tests (not a test module)."""

from tracegate.recorder import TraceRecorder


class _NullStore:
    def save(self, trace):
        pass


def good_runner(task: str):
    rec = TraceRecorder(task=task, framework="fake", store=_NullStore())
    rec.record_llm_call(model="m", prompt=task, response=f"done: {task} SUCCESS_TOKEN")
    return rec.finish()


def loopy_runner(task: str):
    rec = TraceRecorder(task=task, framework="fake", store=_NullStore())
    for _ in range(3):
        rec.record_tool_call(tool_name="poll", arguments={"t": task}, error="timeout")
    rec.record_llm_call(model="m", prompt="", response="gave up")
    return rec.finish()


def metadata_runner(task: str):
    rec = TraceRecorder(task=task, framework="fake", store=_NullStore())
    rec.trace.metadata["result"] = "final answer: 42"
    return rec.finish()


def crashing_runner(task: str):
    raise RuntimeError("agent exploded")
