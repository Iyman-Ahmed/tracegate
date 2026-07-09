"""TraceGate — the black box flight recorder for AI agents."""

from tracegate.recorder import TraceRecorder
from tracegate.schema import (
    AgentInfo,
    AgentTrace,
    LLMCallStep,
    TaskSpec,
    TokenUsage,
    ToolCallStep,
)

__version__ = "0.1.0"

__all__ = [
    "AgentInfo",
    "AgentTrace",
    "LLMCallStep",
    "TaskSpec",
    "TokenUsage",
    "ToolCallStep",
    "TraceRecorder",
    "__version__",
]
