"""Trace every LLM call, tool invocation, and retrieval step.

Stores traces in memory for now; can be extended to persist to DB or file.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from atlas.observability.logger import get_logger

log = get_logger(__name__)


class StepType(StrEnum):
    LLM_CALL = "llm_call"
    RETRIEVAL = "retrieval"
    TOOL_USE = "tool_use"
    RERANK = "rerank"
    EVALUATION = "evaluation"


@dataclass
class TraceStep:
    """A single step in an agent trace."""

    step_id: str
    step_type: StepType
    input_data: dict
    output_data: dict | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class Trace:
    """Full trace for a single research query."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_in + s.tokens_out for s in self.steps)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)
        log.info(
            "trace_step",
            trace_id=self.trace_id,
            step_type=step.step_type,
            latency_ms=round(step.latency_ms, 2),
            tokens_in=step.tokens_in,
            tokens_out=step.tokens_out,
        )

    def summary(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "total_steps": len(self.steps),
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "steps": [
                {
                    "type": s.step_type,
                    "latency_ms": round(s.latency_ms, 2),
                    "tokens": s.tokens_in + s.tokens_out,
                }
                for s in self.steps
            ],
        }


class TraceStore:
    """In-memory trace storage. Replace with DB for production."""

    def __init__(self):
        self._traces: dict[str, Trace] = {}

    def create(self, query: str) -> Trace:
        trace = Trace(query=query)
        self._traces[trace.trace_id] = trace
        return trace

    def get(self, trace_id: str) -> Trace | None:
        return self._traces.get(trace_id)

    def list_recent(self, n: int = 10) -> list[dict]:
        sorted_traces = sorted(self._traces.values(), key=lambda t: t.created_at, reverse=True)
        return [t.summary() for t in sorted_traces[:n]]


# Singleton
trace_store = TraceStore()
