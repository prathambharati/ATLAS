"""Cost tracking for API calls.

Tracks token usage and estimates cost per query, per tool,
and aggregated across all queries. Provides data for the
cost dashboard in the frontend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock

from atlas.observability.logger import get_logger

log = get_logger(__name__)

# Pricing per 1M tokens (as of 2024-2025)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


@dataclass
class QueryCost:
    """Cost breakdown for a single query."""

    query: str
    trace_id: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    timestamp: float = field(default_factory=time.time)
    breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query[:80],
            "trace_id": self.trace_id,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "model": self.model,
            "timestamp": self.timestamp,
            "breakdown": self.breakdown,
        }


class CostTracker:
    """Track API costs across all queries."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._queries: list[QueryCost] = []
        self._lock = Lock()
        self._pricing = MODEL_PRICING.get(
            model, {"input": 0.15, "output": 0.60}
        )

    def record(
        self,
        query: str,
        tokens_in: int,
        tokens_out: int,
        trace_id: str = "",
        breakdown: dict | None = None,
    ) -> QueryCost:
        """Record cost for a query."""
        total = tokens_in + tokens_out
        cost = (
            tokens_in * self._pricing["input"] / 1_000_000
            + tokens_out * self._pricing["output"] / 1_000_000
        )

        entry = QueryCost(
            query=query,
            trace_id=trace_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=total,
            cost_usd=cost,
            model=self.model,
            breakdown=breakdown or {},
        )

        with self._lock:
            self._queries.append(entry)

        log.info(
            "cost_recorded",
            query=query[:40],
            tokens=total,
            cost_usd=round(cost, 6),
        )
        return entry

    def get_summary(self) -> dict:
        """Get aggregate cost summary."""
        with self._lock:
            queries = list(self._queries)

        if not queries:
            return {
                "total_queries": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_tokens_per_query": 0,
                "avg_cost_per_query": 0.0,
                "model": self.model,
                "queries": [],
            }

        total_tokens = sum(q.total_tokens for q in queries)
        total_cost = sum(q.cost_usd for q in queries)

        return {
            "total_queries": len(queries),
            "total_tokens": total_tokens,
            "total_tokens_in": sum(q.tokens_in for q in queries),
            "total_tokens_out": sum(q.tokens_out for q in queries),
            "total_cost_usd": round(total_cost, 6),
            "avg_tokens_per_query": total_tokens // len(queries),
            "avg_cost_per_query": round(total_cost / len(queries), 6),
            "model": self.model,
            "pricing": self._pricing,
            "queries": [q.to_dict() for q in queries[-20:]],
        }

    def get_recent(self, n: int = 10) -> list[dict]:
        """Get the most recent queries."""
        with self._lock:
            return [q.to_dict() for q in self._queries[-n:]]

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(q.cost_usd for q in self._queries)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(q.total_tokens for q in self._queries)


# Singleton
cost_tracker = CostTracker()
