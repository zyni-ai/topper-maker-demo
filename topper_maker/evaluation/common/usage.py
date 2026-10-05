"""Shared LLM token/cost accounting.

A single :class:`UsageTracker` instance is threaded through every client used in one
evaluation (the OpenRouter chat client for moderation/mapping/scoring **and** the
OpenRouter HTR client for extraction) so the pipeline can report the **total cost of
one answer-sheet evaluation**.

Cost is taken from OpenRouter's own accounting (returned on the response ``usage``
object when the request opts in via ``usage: {include: true}``), so we never hard-code
a price table. Token counts come from the same ``usage`` object.

The tracker is a plain mutable object shared by reference; ``record()`` only appends to
a list (atomic under the GIL), so it is safe to call from the bounded-concurrency HTR
worker threads and the async scoring tasks alike. Totals are read once, after all work
for the evaluation has completed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UsageRecord:
    """One LLM call's token + cost usage."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class UsageTracker:
    """Accumulates per-call usage across all clients in one evaluation."""

    records: List[UsageRecord] = field(default_factory=list)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        self.records.append(
            UsageRecord(
                model=model,
                prompt_tokens=int(prompt_tokens or 0),
                completion_tokens=int(completion_tokens or 0),
                cost_usd=float(cost_usd or 0.0),
            )
        )

    def record_response(self, response: Any, model: str) -> None:
        """Extract usage from an OpenAI-style response object and record it.

        OpenRouter returns cost on ``usage.cost`` (USD) when the request opts in. The
        SDK may surface it as a typed attribute or only in ``model_extra``; both are
        handled. Missing usage is silently ignored — accounting must never break a run.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        cost = getattr(usage, "cost", None)
        if cost is None:
            extra = getattr(usage, "model_extra", None) or {}
            cost = extra.get("cost")
        self.record(
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            cost_usd=cost or 0.0,
        )

    # -- Totals ------------------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.records), 6)

    @property
    def prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def by_model(self) -> Dict[str, Dict[str, float]]:
        """Per-model rollup of calls, tokens, and cost."""
        out: Dict[str, Dict[str, float]] = {}
        for r in self.records:
            m = out.setdefault(
                r.model,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
            )
            m["calls"] += 1
            m["prompt_tokens"] += r.prompt_tokens
            m["completion_tokens"] += r.completion_tokens
            m["cost_usd"] = round(m["cost_usd"] + r.cost_usd, 6)
        return out

    def summary(self) -> Dict[str, Any]:
        """Log-/UI-friendly overview. ``cost_reported`` flags whether OpenRouter
        actually returned cost figures (vs all-zero, e.g. tracing/usage off)."""
        return {
            "num_calls": len(self.records),
            "total_cost_usd": self.total_cost_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_reported": any(r.cost_usd > 0 for r in self.records),
            "by_model": self.by_model(),
        }


def usage_or_none(tracker: Optional["UsageTracker"]) -> Optional[Dict[str, Any]]:
    """Summary dict if the tracker saw any calls, else None."""
    if tracker is None or not tracker.records:
        return None
    return tracker.summary()
