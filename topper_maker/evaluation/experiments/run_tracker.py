"""Persist each evaluation run as a LangSmith dataset example.

Every evaluated booklet becomes one example in the ``topper-maker-eval-runs``
dataset, so runs survive a Streamlit refresh and can be compared over time.

PRIVACY: only **metrics** are stored — subject, model, wall time, cost, and the
system's marks — plus a slot for human-verified marks entered later. No student
answer text, transcription, or page image is ever uploaded. This is deliberately
decoupled from ``LANGCHAIN_TRACING_V2`` (which uploads prompts/images and must
stay OFF for real student data): logging here needs only a ``LANGSMITH_API_KEY``
and carries no PII.

Accuracy/deviation are derived from the system total vs the human total marks.
The arithmetic lives in :func:`accuracy_metrics` (pure, unit-tested).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DATASET_NAME = "topper-maker-eval-runs"

# Review reasons that mean the pipeline did NOT produce a trustworthy grade (an
# operational failure, not a low-confidence grade). A 0/N from one of these is a
# crash, not a grading miss — so such runs are recorded but excluded from accuracy
# stats (otherwise a mapping failure masquerades as the system "grading badly").
FAILED_REASONS = frozenset({"mapping_error", "evaluation_error", "moderation_error"})
_DATASET_DESCRIPTION = (
    "Topper Maker evaluation runs — one example per graded booklet. Metrics only "
    "(subject, model, time, cost, system marks, human marks); no student content. "
    "Human total marks enable accuracy/deviation tracking."
)

# Cache the resolved dataset id within a process so we don't look it up every call.
_dataset_id: Optional[str] = None


# -- Pure accuracy math (unit-tested) ------------------------------------------

def accuracy_metrics(
    system_total: float, human_total: float, max_marks: float
) -> Dict[str, Optional[float]]:
    """Deviation, absolute error, and accuracy% of the system vs the human total.

    - ``deviation`` is signed (system − human): positive = system over-marked.
    - ``abs_error`` is its magnitude.
    - ``accuracy_pct`` = ``(1 − abs_error / max_marks) × 100``, clamped to [0, 100];
      ``None`` when ``max_marks`` is 0 (undefined).
    """
    deviation = round(system_total - human_total, 4)
    abs_error = round(abs(deviation), 4)
    if not max_marks:
        accuracy_pct: Optional[float] = None
    else:
        accuracy_pct = round(max(0.0, min(100.0, (1.0 - abs_error / max_marks) * 100.0)), 2)
    return {"deviation": deviation, "abs_error": abs_error, "accuracy_pct": accuracy_pct}


# -- Record view ---------------------------------------------------------------

@dataclass
class RunRecord:
    """Flattened view of one logged run, for UI display and editing."""

    example_id: str
    subject: str
    model: str
    pdf_name: str
    logged_at: str
    system_total: float
    max_marks: float
    system_percentage: float
    time_sec: float
    cost_usd: Optional[float]
    needs_review: bool
    review_reasons: List[str] = field(default_factory=list)
    system_per_question: Dict[str, float] = field(default_factory=dict)
    human_total: Optional[float] = None
    human_per_question: Dict[str, float] = field(default_factory=dict)
    deviation: Optional[float] = None
    abs_error: Optional[float] = None
    accuracy_pct: Optional[float] = None

    @property
    def has_human_marks(self) -> bool:
        return self.human_total is not None

    @property
    def failed(self) -> bool:
        """True if the run hit an operational failure (no trustworthy grade)."""
        return bool(set(self.review_reasons) & FAILED_REASONS)


# -- LangSmith plumbing --------------------------------------------------------

def is_available() -> bool:
    """True if a LangSmith API key is configured (logging will work)."""
    return bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


def _client():
    from langsmith import Client  # imported lazily so the app runs without it

    return Client()


def _ensure_dataset(client) -> str:
    global _dataset_id
    if _dataset_id:
        return _dataset_id
    try:
        ds = client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:  # noqa: BLE001 - not found (or transient); try to create
        try:
            ds = client.create_dataset(
                dataset_name=DATASET_NAME, description=_DATASET_DESCRIPTION
            )
        except Exception as exc:  # noqa: BLE001 - lost a create race; re-read
            if "already exists" in str(exc).lower():
                ds = client.read_dataset(dataset_name=DATASET_NAME)
            else:
                raise
    _dataset_id = str(ds.id)
    return _dataset_id


def _record_from_example(ex: Any) -> RunRecord:
    inp = ex.inputs or {}
    out = ex.outputs or {}
    return RunRecord(
        example_id=str(ex.id),
        subject=inp.get("subject", "?"),
        model=inp.get("model", "?"),
        pdf_name=inp.get("pdf_name", ""),
        logged_at=inp.get("logged_at", ""),
        system_total=float(out.get("system_total", 0.0)),
        max_marks=float(out.get("max_marks", 0.0)),
        system_percentage=float(out.get("system_percentage", 0.0)),
        time_sec=float(out.get("time_sec", 0.0)),
        cost_usd=out.get("cost_usd"),
        needs_review=bool(out.get("needs_review", False)),
        review_reasons=list(out.get("review_reasons", []) or []),
        system_per_question=dict(out.get("system_per_question", {}) or {}),
        human_total=out.get("human_total"),
        human_per_question=dict(out.get("human_per_question", {}) or {}),
        deviation=out.get("deviation"),
        abs_error=out.get("abs_error"),
        accuracy_pct=out.get("accuracy_pct"),
    )


# -- Public API ----------------------------------------------------------------

def log_run(
    *,
    subject: str,
    model: str,
    pdf_name: str,
    total_marks: float,
    max_marks: float,
    percentage: float,
    time_sec: float,
    cost_usd: Optional[float],
    needs_review: bool,
    review_reasons: Optional[List[str]] = None,
    per_question: Optional[Dict[str, float]] = None,
    stage_timings_ms: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """Log one evaluated booklet as a dataset example. Returns its id, or None.

    Never raises: a logging failure must not break a demo run, so any error is
    swallowed and ``None`` returned (the caller surfaces a soft notice).
    """
    if not is_available():
        return None
    try:
        client = _client()
        dataset_id = _ensure_dataset(client)
        ex = client.create_example(
            inputs={
                "subject": subject,
                "model": model,
                "pdf_name": pdf_name,
                "logged_at": datetime.now(timezone.utc).isoformat(),
            },
            outputs={
                "system_total": round(float(total_marks), 4),
                "max_marks": round(float(max_marks), 4),
                "system_percentage": round(float(percentage), 4),
                "time_sec": round(float(time_sec), 3),
                "cost_usd": (round(float(cost_usd), 6) if cost_usd is not None else None),
                "needs_review": bool(needs_review),
                "review_reasons": list(review_reasons or []),
                "system_per_question": {str(k): v for k, v in (per_question or {}).items()},
                # human_* / deviation filled in later via set_human_marks
                "human_total": None,
                "human_per_question": {},
            },
            metadata={
                "subject": subject,
                "model": model,
                "stage_timings_ms": stage_timings_ms or {},
            },
            dataset_id=dataset_id,
        )
        return str(ex.id)
    except Exception as exc:  # noqa: BLE001 - logging is best-effort
        logger.warning("Run logging to LangSmith failed (continuing): %s", exc)
        return None


def set_human_marks(
    example_id: str,
    *,
    human_total: float,
    human_per_question: Optional[Dict[str, float]] = None,
) -> RunRecord:
    """Attach human-verified marks to a logged run and recompute accuracy.

    Reads the existing example (for the system total + max), merges in the human
    marks and derived metrics, and writes them back. Returns the updated record.
    """
    client = _client()
    ex = client.read_example(example_id)
    out = dict(ex.outputs or {})

    metrics = accuracy_metrics(
        system_total=float(out.get("system_total", 0.0)),
        human_total=float(human_total),
        max_marks=float(out.get("max_marks", 0.0)),
    )
    # A failed run produced no real grade: keep the human total and the signed gap
    # (the marks the failure cost), but accuracy is undefined — don't let it count.
    run_failed = bool(set(out.get("review_reasons", []) or []) & FAILED_REASONS)
    if run_failed:
        metrics["accuracy_pct"] = None
    out.update(
        human_total=round(float(human_total), 4),
        human_per_question={str(k): v for k, v in (human_per_question or {}).items()},
        **metrics,
    )
    client.update_example(example_id, outputs=out)
    # Re-read so the returned record reflects exactly what was persisted.
    return _record_from_example(client.read_example(example_id))


def list_runs(limit: int = 200) -> List[RunRecord]:
    """All logged runs, newest first. Empty list if LangSmith is unavailable."""
    if not is_available():
        return []
    client = _client()
    dataset_id = _ensure_dataset(client)
    records = [_record_from_example(ex) for ex in client.list_examples(dataset_id=dataset_id)]
    records.sort(key=lambda r: r.logged_at, reverse=True)
    return records[:limit]


def get_run(example_id: str) -> RunRecord:
    """Fetch one logged run by id."""
    return _record_from_example(_client().read_example(example_id))
