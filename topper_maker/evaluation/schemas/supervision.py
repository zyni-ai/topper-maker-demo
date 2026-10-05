"""Supervision audit trail — records the arbitration decisions made by the supervisor model."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field


class SupervisionMode(str, Enum):
    RECONCILE = "reconcile"        # non-critical: supervisor reconciled candidate scores
    RE_EVALUATE = "re_evaluate"    # critical: supervisor independently re-evaluated whole paper


class QuestionSupervision(BaseModel):
    """Per-question record of the supervisor's arbitration decision."""

    question_id: int = Field(..., description="Question ID.")
    mode: SupervisionMode = Field(
        ..., description="Whether the supervisor reconciled or re-evaluated."
    )
    candidate_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Scores from each candidate model, keyed by model string.",
    )
    chosen_score: float = Field(..., description="Score selected/computed by the supervisor.")
    max_score: float = Field(..., description="Maximum marks for the question.")
    deviation: float = Field(
        0.0,
        description="Max absolute deviation between candidate scores (0 if single candidate).",
    )
    cited_rubric_keys: List[str] = Field(
        default_factory=list,
        description="Rubric point keys the supervisor cited in its reason.",
    )
    reason: str = Field("", description="Supervisor's written justification for the chosen score.")
    is_critical: bool = Field(
        False,
        description="True if this question triggered the critical (re-evaluate) path.",
    )
    unresolved: bool = Field(
        False,
        description="True if the supervisor could not confidently resolve the disagreement.",
    )


class SupervisionRecord(BaseModel):
    """Audit trail for a full supervisor arbitration pass."""

    supervisor_model: str = Field(..., description="Model used for supervision.")
    num_candidates: int = Field(..., description="Number of candidate evaluations arbitrated.")
    candidate_models: List[str] = Field(
        default_factory=list, description="Model strings of the candidate evaluators."
    )
    triggered_critical: bool = Field(
        False,
        description="True if the critical path (full re-evaluation) was triggered for any question.",
    )
    has_unresolved: bool = Field(
        False,
        description="True if any question could not be confidently resolved (reconcile path).",
    )
    questions: List[QuestionSupervision] = Field(
        default_factory=list, description="Per-question arbitration decisions."
    )
    total_deviation: float = Field(
        0.0,
        description="Max absolute deviation between candidate paper totals.",
    )
