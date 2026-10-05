"""Rubric-aware evaluation and marks aggregation."""

from topper_maker.evaluation.scoring.evaluator import AnswerEvaluator
from topper_maker.evaluation.scoring.marks_aggregator import (
    MarksResult,
    apply_counted_flags,
    compute_total_marks,
)
from topper_maker.evaluation.scoring.rubric_scoring import RawAward, resolve_rubric_score

__all__ = [
    "AnswerEvaluator",
    "MarksResult",
    "apply_counted_flags",
    "compute_total_marks",
    "RawAward",
    "resolve_rubric_score",
]
