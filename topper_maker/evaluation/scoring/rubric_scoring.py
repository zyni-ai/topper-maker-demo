"""Deterministic rubric scoring rules.

The LLM judges *whether* each rubric point is satisfied; this module decides *how
many marks* that translates to, applying the board conventions captured in issue
#5 as code (not prompt instructions) so they are reliable and unit-testable:

- **All-or-nothing by default**: a point awards its full marks or zero. Half marks
  are allowed only when the point sets ``allow_partial=True``, in which case the
  award is clamped to ``[0, marks]`` and snapped to the nearest 0.5.
- **Cascading dependencies**: a point with ``depends_on`` is forced to zero if any
  prerequisite point scored zero (e.g. no substitution mark when the formula is
  wrong). Enforced as a fixpoint so chains of dependencies resolve correctly.
- The final score is clamped to the question's ``max_score``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from topper_maker.evaluation.schemas.feedback import RubricAward
from topper_maker.evaluation.schemas.question import RubricPoint

_EPSILON = 1e-6


@dataclass
class RawAward:
    """The model's per-point judgement, before deterministic rules are applied."""

    awarded: bool
    marks_awarded: float = 0.0
    rationale: str = ""


def _snap_half(value: float, ceiling: float) -> float:
    """Clamp to [0, ceiling] and snap to the nearest 0.5."""
    value = max(0.0, min(value, ceiling))
    return round(value * 2) / 2


def resolve_rubric_score(
    points: List[RubricPoint],
    raw_awards: Dict[str, RawAward],
    max_score: float,
) -> Tuple[float, List[RubricAward]]:
    """Apply marking rules to the model's point judgements.

    Args:
        points: The rubric points for the question.
        raw_awards: Map of rubric point key → the model's judgement. Missing keys
            are treated as not awarded.
        max_score: The question's maximum marks (final clamp).

    Returns:
        ``(score, breakdown)`` where ``breakdown`` is one :class:`RubricAward` per
        point, in the rubric's order, including any forced-to-zero annotations.
    """
    # 1. Initial award per point (partial rules applied, no dependencies yet).
    awarded_marks: Dict[str, float] = {}
    rationale: Dict[str, str] = {}
    for point in points:
        raw = raw_awards.get(point.key, RawAward(awarded=False))
        rationale[point.key] = raw.rationale
        if not raw.awarded:
            awarded_marks[point.key] = 0.0
        elif point.allow_partial:
            awarded_marks[point.key] = _snap_half(raw.marks_awarded, point.marks)
        else:
            awarded_marks[point.key] = point.marks

    # 2. Cascading dependencies — fixpoint: zero any point whose prerequisite is zero.
    forced_zero: Dict[str, bool] = {p.key: False for p in points}
    point_by_key = {p.key: p for p in points}
    for _ in range(len(points)):  # at most N iterations to converge
        changed = False
        for point in points:
            if awarded_marks[point.key] <= _EPSILON:
                continue
            for dep_key in point.depends_on:
                dep_marks = awarded_marks.get(dep_key)
                # Unknown dependency keys are ignored (treated as satisfied) so a
                # typo in the rubric never silently zeroes a student's marks.
                if dep_key in point_by_key and (dep_marks is None or dep_marks <= _EPSILON):
                    awarded_marks[point.key] = 0.0
                    forced_zero[point.key] = True
                    changed = True
                    break
        if not changed:
            break

    # 3. Build the audit breakdown and total.
    breakdown: List[RubricAward] = []
    for point in points:
        marks = awarded_marks[point.key]
        breakdown.append(
            RubricAward(
                key=point.key,
                description=point.description,
                marks_possible=point.marks,
                marks_awarded=marks,
                awarded=marks > _EPSILON,
                rationale=rationale[point.key],
                forced_zero_by_dependency=forced_zero[point.key],
            )
        )

    score = min(sum(awarded_marks.values()), max_score)
    return score, breakdown
