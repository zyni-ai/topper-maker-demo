"""Marks aggregation with optional-section support.

Ported from the reference engine and adapted to float scores and this package's
schemas. For optional sections ("answer any five of the following"), the top-N
highest-scoring answered questions are counted and the rest excluded.

Tie-breaking when scores are equal: prefer the higher-max-score question (harder),
then the lower question id (stable ordering).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from topper_maker.evaluation.schemas.feedback import IndividualFeedback, SectionSummary
from topper_maker.evaluation.schemas.question import QuestionItem

_NO_SECTION = "__no_section__"


@dataclass
class MarksResult:
    total_marks: float
    max_marks: float
    percentage: float
    section_summaries: Optional[List[SectionSummary]]
    counted_question_ids: Set[int]


def compute_total_marks(
    responses: List[IndividualFeedback],
    questions: List[QuestionItem],
    optional_sections: Optional[Dict[str, int]] = None,
) -> MarksResult:
    """Compute totals, honouring optional-section top-N rules.

    Only *answered* questions participate; unattempted questions contribute their
    max_score to neither total nor maximum in optional sections (the student chose
    not to answer them), but in mandatory sections every question's max counts
    toward the maximum even if unanswered — matching how a board totals a paper.
    """
    optional_sections = optional_sections or {}
    question_map = {q.id: q for q in questions}
    response_map = {r.id: r for r in responses}

    sections: Dict[str, List[int]] = defaultdict(list)
    for q in questions:
        sections[q.section_id or _NO_SECTION].append(q.id)

    total_marks = 0.0
    max_marks = 0.0
    summaries: List[SectionSummary] = []
    counted: Set[int] = set()

    for section_id, qids in sections.items():
        is_optional = section_id in optional_sections

        answered = [
            (qid, response_map[qid], question_map[qid])
            for qid in qids
            if qid in response_map and _is_attempted(response_map[qid])
        ]

        if is_optional:
            required = optional_sections[section_id]
            ranked = sorted(answered, key=lambda t: (-t[1].score, -t[2].max_score, t[0]))
            selected = ranked[:required]
            excluded = ranked[required:]
            section_type = "optional"
            questions_required = required
        else:
            selected = answered
            excluded = []
            section_type = "mandatory"
            questions_required = len(qids)

        section_scored = sum(r.score for _, r, _ in selected)
        if is_optional:
            # Fixed ceiling: sum of the top-N max_scores available in the section.
            # Must NOT depend on what was detected — the paper's total is fixed.
            all_maxes = sorted(
                (question_map[qid].max_score for qid in qids), reverse=True
            )
            section_possible = sum(all_maxes[:required])
        else:
            # Mandatory: the maximum includes every question in the section.
            section_possible = sum(question_map[qid].max_score for qid in qids)

        total_marks += section_scored
        max_marks += section_possible
        counted.update(qid for qid, _, _ in selected)

        if section_id != _NO_SECTION:
            summaries.append(
                SectionSummary(
                    section_id=section_id,
                    section_type=section_type,
                    questions_answered=len(answered),
                    questions_required=questions_required,
                    questions_counted=[qid for qid, _, _ in selected],
                    questions_excluded=[qid for qid, _, _ in excluded],
                    marks_scored=round(section_scored, 2),
                    marks_possible=round(section_possible, 2),
                )
            )

    percentage = round((total_marks / max_marks) * 100, 2) if max_marks > 0 else 0.0
    return MarksResult(
        total_marks=round(total_marks, 2),
        max_marks=round(max_marks, 2),
        percentage=percentage,
        section_summaries=summaries or None,
        counted_question_ids=counted,
    )


def apply_counted_flags(
    responses: List[IndividualFeedback], counted_question_ids: Set[int]
) -> None:
    """Set ``counted_in_total`` on each response in place."""
    for response in responses:
        response.counted_in_total = response.id in counted_question_ids


def _is_attempted(feedback: IndividualFeedback) -> bool:
    from topper_maker.evaluation.schemas.feedback import AnswerStatus

    return feedback.answer_status not in (
        AnswerStatus.UNATTEMPTED,
        AnswerStatus.BLANK_PAGE_SUSPECTED,
    )
