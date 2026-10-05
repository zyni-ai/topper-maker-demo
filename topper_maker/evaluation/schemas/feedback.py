"""Output schemas: per-question feedback, rubric audit trail, and section summaries."""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from topper_maker.evaluation.schemas.supervision import QuestionSupervision


class AnswerStatus(str, Enum):
    """Distinguishes *why* a question scored zero — critical for fair review.

    A genuinely unattempted question and a question whose page was blank due to a
    scan error look identical in the marks, but must be surfaced differently to the
    reviewer (decision from issue #9).
    """

    ANSWERED = "answered"
    UNATTEMPTED = "unattempted"               # no answer found, page extracted fine
    BLANK_PAGE_SUSPECTED = "blank_page_suspected"  # answer missing AND its page was blank
    EVALUATION_ERROR = "evaluation_error"     # LLM/processing failure for this question


class RubricAward(BaseModel):
    """Per-rubric-point award — the audit trail behind a question's score."""

    key: str = Field(..., description="Rubric point key.")
    description: str = Field(..., description="Rubric point description.")
    marks_possible: float = Field(..., description="Marks this point could award.")
    marks_awarded: float = Field(..., description="Marks actually awarded (after rules).")
    awarded: bool = Field(..., description="True if any marks were awarded for this point.")
    rationale: str = Field("", description="Why the point was/was not awarded.")
    forced_zero_by_dependency: bool = Field(
        False,
        description="True if this point was zeroed because a prerequisite point failed "
        "(cascading rule), regardless of the student's downstream work.",
    )


class IndividualFeedback(BaseModel):
    """Evaluation result for a single question."""

    id: int = Field(..., description="Question ID.")
    answer_status: AnswerStatus = Field(
        AnswerStatus.ANSWERED, description="Whether/why the question was answered."
    )

    user_answer: Optional[str] = Field(None, description="Transcribed student answer.")
    user_answer_format: Literal["md", "latex"] = Field(
        "md", description="Format of user_answer."
    )
    user_answer_image_codes: Optional[List[str]] = Field(
        None, description="UUIDs of student answer images uploaded to S3."
    )

    is_correct: bool = Field(False, description="True only if full marks were awarded.")
    score: float = Field(0.0, description="Marks awarded (0 .. max_score).")
    max_score: float = Field(..., description="Maximum marks for the question.")

    feedback: str = Field("", description="Explanation of the score for the student.")
    feedback_format: Literal["md", "latex"] = Field("md", description="Format of feedback.")

    rubric_breakdown: Optional[List[RubricAward]] = Field(
        None, description="Per-point awards when the question was rubric-marked."
    )

    counted_in_total: bool = Field(
        True,
        description="False if excluded from the total by optional-section rules.",
    )
    needs_review: bool = Field(
        False, description="True if this question should be checked by a human."
    )
    low_trust_short_answer: bool = Field(
        False,
        description="True if this is a short objective answer scored below full marks — a "
        "likely HTR misread that should be eyeballed before finalising (issue #37).",
    )
    diagram_expected_missing: bool = Field(
        False,
        description="True if the rubric expects a diagram for this question but no student "
        "figure was extracted/bound — the diagram point could not be judged from an image, "
        "so the question is routed for human review (issue #124).",
    )
    has_evaluation_error: bool = Field(
        False, description="True if evaluation failed for this question."
    )
    supervision: Optional[QuestionSupervision] = Field(
        None,
        description="Supervisor's arbitration record for this question, when a supervisor was used.",
    )


class SectionSummary(BaseModel):
    """Marks breakdown for one section (mandatory or optional)."""

    section_id: str = Field(..., description="Section identifier.")
    section_type: Literal["mandatory", "optional"] = Field(..., description="Section type.")
    questions_answered: int = Field(..., description="Questions answered in this section.")
    questions_required: int = Field(..., description="Questions required to be counted.")
    questions_counted: List[int] = Field(
        default_factory=list, description="Question IDs whose scores were counted."
    )
    questions_excluded: List[int] = Field(
        default_factory=list,
        description="Question IDs answered but not counted (optional-section overflow).",
    )
    marks_scored: float = Field(..., description="Marks scored from counted questions.")
    marks_possible: float = Field(..., description="Max marks from counted questions.")
