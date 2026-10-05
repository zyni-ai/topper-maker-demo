"""Top-level request/response state and review-routing models."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from topper_maker.evaluation.schemas.feedback import IndividualFeedback, SectionSummary
from topper_maker.evaluation.schemas.question import QuestionItem
from topper_maker.evaluation.schemas.supervision import SupervisionRecord


class ReviewReason(str, Enum):
    """Why a booklet (or page) was routed to a human reviewer.

    Routing policy (decisions from issue #9):

    - ``LOW_HTR_CONFIDENCE``  → route the WHOLE booklet to review.
    - ``PARTIAL_EXTRACTION``  → route the WHOLE booklet to review.
    - ``ROTATED_PAGE`` → flag the affected pages; booklet review.
    - ``BLANK_PAGE_SCAN_ERROR`` → surface separately from genuine non-attempts.
    - ``EVALUATION_ERROR`` / ``MODERATION_ERROR`` → operational failures.
    """

    LOW_HTR_CONFIDENCE = "low_htr_confidence"
    PARTIAL_EXTRACTION = "partial_extraction"
    ROTATED_PAGE = "rotated_page"
    BLANK_PAGE_SCAN_ERROR = "blank_page_scan_error"
    EVALUATION_ERROR = "evaluation_error"
    MODERATION_ERROR = "moderation_error"
    MAPPING_ERROR = "mapping_error"
    UNVERIFIED_SHORT_ANSWER = "unverified_short_answer"
    UNMAPPED_ANSWER = "unmapped_answer"
    # A question's rubric expects a diagram, but no student figure was extracted or
    # bound to it — the diagram point cannot be judged from an absent image, so we
    # fail closed to human review rather than scoring it on nothing (issue #124).
    DIAGRAM_EXPECTED_NOT_FOUND = "diagram_expected_not_found"
    MODEL_DISAGREEMENT = "model_disagreement"      # candidate models disagreed significantly
    SUPERVISOR_UNRESOLVED = "supervisor_unresolved"  # supervisor couldn't resolve disagreement


class PageQuality(BaseModel):
    """Per-page extraction quality signals, surfaced for observability and review."""

    page_number: int
    is_blank: bool = False
    orientation: str = "upright"
    mean_confidence: float = 1.0
    min_confidence: float = 1.0
    num_text_blocks: int = 0
    num_diagram_blocks: int = 0
    truncated: bool = False
    warnings: List[str] = Field(default_factory=list)


class EvaluationRequest(BaseModel):
    """Everything needed to evaluate one student's answer sheet."""

    user_id: str = Field(..., description="Student/user identifier.")
    user_test_id: str = Field(..., description="Unique attempt identifier (used in S3 paths).")
    skill: str = Field(..., description="Subject/skill, e.g. 'physics'.")
    class_type: str = Field(..., description="Grade, e.g. '2nd_puc'.")
    answer_sheet_url: str = Field(..., description="URL of the scanned answer-sheet PDF.")
    questions_list: List[QuestionItem] = Field(..., description="Questions with rubrics.")
    optional_sections: Optional[Dict[str, int]] = Field(
        None,
        description="Map of section_id → number of questions to count (top-N by score). "
        "e.g. {'part_b': 5} counts the best 5 answered questions in part_b.",
    )
    request_id: Optional[str] = Field(
        None, description="Caller-supplied correlation id for tracing (defaults to user_test_id)."
    )

    @model_validator(mode="after")
    def _check_unique_question_ids(self) -> "EvaluationRequest":
        ids = [q.id for q in self.questions_list]
        seen: set = set()
        dupes = [i for i in ids if i in seen or seen.add(i)]  # type: ignore[func-returns-value]
        if dupes:
            raise ValueError(
                f"questions_list contains duplicate question id(s): {sorted(set(dupes))}. "
                "Each question must have a unique id."
            )
        return self


class EvaluationResponse(BaseModel):
    """Evaluation result for one student's answer sheet."""

    is_valid: bool = Field(
        True, description="False if input validation or moderation rejected the sheet."
    )
    rejection_reason: Optional[str] = Field(
        None, description="Human-readable reason when is_valid is False."
    )
    moderation_categories: Optional[List[str]] = Field(
        None, description="Flagged moderation categories, if any."
    )
    moderation_scores: Optional[Dict[str, float]] = Field(
        None, description="Scores for flagged moderation categories."
    )

    responses: List[IndividualFeedback] = Field(
        default_factory=list, description="Per-question feedback."
    )
    total_marks: float = Field(0.0, description="Total marks after optional-section rules.")
    max_marks: float = Field(0.0, description="Maximum marks after optional-section rules.")
    percentage: float = Field(0.0, description="(total_marks / max_marks) * 100.")
    section_summaries: Optional[List[SectionSummary]] = Field(
        None, description="Per-section marks breakdown."
    )

    # Review routing & observability ---------------------------------------------
    needs_human_review: bool = Field(
        False, description="True if the booklet should be checked by a human."
    )
    review_reasons: List[ReviewReason] = Field(
        default_factory=list, description="Why review was triggered."
    )
    flagged_pages: List[int] = Field(
        default_factory=list, description="Page numbers with quality issues."
    )
    page_quality: List[PageQuality] = Field(
        default_factory=list, description="Per-page extraction quality signals."
    )
    has_processing_errors: bool = Field(
        False, description="True if any question hit an evaluation error."
    )
    stage_timings_ms: Dict[str, float] = Field(
        default_factory=dict, description="Wall-clock duration per pipeline stage (ms)."
    )
    request_id: Optional[str] = Field(None, description="Correlation id echoed back.")
    analysis_url: Optional[str] = Field(
        None, description="S3 URL of the persisted extraction+moderation analysis, if uploaded."
    )

    # Supervisor arbitration (populated when >1 eval_model + supervisor_model is set) ----------
    supervision: Optional[SupervisionRecord] = Field(
        None,
        description="Supervisor arbitration audit trail, when supervision was used.",
    )
    candidate_responses: Optional[Dict[str, List[IndividualFeedback]]] = Field(
        None,
        description="Per-candidate feedback lists keyed by model string. Populated when "
        "supervision was used so the UI can show candidate tabs alongside the supervisor result.",
    )

    # Cost / token accounting -----------------------------------------------------
    cost_usd: Optional[float] = Field(
        None,
        description="Total LLM cost (USD) of this evaluation across all stages, as "
        "reported by OpenRouter. None if the provider returned no cost figures.",
    )
    usage: Optional[Dict[str, object]] = Field(
        None,
        description="Token/cost accounting summary (num_calls, total_tokens, by_model …).",
    )
