"""Input schemas describing a question and its marking rubric.

A question may be marked one of two ways:

1. **Rubric marking** (preferred, board-style) — ``rubric_points`` is supplied and
   the evaluator awards marks point-by-point. This mirrors KSEAB value-point
   marking schemes.
2. **Expected-answer marking** (fallback) — only ``expected_answer`` is supplied
   and the evaluator compares the student answer holistically.

When ``rubric_points`` is present it takes precedence over ``expected_answer``.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class RubricPoint(BaseModel):
    """A single value point in a marking scheme.

    Design notes (decisions captured from issue #5):

    - **Partial marks**: a point is all-or-nothing (``marks`` or ``0``) *unless*
      ``allow_partial`` is set, in which case the evaluator may award a half mark.
      This keeps the default behaviour aligned with board convention while
      allowing rubrics that explicitly permit partial credit.
    - **Cascading dependencies** (numericals): ``depends_on`` lists the keys of
      points that must be awarded before this one can be. If a prerequisite is
      not awarded (e.g. wrong formula), this point is forced to zero regardless of
      what the student wrote downstream. Enforced deterministically in code, not
      left to the model.
    - **Diagram points**: set ``is_diagram=True`` for points evaluated against a
      drawn figure. Per the board convention, when a rubric describes the expected
      diagram, the student's diagram (or its description) is evaluated against that
      text via the vision model.
    """

    key: str = Field(
        ...,
        description="Short stable identifier for this point, e.g. 'formula', 'substitution'. "
        "Used by depends_on and in the audit breakdown.",
    )
    description: str = Field(
        ..., description="What the student must demonstrate to earn this point."
    )
    marks: float = Field(
        ..., gt=0, description="Marks awarded when this point is satisfied."
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Acceptable phrasings / key terms that signal the point is met. "
        "Guidance for the evaluator, not a strict string match.",
    )
    required: bool = Field(
        True,
        description="False marks the point as an alternative/bonus that need not be present.",
    )
    allow_partial: bool = Field(
        False,
        description="If True, a half mark may be awarded for a partially-correct point. "
        "If False (default), the point is all-or-nothing.",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Keys of rubric points that must be awarded before this one is eligible. "
        "Models board cascading rules (e.g. no substitution mark if the formula is wrong).",
    )
    is_diagram: bool = Field(
        False,
        description="True when this point is evaluated against a diagram/figure rather than text.",
    )


class QuestionItem(BaseModel):
    """A question, its expected answer and/or rubric, and any reference images."""

    id: int = Field(..., description="Unique identifier for the question.")
    question_number: int = Field(..., description="Display number on the paper.")
    question_statement: str = Field(..., description="The question text.")
    max_score: float = Field(..., gt=0, description="Maximum marks for the question.")

    # Optional reference material -------------------------------------------------
    expected_answer: Optional[str] = Field(
        None,
        description="Model answer text. Used when rubric_points is not supplied.",
    )
    rubric_points: Optional[List[RubricPoint]] = Field(
        None,
        description="Marking scheme as value points. When present, overrides expected_answer.",
    )

    question_image: bool = Field(False, description="True if the question has a figure.")
    question_image_url: Optional[List[str]] = Field(
        None, description="S3 URL(s) of the question figure(s), if any."
    )
    answer_image: bool = Field(
        False, description="True if the expected answer includes a figure."
    )
    expected_answer_image_url: Optional[List[str]] = Field(
        None, description="S3 URL(s) of the expected-answer figure(s), if any."
    )

    # Metadata --------------------------------------------------------------------
    topic: Optional[str] = Field(None, description="Topic tag.")
    section_id: Optional[str] = Field(
        None,
        description="Section/part this question belongs to (e.g. 'part_b'). "
        "Drives optional-section grouping in the marks aggregator.",
    )

    @model_validator(mode="after")
    def _check_has_reference(self) -> "QuestionItem":
        """A question needs either an expected answer or a rubric to be markable."""
        if not self.expected_answer and not self.rubric_points:
            raise ValueError(
                f"Question {self.id} has neither expected_answer nor rubric_points; "
                "it cannot be evaluated."
            )
        return self

    @property
    def uses_rubric(self) -> bool:
        return bool(self.rubric_points)

    @property
    def rubric_marks_total(self) -> float:
        """Sum of marks across required rubric points (sanity-check against max_score)."""
        if not self.rubric_points:
            return 0.0
        return sum(p.marks for p in self.rubric_points if p.required)
