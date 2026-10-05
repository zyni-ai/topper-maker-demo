"""Schemas for the paper/rubric compiler.

Two layers:

1. **Raw extraction schemas** (``_Paper*`` / ``_Key*``) — what the vision LLM returns
   when it reads the question-paper PDF and the answer-key PDF. These are deliberately
   permissive (most fields optional) so a partial read degrades to warnings rather than
   a hard validation failure. They are an internal contract, not part of the public API.

2. **Compiled output** (:class:`CompiledPaper`) — the reviewable artifact. Its
   ``questions`` list is made of the production :class:`QuestionItem` objects the
   evaluation engine already consumes, plus the ``optional_sections`` map for
   ``EvaluationRequest``. A human verifies this JSON before any student is graded.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from topper_maker.evaluation.schemas.question import QuestionItem

# ---------------------------------------------------------------------------
# Raw extraction — question paper
# ---------------------------------------------------------------------------


class RawPaperQuestion(BaseModel):
    """One question as read from the question-paper body (not the instructions block)."""

    number: int = Field(..., description="The question number printed on the paper.")
    statement: str = Field(..., description="Full question text, verbatim.")
    max_marks: Optional[float] = Field(
        None, description="Marks for this question if printed next to it (e.g. '[4 marks]')."
    )
    section_label: Optional[str] = Field(
        None, description="Section/part letter this question sits under, e.g. 'A', 'B'."
    )
    section_title: Optional[str] = Field(
        None, description="Human title of the section, e.g. 'Numerical / Diagram'."
    )
    question_type: Optional[str] = Field(
        None,
        description="One of: mcq, match, fill_blank, very_short, short, numerical, "
        "long, derivation. Best-effort classification.",
    )
    has_figure: bool = Field(False, description="True if the question itself includes a figure.")
    choice_group: Optional[str] = Field(
        None,
        description="A label shared by questions that are alternatives of one another "
        "(an 'OR' / internal choice). Questions with the same non-null choice_group form "
        "an 'answer any one' set. Null for compulsory questions.",
    )


class PaperExtraction(BaseModel):
    """Structured read of the whole question paper."""

    exam_title: Optional[str] = Field(None, description="Title printed on the paper.")
    subject: Optional[str] = Field(None, description="Subject, e.g. 'Physics'.")
    class_type: Optional[str] = Field(None, description="Class/grade, e.g. 'XI (Science)'.")
    total_marks: Optional[float] = Field(None, description="Maximum marks for the paper.")
    questions: List[RawPaperQuestion] = Field(default_factory=list)
    section_optional_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Maps a section_label to the number of questions the student must answer "
        "from that section (e.g. {'B': 5} means 'answer any 5 from section B'). "
        "Only set for sections with an explicit 'answer any N' instruction; leave empty "
        "for fully compulsory sections.",
    )


# ---------------------------------------------------------------------------
# Raw extraction — answer key
# ---------------------------------------------------------------------------


class RawRubricPoint(BaseModel):
    """A marking value-point as read from the answer key."""

    key: str = Field(..., description="Short stable identifier, e.g. 'formula', 'diagram'.")
    description: str = Field(..., description="What the student must demonstrate.")
    marks: float = Field(..., description="Marks for this point (must be > 0 to be usable).")
    keywords: List[str] = Field(default_factory=list, description="Acceptable key terms.")
    required: bool = Field(
        True,
        description="False for an alternative/bonus point that need not be present (and is "
        "excluded from the marks-sum sanity check).",
    )
    allow_partial: bool = Field(
        False, description="True if the key allows a half mark for a partially-correct point."
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Keys of points required before this one (numericals: substitution "
        "depends on formula, answer depends on substitution).",
    )
    is_diagram: bool = Field(False, description="True if the point is judged against a figure.")


class RawKeyQuestion(BaseModel):
    """One question's answer/rubric as read from the answer key."""

    number: int = Field(..., description="Question number this answer/rubric belongs to.")
    answer_kind: str = Field(
        "rubric",
        description="'objective' for MCQ/match/fill-blank (holistic expected_answer marking); "
        "'rubric' for descriptive questions with a value-point split.",
    )
    expected_answer: Optional[str] = Field(
        None, description="Model answer text (used for objective questions)."
    )
    rubric_points: List[RawRubricPoint] = Field(
        default_factory=list, description="Value-point split (used for descriptive questions)."
    )
    max_marks: Optional[float] = Field(None, description="Max marks for the question if stated.")
    notes: Optional[str] = Field(
        None, description="Any per-question marking note from the key (e.g. award step-by-step)."
    )


class SectionDistribution(BaseModel):
    """A row of the answer key's mark-distribution table."""

    section_label: Optional[str] = Field(None, description="Section letter, e.g. 'A'.")
    question_range: Optional[str] = Field(None, description="e.g. 'Q1-Q10'.")
    marks_each: Optional[float] = Field(None, description="Marks per question in the section.")
    total: Optional[float] = Field(None, description="Section total marks.")


class KeyExtraction(BaseModel):
    """Structured read of the whole answer key / evaluation rubric."""

    section_distribution: List[SectionDistribution] = Field(default_factory=list)
    questions: List[RawKeyQuestion] = Field(default_factory=list)
    global_partial_rules: List[str] = Field(
        default_factory=list,
        description="General partial-marking guidelines that apply across questions.",
    )
    penalties: List[str] = Field(
        default_factory=list, description="Common-mistake deductions listed in the key."
    )
    teacher_notes: List[str] = Field(
        default_factory=list, description="Free-form examiner instructions from the key."
    )


# ---------------------------------------------------------------------------
# Compiled output — the reviewable artifact
# ---------------------------------------------------------------------------


class CompiledPaper(BaseModel):
    """The compiler's output: ready-to-use evaluation inputs plus an audit of how
    they were derived.

    ``questions`` + ``optional_sections`` drop straight into an ``EvaluationRequest``.
    ``marking_guidance``, ``penalties`` and ``warnings`` are carried for the human
    reviewer (and future deduction-rule work) but are not yet consumed by the scorer.
    """

    subject: Optional[str] = None
    class_type: Optional[str] = None
    exam_title: Optional[str] = None
    total_marks: Optional[float] = None

    questions: List[QuestionItem] = Field(
        default_factory=list, description="Compiled questions, usable as questions_list."
    )
    optional_sections: Dict[str, int] = Field(
        default_factory=dict,
        description="section_id → number of questions to count (best-N). Includes OR-choice "
        "groups, each with N=1.",
    )

    marking_guidance: List[str] = Field(
        default_factory=list, description="Global partial-marking rules + teacher notes from the key."
    )
    penalties: List[str] = Field(
        default_factory=list, description="Common-mistake deductions from the key."
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Reconciliation issues a human should resolve before grading.",
    )

    def summary(self) -> Dict[str, object]:
        """Compact, log-friendly overview."""
        rubric_qs = sum(1 for q in self.questions if q.uses_rubric)
        return {
            "subject": self.subject,
            "class_type": self.class_type,
            "total_marks": self.total_marks,
            "num_questions": len(self.questions),
            "rubric_marked": rubric_qs,
            "holistic_marked": len(self.questions) - rubric_qs,
            "optional_sections": self.optional_sections,
            "num_warnings": len(self.warnings),
        }
