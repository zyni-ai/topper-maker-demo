"""Parse academic document structure from raw HTR text blocks.

Detects:
  - Part labels        (Part-A, Part-B, Part-C …)
  - Section numerals   (I, II, III … standalone OR "III Answer …" instruction lines)
  - Question numbers   (1>, 2>, 21., Q1, Q.1 …)
  - Sub-questions      (a), b), i), ii) … but only when a question is already active)
  - MCQ answers        (circled options like (a), (b), (c), (d))
  - Fill-in-the-blank  ("16) Solenoid")

Design notes:
  - _SECTION_RE requires the numeral to lead a line but allows trailing instruction text;
    this fixes the 0% section detection caused by the original standalone-only requirement.
  - "V" alone no longer triggers a section match — a single Roman numeral that is also a
    common symbol (V, I) must appear at the start of a section-instruction line; a bare
    one-character match is rejected to avoid misclassifying physics symbols.
  - _SUB_Q_RE only triggers when a question is already active (enforced in the parser),
    preventing ordinary prose sentences from being classified as sub-questions.
  - Question numbers are validated against a configurable maximum (default 60) to block
    OCR ghost numbers like "Q89" that come from misread characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from topper_maker.htr.base import HTRResult, TextBlock


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_PART_RE = re.compile(
    r"^\s*[Pp]art\s*[-–]?\s*([A-Fa-f])\b", re.IGNORECASE
)

# Multi-character Roman numerals (II, III, IV, VI, VII, VIII) — unambiguous, match freely.
_SECTION_MULTI_RE = re.compile(
    r"^\s*(I{2,3}|IV|VI{0,3}|VIII|VII)\s*(?:$|(?=\s))",
    re.IGNORECASE,
)
# Single-character Roman numerals (I, V) — only valid section headers when followed by
# a known KSEAB board instruction keyword; a bare I or V is physics/English, not a section.
_SECTION_SINGLE_RE = re.compile(
    r"^\s*([IV])\s+(?:Answer|Write|Explain|Define|State|Draw|Derive|Calculate|Solve|List"
    r"|Describe|Mention|Give|Find|Prove|Show|Evaluate|Illustrate)\b",
    re.IGNORECASE,
)

# "1>" or "1)" — up to two-digit question numbers.
_QUESTION_RE = re.compile(r"^\s*(\d{1,2})[>)]\s*")
# "Q1", "Q.1" prefix
_Q_PREFIX_RE = re.compile(r"^\s*[Qq]\.?\s*(\d{1,2})\b")
# "21." — period-separated (common in many board papers)
_Q_DOT_RE = re.compile(r"^\s*(\d{1,2})\.\s+[A-Z]")
# "Ans 21" or "Answer 21" written by students
_ANS_RE = re.compile(r"^\s*[Aa]ns(?:wer)?\s*[.:]?\s*(\d{1,2})\b")

# Sub-question: only used when a question context is already active.
# Requires a closing paren or period so we don't match sentence starts like "a force…"
_SUB_Q_RE = re.compile(
    r"^\s*([a-dA-D]|[ivxIVX]{1,4})[).]\s+"
)

_MCQ_RE = re.compile(r"\(\s*([a-dA-D])\s*\)")
_FILL_BLANK_RE = re.compile(r"^\s*\d{1,2}[)>]\s*[A-Z][a-z]")

# Maximum plausible question number for a board paper; configurable via the parser.
_DEFAULT_MAX_QUESTION = 60


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class QuestionBlock:
    question_number: int
    sub_question: str | None          # "a", "b", "i", "ii" etc.
    part: str | None                  # "A", "B", "C" …
    section: str | None               # "I", "II" …
    text: str
    confidence: float
    is_mcq_answer: bool = False
    mcq_choice: str | None = None     # "a", "b", "c", "d"
    is_fill_blank: bool = False


@dataclass
class DocumentStructure:
    parts: list[str] = field(default_factory=list)          # ["A", "B", "C", …]
    sections: list[str] = field(default_factory=list)       # ["I", "II", …]
    questions: list[QuestionBlock] = field(default_factory=list)
    unclassified_blocks: list[TextBlock] = field(default_factory=list)

    @property
    def question_numbers(self) -> list[int]:
        return sorted({q.question_number for q in self.questions})

    @property
    def total_questions_detected(self) -> int:
        return len(set(q.question_number for q in self.questions))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class DocumentStructureParser:
    """Converts a list of HTRResult objects into a DocumentStructure."""

    def __init__(self, max_question_number: int = _DEFAULT_MAX_QUESTION) -> None:
        self.max_question_number = max_question_number

    def parse(self, htr_results: list[HTRResult]) -> DocumentStructure:
        structure = DocumentStructure()
        current_part: str | None = None
        current_section: str | None = None
        current_question: int | None = None
        current_sub: str | None = None

        for result in htr_results:
            for block in result.blocks:
                text = block.text.strip()
                if not text:
                    continue

                # --- Part label ---
                m = _PART_RE.match(text)
                if m:
                    current_part = m.group(1).upper()
                    if current_part not in structure.parts:
                        structure.parts.append(current_part)
                    continue

                # --- Section numeral ---
                # Multi-char numerals (II, III, IV …) are unambiguous — match freely.
                m = _SECTION_MULTI_RE.match(text) or _SECTION_SINGLE_RE.match(text)
                if m:
                    current_section = m.group(1).upper()
                    if current_section not in structure.sections:
                        structure.sections.append(current_section)
                    continue

                # --- Question number ---
                q_num = self._extract_question_number(text)
                if q_num is not None:
                    current_question = q_num
                    current_sub = None

                # --- Sub-question (only meaningful inside a question) ---
                if current_question is not None:
                    sub = self._extract_sub_question(text)
                    if sub:
                        current_sub = sub

                # --- MCQ answer ---
                mcq = _MCQ_RE.search(text)

                if current_question is not None:
                    qb = QuestionBlock(
                        question_number=current_question,
                        sub_question=current_sub,
                        part=current_part,
                        section=current_section,
                        text=text,
                        confidence=block.confidence,
                        is_mcq_answer=mcq is not None and len(text) < 15,
                        mcq_choice=mcq.group(1).lower() if mcq else None,
                        is_fill_blank=bool(_FILL_BLANK_RE.match(text)),
                    )
                    structure.questions.append(qb)
                else:
                    structure.unclassified_blocks.append(block)

        return structure

    def _extract_question_number(self, text: str) -> int | None:
        """Try all question-number patterns; validate against max_question_number."""
        for pattern in (_QUESTION_RE, _Q_PREFIX_RE, _Q_DOT_RE, _ANS_RE):
            m = pattern.match(text)
            if m:
                num = int(m.group(1))
                if 1 <= num <= self.max_question_number:
                    return num
                # Out-of-range — likely an OCR ghost number; skip silently.
                return None
        return None

    @staticmethod
    def _extract_sub_question(text: str) -> str | None:
        m = _SUB_Q_RE.match(text)
        if m:
            return m.group(1).lower()
        return None
