"""Deterministic reconciliation of a question paper and its answer key.

This is the pure, testable core of the compiler: given the two raw vision-LLM reads,
it joins them by question number and produces production :class:`QuestionItem`s plus
the ``optional_sections`` map, surfacing every ambiguity as a warning rather than
guessing silently.

Design rules (decided with the user, 2026-06-11):

- **The question bodies + the answer key's mark distribution are authoritative.** A
  paper's "General Instructions" block can contradict its own body (our sample does);
  we never read structure from instructions, only from the actual numbered questions.
- **Internal "OR" choices are choice groups.** Questions sharing a ``choice_group`` are
  "answer any one" → their own ``section_id`` with ``optional_sections[...] = 1``. The
  marks engine already counts the best-scoring attempt and drops the rest (board rule).
- **Marks arithmetic stays in code.** We validate that rubric point marks sum to the
  question max and warn on mismatch; we never let the LLM's totals through unchecked.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional

from topper_maker.evaluation.authoring.schemas import (
    CompiledPaper,
    KeyExtraction,
    PaperExtraction,
    RawKeyQuestion,
    RawPaperQuestion,
    RawRubricPoint,
)
from topper_maker.evaluation.schemas.question import QuestionItem, RubricPoint

_MARKS_TOLERANCE = 0.5


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "x"


def _base_section_id(q: RawPaperQuestion) -> str:
    if q.section_label:
        return f"section_{_slug(q.section_label)}"
    if q.section_title:
        return f"section_{_slug(q.section_title)}"
    return "section_unknown"


def _section_id_for(q: RawPaperQuestion, choice_section_ids: Dict[str, str]) -> str:
    """A choice-group question gets its own dedicated section so best-N never spills
    onto the compulsory questions in the same printed section."""
    if q.choice_group and q.choice_group in choice_section_ids:
        return choice_section_ids[q.choice_group]
    return _base_section_id(q)


def _marks_each_from_distribution(key: KeyExtraction) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in key.section_distribution:
        if row.section_label and row.marks_each:
            out[_slug(row.section_label)] = row.marks_each
    return out


def _convert_rubric_points(
    raw_points: List[RawRubricPoint], warnings: List[str], qnum: int
) -> List[RubricPoint]:
    points: List[RubricPoint] = []
    seen_keys: set[str] = set()
    key_renames: Dict[str, str] = {}  # original_key → renamed_key (for depends_on fixup)
    for rp in raw_points:
        if rp.marks <= 0:
            warnings.append(
                f"Q{qnum}: rubric point '{rp.key}' has non-positive marks ({rp.marks}); dropped."
            )
            key_renames[rp.key] = ""  # mark as dropped so depends_on refs are removed
            continue
        key = rp.key
        if key in seen_keys:  # keys must be unique within a question (depends_on references them)
            new_key = f"{key}_{len(points)}"
            warnings.append(
                f"Q{qnum}: duplicate rubric key '{key}'; renamed to '{new_key}' — "
                "update any depends_on references in the answer key."
            )
            key_renames[key] = new_key
            key = new_key
        seen_keys.add(key)
        points.append(
            RubricPoint(
                key=key,
                description=rp.description,
                marks=rp.marks,
                keywords=rp.keywords,
                required=rp.required,
                allow_partial=rp.allow_partial,
                depends_on=[d for d in rp.depends_on if d],
                is_diagram=rp.is_diagram,
            )
        )
    # Fix up depends_on references using the rename map.
    if key_renames:
        valid_keys = {p.key for p in points}
        for p in points:
            fixed = []
            for dep in p.depends_on:
                resolved = key_renames.get(dep, dep)
                if resolved == "":
                    warnings.append(
                        f"Q{qnum}: depends_on references dropped key '{dep}' in point '{p.key}'; "
                        "dependency removed."
                    )
                elif resolved not in valid_keys:
                    warnings.append(
                        f"Q{qnum}: depends_on references unknown key '{dep}' in point '{p.key}'; "
                        "dependency removed."
                    )
                else:
                    fixed.append(resolved)
            p.depends_on = fixed
    return points


def _resolve_max_score(
    paper_q: RawPaperQuestion,
    key_q: Optional[RawKeyQuestion],
    rubric_points: List[RubricPoint],
    marks_each: Dict[str, float],
) -> Optional[float]:
    if paper_q.max_marks and paper_q.max_marks > 0:
        return paper_q.max_marks
    if key_q and key_q.max_marks and key_q.max_marks > 0:
        return key_q.max_marks
    if rubric_points:
        return sum(p.marks for p in rubric_points if p.required)
    if paper_q.section_label:
        each = marks_each.get(_slug(paper_q.section_label))
        if each:
            return each
    return None


def _build_question(
    paper_q: RawPaperQuestion,
    key_q: Optional[RawKeyQuestion],
    qid: int,
    section_id: str,
    marks_each: Dict[str, float],
    warnings: List[str],
) -> QuestionItem:
    rubric_points: List[RubricPoint] = []
    expected_answer: Optional[str] = None
    is_objective = key_q is not None and key_q.answer_kind.lower() == "objective"

    if key_q is None:
        warnings.append(
            f"Q{paper_q.number}: no matching entry in the answer key; "
            "emitted with a placeholder answer — supply a rubric or expected answer before grading."
        )
        expected_answer = "(No answer key was found for this question — needs manual entry.)"
    elif key_q.answer_kind.lower() == "objective" or not key_q.rubric_points:
        if key_q.answer_kind.lower() not in ("objective",) and not key_q.rubric_points:
            warnings.append(
                f"Q{paper_q.number}: answer_kind is '{key_q.answer_kind}' but no rubric points "
                "were extracted; falling back to expected-answer holistic marking."
            )
        expected_answer = key_q.expected_answer or key_q.notes
        if not expected_answer:
            warnings.append(
                f"Q{paper_q.number}: answer key has no expected answer text; placeholder used."
            )
            expected_answer = "(Answer key text missing — needs manual entry.)"
    else:
        rubric_points = _convert_rubric_points(key_q.rubric_points, warnings, paper_q.number)
        if not rubric_points:
            expected_answer = key_q.expected_answer or "(Rubric unusable — needs manual entry.)"
            warnings.append(
                f"Q{paper_q.number}: rubric points could not be used; fell back to expected-answer marking."
            )

    max_score = _resolve_max_score(paper_q, key_q, rubric_points, marks_each)
    if not max_score or max_score <= 0:
        warnings.append(
            f"Q{paper_q.number}: could not determine max marks; defaulted to 1.0 — please set."
        )
        max_score = 1.0

    # A holistic objective worth more than one mark is the smell of a match /
    # multi-blank question that should have been split into per-component rubric
    # points (issue #36): holistic marking gives one all-or-nothing decision instead
    # of one judgement per pair/blank, so a 3-of-5 answer cannot be scored 3.
    if is_objective and max_score > 1:
        warnings.append(
            f"Q{paper_q.number}: marked holistically (single expected answer) but worth "
            f"{max_score} marks. If this is a match / multi-blank question, split it into "
            "one rubric point per pair/blank so it can be scored per-component."
        )

    # Marks sanity check for rubric questions — only *required* points count toward the
    # expected total; alternative/bonus points (required=False) may legitimately push the
    # raw sum above max_score.
    if rubric_points:
        rubric_total = sum(p.marks for p in rubric_points if p.required)
        if abs(rubric_total - max_score) > _MARKS_TOLERANCE:
            warnings.append(
                f"Q{paper_q.number}: required rubric points sum to {rubric_total} but max marks "
                f"is {max_score}; verify the marking split."
            )

    return QuestionItem(
        id=qid,
        question_number=paper_q.number,
        question_statement=paper_q.statement,
        max_score=max_score,
        expected_answer=expected_answer,
        rubric_points=rubric_points or None,
        question_image=paper_q.has_figure,
        section_id=section_id,
        topic=paper_q.section_title,
    )


def reconcile(paper: PaperExtraction, key: KeyExtraction) -> CompiledPaper:
    """Join the paper and key reads into a reviewable :class:`CompiledPaper`."""
    warnings: List[str] = []
    key_by_num: Dict[int, RawKeyQuestion] = {}
    for kq in key.questions:
        if kq.number in key_by_num:
            warnings.append(f"Answer key has duplicate entries for Q{kq.number}; used the first.")
            continue
        key_by_num[kq.number] = kq

    marks_each = _marks_each_from_distribution(key)

    # Build a dedicated section_id for each OR-choice group.
    groups: Dict[str, List[RawPaperQuestion]] = defaultdict(list)
    for pq in paper.questions:
        if pq.choice_group:
            groups[pq.choice_group].append(pq)
    choice_section_ids: Dict[str, str] = {}
    optional_sections: Dict[str, int] = {}
    for group_label, members in groups.items():
        if len(members) < 2:
            if len(members) == 1:
                warnings.append(
                    f"Choice group '{group_label}' has only one member "
                    f"(Q{members[0].number}); treated as a compulsory question."
                )
            continue
        base = _base_section_id(members[0])
        section_id = f"{base}_choice_{_slug(group_label)}"
        choice_section_ids[group_label] = section_id
        optional_sections[section_id] = 1  # answer any ONE of the alternatives
        numbers = [m.number for m in members]
        if len(set(numbers)) != len(numbers):
            warnings.append(
                f"OR-choice group '{group_label}' reuses question number(s) {numbers}; "
                "the student's attempt cannot be auto-distinguished by number — verify mapping."
            )

    # "Answer any N" sections: add optional_sections entries for whole-section optionals (#48).
    for section_label, count in (paper.section_optional_counts or {}).items():
        section_id = f"section_{_slug(section_label)}"
        if count <= 0:
            warnings.append(
                f"Section '{section_label}': section_optional_counts value {count} is ≤ 0; ignored."
            )
            continue
        if section_id in optional_sections:
            warnings.append(
                f"Section '{section_label}': section_optional_counts N={count} conflicts with "
                f"an existing optional_sections entry (N={optional_sections[section_id]}); "
                "the existing entry was kept — verify the paper structure."
            )
        else:
            optional_sections[section_id] = count

    # Build a QuestionItem per paper question (the authoritative question set).
    questions: List[QuestionItem] = []
    used_numbers: set[int] = set()
    for idx, pq in enumerate(paper.questions, start=1):
        section_id = _section_id_for(pq, choice_section_ids)
        questions.append(
            _build_question(
                paper_q=pq,
                key_q=key_by_num.get(pq.number),
                qid=idx,
                section_id=section_id,
                marks_each=marks_each,
                warnings=warnings,
            )
        )
        used_numbers.add(pq.number)

    # Answer-key entries with no matching question in the paper.
    for num in sorted(set(key_by_num) - used_numbers):
        warnings.append(
            f"Answer key has Q{num} but the question paper has no such question; ignored."
        )

    # Total marks cross-check (best-N choice groups count one member each).
    counted_max = _expected_total(questions, optional_sections)
    if paper.total_marks and abs(counted_max - paper.total_marks) > _MARKS_TOLERANCE:
        warnings.append(
            f"Compiled max marks ({counted_max}) differ from the paper's stated total "
            f"({paper.total_marks}); verify marks and choice groups."
        )

    return CompiledPaper(
        subject=paper.subject,
        class_type=paper.class_type,
        exam_title=paper.exam_title,
        total_marks=paper.total_marks,
        questions=questions,
        optional_sections=optional_sections,
        marking_guidance=[*key.global_partial_rules, *key.teacher_notes],
        penalties=key.penalties,
        warnings=warnings,
    )


def _expected_total(questions: List[QuestionItem], optional_sections: Dict[str, int]) -> float:
    """Upper-bound estimate of the marks the paper can yield, counting the N highest-max
    members of each choice/optional section.

    This is only used to drive a cross-check warning. It is an approximation of
    ``marks_aggregator.compute_total_marks`` (which ranks *answered* questions by *score*,
    not by max_score, and depends on the student) — exact for 1-of-N choices with equal
    marks, an upper bound otherwise. Do not treat it as the authoritative maximum."""
    by_section: Dict[str, List[float]] = defaultdict(list)
    for q in questions:
        by_section[q.section_id or "__none__"].append(q.max_score)
    total = 0.0
    for section_id, maxes in by_section.items():
        if section_id in optional_sections:
            n = optional_sections[section_id]
            total += sum(sorted(maxes, reverse=True)[:n])
        else:
            total += sum(maxes)
    return round(total, 2)
