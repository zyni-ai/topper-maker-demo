"""Pure, testable metrics for validating HTR transcription quality (issue #10).

These functions take plain data (strings, dicts) so they can be unit-tested without
the API or a PDF. The runner in ``examples/validate_htr.py`` wires them to the live
:class:`HTRContentExtractor` output.

Metrics map to issue #10's success criteria:
- question-number recall (target >= 0.90),
- mean per-page HTR confidence (target >= 0.75),
- WER / CER against a hand-typed ground truth (when supplied),
- a failure-mode breakdown for the qualitative review.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set

# Patterns for a question number printed at the start of an answer. Kept conservative
# so prose digits (years, quantities) are unlikely to be mistaken for question numbers.
_Q_PATTERNS = [
    re.compile(r"(?:^|\n)\s*Q\.?\s*(\d{1,2})\b", re.IGNORECASE),   # "Q1", "Q. 1"
    re.compile(r"(?:^|\n)\s*(\d{1,2})\s*[.)>:\]]"),                # "1.", "1)", "1>", "1]"
    re.compile(r"\bAns(?:wer)?\s*(?:to)?\s*(?:Q\.?\s*)?(\d{1,2})\b", re.IGNORECASE),  # "Ans 1"
]


def detect_question_numbers(text: str, max_q: int = 60) -> Set[int]:
    """Heuristically recover the question numbers a transcript appears to answer.

    This is a measurement aid, not the production layout parser — it errs toward the
    common "N)", "N.", "QN", "Ans N" markers and ignores numbers outside ``1..max_q``.
    """
    found: Set[int] = set()
    for pat in _Q_PATTERNS:
        for m in pat.finditer(text or ""):
            n = int(m.group(1))
            if 1 <= n <= max_q:
                found.add(n)
    return found


def question_recall(found: Set[int], expected: Sequence[int]) -> Dict[str, object]:
    """Recall of expected question numbers among those detected in the transcript."""
    expected_set = {int(n) for n in expected}
    if not expected_set:
        return {"recall": None, "found": sorted(found), "missed": [], "extra": []}
    hit = found & expected_set
    return {
        "recall": round(len(hit) / len(expected_set), 4),
        "expected": sorted(expected_set),
        "found_expected": sorted(hit),
        "missed": sorted(expected_set - found),
        "extra": sorted(found - expected_set),
    }


def normalize_text(s: str) -> str:
    """Lowercase and collapse whitespace for order-sensitive WER/CER comparison."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def text_error_rates(hypothesis: str, reference: str) -> Dict[str, Optional[float]]:
    """WER and CER of ``hypothesis`` against a ground-truth ``reference`` (via jiwer).

    Returns ``None`` rates when no reference is supplied. Order-sensitive — the
    transcript's block order matters (a known WER inflator for grid/MCQ layouts).
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return {"wer": None, "cer": None}
    import jiwer

    return {
        "wer": round(jiwer.wer(ref, hyp), 4),
        "cer": round(jiwer.cer(ref, hyp), 4),
    }


def mean_confidence(page_quality: Sequence[Dict[str, object]]) -> Optional[float]:
    """Mean per-page HTR confidence over pages that actually produced text."""
    vals = [
        float(p["mean_confidence"])
        for p in page_quality
        if int(p.get("num_text_blocks", 0)) > 0 and not p.get("is_blank")
    ]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize_failure_modes(
    page_quality: Sequence[Dict[str, object]], conf_threshold: float = 0.75
) -> Dict[str, List[int]]:
    """Group pages by qualitative failure mode for the manual review step."""
    def pages(pred) -> List[int]:
        return sorted(int(p["page_number"]) for p in page_quality if pred(p))

    return {
        "low_confidence": pages(
            lambda p: int(p.get("num_text_blocks", 0)) > 0
            and float(p.get("mean_confidence", 1.0)) < conf_threshold
        ),
        "truncated": pages(lambda p: bool(p.get("truncated"))),
        "blank": pages(lambda p: bool(p.get("is_blank"))),
        "empty_non_blank": pages(
            lambda p: int(p.get("num_text_blocks", 0)) == 0 and not p.get("is_blank")
        ),
        "not_upright": pages(
            lambda p: str(p.get("orientation", "unknown")) not in ("upright", "unknown")
        ),
    }


def evaluate_criteria(
    recall: Optional[float],
    mean_conf: Optional[float],
    recall_target: float = 0.90,
    conf_target: float = 0.75,
) -> Dict[str, object]:
    """Check the two automatable issue-#10 success criteria (human agreement is manual)."""
    return {
        "question_recall_ok": None if recall is None else recall >= recall_target,
        "mean_confidence_ok": None if mean_conf is None else mean_conf >= conf_target,
        "targets": {"question_recall": recall_target, "mean_confidence": conf_target},
        "note": "Human-agreement >= 0.80 is a manual review step, not computed here.",
    }
