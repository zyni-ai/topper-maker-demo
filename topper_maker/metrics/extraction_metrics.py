"""Real extraction quality metrics: CER, WER, confidence stats, structure recall."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import ClassVar

from jiwer import cer, wer

from topper_maker.htr.base import HTRResult
from topper_maker.layout.document_structure import DocumentStructure


# ---------------------------------------------------------------------------
# Ground-truth schema (matches ground_truth/*.json)
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthPage:
    page_number: int
    full_text: str                        # Entire page verbatim
    question_numbers: list[int] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-page metrics
# ---------------------------------------------------------------------------

@dataclass
class PageMetrics:
    page_number: int
    cer: float                 # Character Error Rate  (0 = perfect)
    wer: float                 # Word Error Rate       (0 = perfect)
    mean_confidence: float
    min_confidence: float
    num_blocks: int
    needs_human_review: bool   # True when confidence or error rate is too high

    REVIEW_CONFIDENCE_THRESHOLD: ClassVar[float] = 0.80
    REVIEW_WER_THRESHOLD: ClassVar[float] = 0.15


# ---------------------------------------------------------------------------
# Structure detection metrics
# ---------------------------------------------------------------------------

@dataclass
class StructureMetrics:
    # Questions
    gt_question_count: int
    detected_question_count: int
    question_recall: float     # detected ∩ gt / gt
    question_precision: float  # detected ∩ gt / detected

    # Parts
    gt_parts: list[str]
    detected_parts: list[str]
    part_recall: float

    # Sections
    gt_sections: list[str]
    detected_sections: list[str]
    section_recall: float


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class ExtractionReport:
    page_metrics: list[PageMetrics] = field(default_factory=list)
    structure: StructureMetrics | None = None

    @property
    def mean_cer(self) -> float:
        return _mean([m.cer for m in self.page_metrics])

    @property
    def mean_wer(self) -> float:
        return _mean([m.wer for m in self.page_metrics])

    @property
    def mean_confidence(self) -> float:
        return _mean([m.mean_confidence for m in self.page_metrics])

    @property
    def flagged_for_review(self) -> list[int]:
        return [m.page_number for m in self.page_metrics if m.needs_human_review]

    def summary(self) -> dict:
        return {
            "pages_processed": len(self.page_metrics),
            "mean_cer": round(self.mean_cer, 4),
            "mean_wer": round(self.mean_wer, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "pages_flagged_for_review": self.flagged_for_review,
            "structure": (
                {
                    "question_recall": round(self.structure.question_recall, 4),
                    "question_precision": round(self.structure.question_precision, 4),
                    "part_recall": round(self.structure.part_recall, 4),
                    "section_recall": round(self.structure.section_recall, 4),
                }
                if self.structure
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class MetricsCalculator:
    def compute_page_metrics(
        self,
        htr_result: HTRResult,
        ground_truth: GroundTruthPage | None = None,
    ) -> PageMetrics:
        mean_conf = htr_result.mean_confidence
        min_conf = htr_result.min_confidence

        if ground_truth is not None:
            hypothesis = htr_result.full_text
            reference = ground_truth.full_text
            page_cer = self._safe_cer(reference, hypothesis)
            page_wer = self._safe_wer(reference, hypothesis)
        else:
            # No ground truth: use confidence as a proxy
            page_cer = max(0.0, 1.0 - mean_conf)
            page_wer = max(0.0, 1.0 - mean_conf)

        needs_review = (
            mean_conf < PageMetrics.REVIEW_CONFIDENCE_THRESHOLD
            or page_wer > PageMetrics.REVIEW_WER_THRESHOLD
        )

        return PageMetrics(
            page_number=htr_result.page_number,
            cer=round(page_cer, 4),
            wer=round(page_wer, 4),
            mean_confidence=round(mean_conf, 4),
            min_confidence=round(min_conf, 4),
            num_blocks=len(htr_result.blocks),
            needs_human_review=needs_review,
        )

    def compute_structure_metrics(
        self,
        structure: DocumentStructure,
        ground_truth_pages: list[GroundTruthPage],
    ) -> StructureMetrics:
        gt_questions: set[int] = set()
        gt_parts: set[str] = set()
        gt_sections: set[str] = set()

        for gt in ground_truth_pages:
            gt_questions.update(gt.question_numbers)
            gt_parts.update(gt.parts)
            gt_sections.update(gt.sections)

        det_questions = set(structure.question_numbers)
        det_parts = set(structure.parts)
        det_sections = set(structure.sections)

        q_tp = len(gt_questions & det_questions)
        q_recall = q_tp / len(gt_questions) if gt_questions else 0.0
        q_precision = q_tp / len(det_questions) if det_questions else 0.0

        p_tp = len(gt_parts & det_parts)
        p_recall = p_tp / len(gt_parts) if gt_parts else 0.0

        s_tp = len(gt_sections & det_sections)
        s_recall = s_tp / len(gt_sections) if gt_sections else 0.0

        return StructureMetrics(
            gt_question_count=len(gt_questions),
            detected_question_count=len(det_questions),
            question_recall=round(q_recall, 4),
            question_precision=round(q_precision, 4),
            gt_parts=sorted(gt_parts),
            detected_parts=sorted(det_parts),
            part_recall=round(p_recall, 4),
            gt_sections=sorted(gt_sections),
            detected_sections=sorted(det_sections),
            section_recall=round(s_recall, 4),
        )

    @staticmethod
    def _safe_cer(reference: str, hypothesis: str) -> float:
        if not reference.strip():
            return 0.0
        try:
            return float(cer(reference, hypothesis))
        except Exception:
            return 1.0

    @staticmethod
    def _safe_wer(reference: str, hypothesis: str) -> float:
        if not reference.strip():
            return 0.0
        try:
            return float(wer(reference, hypothesis))
        except Exception:
            return 1.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0
