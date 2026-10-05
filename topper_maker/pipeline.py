"""End-to-end extraction pipeline: ingest → QC → preprocess → HTR → structure → metrics."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from langsmith import traceable

from topper_maker.htr.base import BaseHTR, HTRResult
from topper_maker.ingestion.pdf_extractor import PDFExtractor, PageImage
from topper_maker.ingestion.quality_checker import QualityChecker, QualityReport
from topper_maker.layout.document_structure import DocumentStructure, DocumentStructureParser
from topper_maker.metrics.extraction_metrics import (
    ExtractionReport,
    GroundTruthPage,
    MetricsCalculator,
)
from topper_maker.preprocessing.image_processor import ImageProcessor, PreprocessConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    htr_engine: str = "openrouter"   # "openrouter" | "azure"
    target_dpi: int = 300
    max_pages: int = 5               # 0 = all
    max_question_number: int = 60    # validated upper bound for question numbers
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ground_truth_path: str | None = None
    openrouter_api_key: str | None = None
    openrouter_model: str | None = None
    azure_endpoint: str | None = None
    azure_key: str | None = None

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            htr_engine=os.getenv("HTR_ENGINE", "openrouter"),
            target_dpi=int(os.getenv("TARGET_DPI", "300")),
            max_pages=int(os.getenv("MAX_PAGES", "5")),
            max_question_number=int(os.getenv("MAX_QUESTION_NUMBER", "60")),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_model=os.getenv("OPENROUTER_MODEL"),
            azure_endpoint=os.getenv("AZURE_DOC_INTEL_ENDPOINT"),
            azure_key=os.getenv("AZURE_DOC_INTEL_KEY"),
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    config: PipelineConfig
    pages: list[PageImage] = field(default_factory=list)
    quality_reports: list[QualityReport] = field(default_factory=list)
    htr_results: list[HTRResult] = field(default_factory=list)
    structure: DocumentStructure | None = None
    report: ExtractionReport | None = None
    errors: list[str] = field(default_factory=list)

    def quality_report_for(self, page_number: int) -> QualityReport | None:
        return next((r for r in self.quality_reports if r.page_number == page_number), None)


# ---------------------------------------------------------------------------
# Trace redaction
# ---------------------------------------------------------------------------

# Config fields that hold credentials. They live on PipelineConfig, which is
# reachable from the PipelineResult that `run()` returns — so without redaction
# LangSmith serialises them into the trace in plaintext.
_SECRET_CONFIG_FIELDS = ("openrouter_api_key", "azure_key")


def _redact_secrets(outputs: dict) -> dict:
    """Replace credential fields with '***' before LangSmith serialises the run.

    Builds a copy via dataclasses.replace; the live config object is untouched.
    """
    result = outputs.get("output")
    if not hasattr(result, "config"):
        return outputs
    safe_config = replace(
        result.config,
        **{
            f: ("***" if getattr(result.config, f) else None)
            for f in _SECRET_CONFIG_FIELDS
        },
    )
    return {**outputs, "output": replace(result, config=safe_config)}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ExtractionPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig.from_env()
        self._htr: BaseHTR | None = None

    # -- Public entry point --------------------------------------------------

    @traceable(name="extraction_pipeline", run_type="chain", process_outputs=_redact_secrets)
    def run(self, pdf_path: str | Path) -> PipelineResult:
        result = PipelineResult(config=self.config)
        pdf_path = Path(pdf_path)

        logger.info("=== Stage 1: PDF Extraction ===")
        extraction = PDFExtractor(
            target_dpi=self.config.target_dpi,
            max_pages=self.config.max_pages,
        ).extract(pdf_path)

        result.pages = extraction.pages
        result.errors.extend(extraction.errors)
        logger.info(
            "Extracted %d pages (total in PDF: %d)",
            len(result.pages), extraction.total_pages,
        )

        logger.info("=== Stage 2: Quality Check ===")
        checker = QualityChecker()
        result.quality_reports = checker.check_all(result.pages)
        failed_qc = {r.page_number for r in result.quality_reports if not r.passed}
        blank_pages = {r.page_number for r in result.quality_reports if r.is_blank}
        if failed_qc:
            for r in result.quality_reports:
                if not r.passed:
                    logger.warning("Page %d quality issues: %s", r.page_number, r.warnings)

        logger.info("=== Stage 3: Preprocessing ===")
        processor = ImageProcessor(self.config.preprocess)
        processed_pages: list[PageImage] = []
        for page in result.pages:
            if page.page_number in blank_pages:
                logger.info("  Skipping blank page %d (no HTR needed)", page.page_number)
                continue
            processed_image = processor.process(page.image)
            from topper_maker.ingestion.pdf_extractor import PageImage as PI
            processed_pages.append(
                PI(
                    page_number=page.page_number,
                    image=processed_image,
                    width_px=page.width_px,
                    height_px=page.height_px,
                    dpi=page.dpi,
                    source_path=page.source_path,
                )
            )

        logger.info("=== Stage 4: HTR ===")
        htr_engine = self._get_htr_engine()
        result.htr_results = []
        for page in processed_pages:
            logger.info("  Running HTR on page %d ...", page.page_number)
            try:
                htr_out = htr_engine.extract(page.image, page.page_number)
                result.htr_results.append(htr_out)
                logger.info(
                    "  Page %d: %d blocks, mean_conf=%.3f",
                    page.page_number, len(htr_out.blocks), htr_out.mean_confidence,
                )
            except Exception as exc:
                logger.error("  HTR failed on page %d: %s", page.page_number, exc)
                result.errors.append(f"HTR page {page.page_number}: {exc}")

        logger.info("=== Stage 5: Layout / Structure Parsing ===")
        parser = DocumentStructureParser(
            max_question_number=self.config.max_question_number,
        )
        result.structure = parser.parse(result.htr_results)
        logger.info(
            "Detected parts=%s  sections=%s  questions=%s",
            result.structure.parts,
            result.structure.sections,
            result.structure.question_numbers,
        )

        logger.info("=== Stage 6: Metrics ===")
        gt_pages = self._load_ground_truth()
        calculator = MetricsCalculator()

        extraction_report = ExtractionReport()
        for htr_out in result.htr_results:
            gt = next(
                (g for g in gt_pages if g.page_number == htr_out.page_number), None
            )
            pm = calculator.compute_page_metrics(htr_out, gt)

            # Pages that failed QC or had a truncated HTR response (finish_reason=length)
            # are flagged for human review regardless of confidence scores.
            if htr_out.page_number in failed_qc or htr_out.truncated:
                pm.needs_human_review = True

            extraction_report.page_metrics.append(pm)

        if gt_pages and result.structure:
            extraction_report.structure = calculator.compute_structure_metrics(
                result.structure, gt_pages
            )

        result.report = extraction_report
        return result

    # -- Helpers -------------------------------------------------------------

    def _get_htr_engine(self) -> BaseHTR:
        if self._htr is not None:
            return self._htr

        engine = self.config.htr_engine.lower()
        if engine == "azure":
            from topper_maker.htr.azure_htr import AzureHTR
            self._htr = AzureHTR(
                endpoint=self.config.azure_endpoint,
                api_key=self.config.azure_key,
            )
        else:
            from topper_maker.htr.openrouter_htr import OpenRouterHTR
            self._htr = OpenRouterHTR(
                api_key=self.config.openrouter_api_key,
                model=self.config.openrouter_model,
            )

        return self._htr

    def _load_ground_truth(self) -> list[GroundTruthPage]:
        if not self.config.ground_truth_path:
            return []
        gt_path = Path(self.config.ground_truth_path)
        if not gt_path.exists():
            logger.warning("Ground truth file not found: %s", gt_path)
            return []
        with gt_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        return [
            GroundTruthPage(
                page_number=item["page_number"],
                full_text=item["full_text"],
                question_numbers=item.get("question_numbers", []),
                parts=item.get("parts", []),
                sections=item.get("sections", []),
            )
            for item in raw
        ]
