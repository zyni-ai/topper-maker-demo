"""Handwritten content extractor — the HTR analogue of the reference OCR stage.

Produces the **same output contract** the downstream answer-mapper expects from
the reference Mistral-OCR extractor::

    {
      "success": bool,
      "total_pages": int,
      "pages": [
        {"page_number": int, "text_content": str,
         "images": [{"index": int, "name": str, "base64": str, "bbox": [...]}]}
      ],
      "error": str | None,
    }

…plus a parallel ``page_quality`` list carrying per-page QC signals used for
human-review routing. Keeping the core contract identical means the answer-mapper
and everything after it are unchanged whether the source is OCR or HTR.

Per-page processing:
  render (pymupdf) → QC (blank/orientation) → optional preprocess → HTR (Claude)
  → split text vs diagram blocks → crop diagram regions from the *original* page.

Pages are transcribed concurrently with a bounded semaphore. Blank pages (detected
by QC) skip the HTR call entirely to save API budget.
"""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from PIL import Image

from langsmith import traceable

from topper_maker.evaluation.common.concurrency import gather_bounded
from topper_maker.evaluation.common.image_utils import (
    bbox_area_frac,
    crop_normalised_bbox,
    pil_to_base64,
)
from topper_maker.evaluation.config import EvaluationConfig
from topper_maker.evaluation.schemas.state import PageQuality
from topper_maker.htr.base import RegionType
from topper_maker.htr.openrouter_htr import OpenRouterHTR
from topper_maker.ingestion.pdf_extractor import PageImage
from topper_maker.ingestion.quality_checker import Orientation, QualityChecker
from topper_maker.preprocessing.image_processor import ImageProcessor, PreprocessConfig

logger = logging.getLogger(__name__)

# Region types whose text contributes to the page's textual content. Diagrams are
# handled separately (cropped to images); everything legible goes into the text.
_TEXT_REGIONS = {RegionType.TEXT, RegionType.MATH, RegionType.TABLE, RegionType.UNKNOWN}


@dataclass
class _RenderedPage:
    page_number: int
    image: Image.Image  # original RGB render (used for diagram cropping)


@dataclass
class ExtractionResult:
    success: bool
    total_pages: int = 0
    pages: List[Dict[str, Any]] = field(default_factory=list)
    page_quality: List[PageQuality] = field(default_factory=list)
    error: str | None = None


class HTRContentExtractor:
    """Renders, quality-checks, and transcribes a PDF into the OCR-compatible shape."""

    def __init__(self, config: EvaluationConfig, usage_tracker=None) -> None:
        self._config = config
        self._quality = QualityChecker()
        self._preprocessor = (
            ImageProcessor(PreprocessConfig()) if config.apply_preprocessing else None
        )
        self._htr = OpenRouterHTR(
            api_key=config.effective_llm_api_key,
            model=config.htr_model,
            max_tokens=config.htr_max_tokens,
            base_url=config.llm_base_url,
            site_url=config.site_url,
            site_name=config.site_name,
            usage_tracker=usage_tracker,
        )

    @traceable(name="htr_extract", run_type="chain")
    async def extract(self, pdf_bytes: bytes) -> ExtractionResult:
        """Extract text + diagram images from every page of ``pdf_bytes``."""
        try:
            rendered = await asyncio.get_running_loop().run_in_executor(
                None, self._render_pages, pdf_bytes
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("PDF rendering failed: %s", exc)
            return ExtractionResult(success=False, error=f"PDF rendering failed: {exc}")

        if not rendered:
            return ExtractionResult(success=False, error="PDF produced no pages.")

        # Transcribe pages concurrently (bounded). Each returns (page_dict, PageQuality).
        # _safe_process catches per-page exceptions so one bad page doesn't abort all (#51).
        factories = [
            (lambda p=page: self._safe_process(p)) for page in rendered
        ]
        results = await gather_bounded(factories, self._config.max_concurrent_htr)

        pages = [r[0] for r in results]
        quality = [r[1] for r in results]
        # Global image indices must be assigned across the whole booklet, in page order.
        self._assign_global_indices(pages)

        return ExtractionResult(
            success=True,
            total_pages=len(pages),
            pages=pages,
            page_quality=quality,
        )

    # -- Rendering ---------------------------------------------------------------

    def _render_pages(self, pdf_bytes: bytes) -> List[_RenderedPage]:
        """Render each PDF page to an RGB PIL image at the target DPI (sync)."""
        import fitz  # pymupdf

        rendered: List[_RenderedPage] = []
        scale = self._config.target_dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            limit = min(doc.page_count, self._config.max_pages)
            for idx in range(limit):
                pixmap = doc[idx].get_pixmap(matrix=matrix, alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                rendered.append(_RenderedPage(page_number=idx + 1, image=image))
        return rendered

    # -- Per-page processing -----------------------------------------------------

    @traceable(name="htr_page", run_type="chain")
    async def _process_page(self, page: _RenderedPage) -> tuple[Dict[str, Any], PageQuality]:
        """Quality-check, transcribe, and split one page into text + diagram images."""
        page_image = PageImage(
            page_number=page.page_number,
            image=page.image,
            width_px=page.image.width,
            height_px=page.image.height,
            dpi=self._config.target_dpi,
            source_path=None,  # type: ignore[arg-type]  # not file-backed here
        )
        qc = self._quality.check(page_image)

        # Blank page → skip HTR entirely (cost saving) and mark for differentiated review.
        if qc.is_blank:
            logger.info("Page %d is blank; skipping HTR.", page.page_number)
            return (
                self._blank_page_dict(page.page_number),
                PageQuality(
                    page_number=page.page_number,
                    is_blank=True,
                    orientation=qc.orientation.value,
                    warnings=["blank page — no ink detected"],
                ),
            )

        # Optional preprocessing for the HTR input only; cropping uses the original.
        htr_input = page.image
        if self._preprocessor is not None:
            htr_input = await asyncio.get_running_loop().run_in_executor(
                None, self._preprocessor.process, page.image
            )

        # HTR is synchronous; run it off the event loop so pages transcribe in parallel.
        htr_result = await asyncio.get_running_loop().run_in_executor(
            None, self._htr.extract, htr_input, page.page_number
        )

        text_content, images, diagram_warnings = self._split_blocks(
            page.image, htr_result.blocks, page.page_number
        )

        warnings: List[str] = []
        if qc.orientation not in (Orientation.UPRIGHT, Orientation.UNKNOWN):
            warnings.append(f"page may be rotated ({qc.orientation.value})")
        if htr_result.truncated:
            warnings.append("HTR response truncated (partial extraction)")
        if not htr_result.blocks:
            warnings.append("HTR returned no blocks")
        warnings.extend(diagram_warnings)

        quality = PageQuality(
            page_number=page.page_number,
            is_blank=False,
            orientation=qc.orientation.value,
            mean_confidence=round(htr_result.mean_confidence, 4),
            min_confidence=round(htr_result.min_confidence, 4),
            num_text_blocks=sum(1 for b in htr_result.blocks if b.region_type in _TEXT_REGIONS),
            num_diagram_blocks=len(images),
            truncated=htr_result.truncated,
            warnings=warnings,
        )
        page_dict = {
            "page_number": page.page_number,
            "text_content": text_content,
            "images": images,
        }
        return page_dict, quality

    async def _safe_process(self, page: _RenderedPage) -> tuple[Dict[str, Any], PageQuality]:
        """Wrapper around _process_page that catches exceptions so one bad page
        doesn't crash the whole booklet extraction (#51)."""
        try:
            return await self._process_page(page)
        except Exception as exc:
            logger.error("Page %d processing failed: %s", page.page_number, exc)
            return (
                {
                    "page_number": page.page_number,
                    "text_content": f"[PAGE_ERROR: {exc}]",
                    "images": [],
                },
                PageQuality(
                    page_number=page.page_number,
                    is_blank=False,
                    orientation="unknown",
                    warnings=[f"page processing error: {exc}"],
                ),
            )

    def _split_blocks(
        self, original: Image.Image, blocks: list, page_number: int
    ) -> tuple[str, List[Dict[str, Any]], List[str]]:
        """Join text blocks; crop diagram blocks from the original page image.

        Returns ``(text, images, warnings)``. The warnings make two previously
        silent degradations visible (issue #124):

        - a diagram region smaller than ``min_diagram_area_frac`` is dropped (not
          cropped) — likely noise, but a genuine small figure would vanish silently;
        - a diagram bbox at/above ``max_diagram_area_frac`` covers essentially the
          whole page (usually a missing/degenerate box). We still keep the crop so
          the vision judge sees the work, but flag that the figure was not localised.
        """
        text_parts: List[str] = []
        images: List[Dict[str, Any]] = []
        warnings: List[str] = []
        diagram_idx = 0

        for block in blocks:
            if block.region_type == RegionType.DIAGRAM:
                area = bbox_area_frac(block.bbox)
                if area < self._config.min_diagram_area_frac:
                    warnings.append(
                        f"diagram region too small to crop ({area:.1%} of page); dropped"
                    )
                else:
                    if area >= self._config.max_diagram_area_frac:
                        warnings.append(
                            f"diagram bbox covers ~the whole page ({area:.0%}); figure "
                            "could not be localised — sending the full page to the judge"
                        )
                    crop = crop_normalised_bbox(
                        original, block.bbox, padding_frac=self._config.bbox_padding_frac
                    )
                    images.append(
                        {
                            # 'index' is page-local here; the global index is assigned later.
                            "index": diagram_idx,
                            "name": f"diagram_p{page_number}_{diagram_idx}.jpeg",
                            "base64": pil_to_base64(crop),
                            "bbox": block.bbox,
                        }
                    )
                    diagram_idx += 1
            # Keep any text on the block (caption/label, or text wrongly tagged diagram).
            if block.text.strip():
                text_parts.append(block.text)

        return "\n".join(text_parts), images, warnings

    # -- Helpers -----------------------------------------------------------------

    @staticmethod
    def _blank_page_dict(page_number: int) -> Dict[str, Any]:
        return {
            "page_number": page_number,
            "text_content": "[BLANK PAGE]",
            "images": [],
        }

    @staticmethod
    def _assign_global_indices(pages: List[Dict[str, Any]]) -> None:
        """Assign booklet-wide image indices in page order (matches OCR semantics)."""
        global_idx = 0
        for page in pages:
            for img in page.get("images", []):
                img["index"] = global_idx
                global_idx += 1
