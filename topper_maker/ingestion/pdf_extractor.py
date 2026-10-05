"""PDF → per-page PIL Images using pymupdf (no poppler dependency)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf
from PIL import Image


@dataclass
class PageImage:
    page_number: int          # 1-based
    image: Image.Image
    width_px: int
    height_px: int
    dpi: int
    source_path: Path


@dataclass
class ExtractionResult:
    pages: list[PageImage] = field(default_factory=list)
    total_pages: int = 0
    errors: list[str] = field(default_factory=list)


class PDFExtractor:
    """Renders PDF pages to PIL Images at a target DPI."""

    def __init__(self, target_dpi: int = 300, max_pages: int = 0) -> None:
        if target_dpi < 72:
            raise ValueError(f"target_dpi must be >= 72, got {target_dpi}")
        self.target_dpi = target_dpi
        self.max_pages = max_pages  # 0 = all pages

    def extract(self, pdf_path: str | Path) -> ExtractionResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

        result = ExtractionResult()

        with fitz.open(str(pdf_path)) as doc:
            result.total_pages = doc.page_count
            limit = self.max_pages if self.max_pages > 0 else doc.page_count

            for page_idx in range(min(limit, doc.page_count)):
                try:
                    page = doc[page_idx]
                    page_image = self._render_page(page, page_idx + 1, pdf_path)
                    result.pages.append(page_image)
                except Exception as exc:
                    result.errors.append(f"Page {page_idx + 1}: {exc}")

        return result

    def _render_page(
        self, page: fitz.Page, page_number: int, source_path: Path
    ) -> PageImage:
        # Scale factor: pymupdf default is 72 DPI
        scale = self.target_dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        img_bytes = pixmap.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        return PageImage(
            page_number=page_number,
            image=image,
            width_px=pixmap.width,
            height_px=pixmap.height,
            dpi=self.target_dpi,
            source_path=source_path,
        )
