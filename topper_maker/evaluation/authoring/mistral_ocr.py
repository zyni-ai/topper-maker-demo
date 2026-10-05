"""Mistral OCR document reader for the paper/rubric compiler.

The compiler reads **printed** PDFs (question paper + answer key). A dedicated
document-OCR model handles dense, multi-column, table-heavy keys and — crucially —
emits **mathematics as LaTeX**, which a general vision LLM does less reliably. We use
it as a first pass: PDF → clean markdown (with LaTeX), which is then handed to the
structured-extraction LLM. This improves rubric extraction *and* gives LaTeX-native
``compiled.json`` output.

Only used on the authoring/compiler side; handwritten answer-sheet HTR stays on the
vision LLM (a print-OCR model is a weaker bet on messy handwriting).

Requires ``MISTRAL_API_KEY`` and the ``mistralai`` package (``uv sync --extra ocr``).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "mistral-ocr-latest"


class MistralOCR:
    """Thin wrapper over Mistral's OCR endpoint that returns page markdown."""

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set; cannot create the OCR client.")
        # Imported lazily so importing this module never requires the SDK/credentials.
        # The client class moved between SDK majors (top-level in 1.x, mistralai.client
        # in 2.x); support both.
        try:
            from mistralai import Mistral
        except ImportError:
            from mistralai.client import Mistral

        self._client = Mistral(api_key=api_key)
        self.model = model

    def read_markdown(self, pdf_path: str | Path, max_pages: Optional[int] = None) -> str:
        """OCR a local PDF and return its pages concatenated as markdown (LaTeX math).

        Pages are separated by a form-feed-style divider so the downstream extractor
        can still see page boundaries.
        """
        path = Path(pdf_path)
        pdf_bytes = path.read_bytes()
        b64 = base64.standard_b64encode(pdf_bytes).decode()

        kwargs = {
            "model": self.model,
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64}",
                "document_name": path.name,
            },
        }
        if max_pages:
            # Cap to actual PDF page count so we never request pages that don't exist.
            try:
                import fitz  # pymupdf
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    actual_pages = doc.page_count
                max_pages = min(max_pages, actual_pages)
            except Exception:  # noqa: BLE001 - proceed with the caller's limit on failure
                pass
            # Mistral page indices are 0-based.
            kwargs["pages"] = list(range(max_pages))

        logger.info("Mistral OCR reading %s (model=%s)", path.name, self.model)
        response = self._client.ocr.process(**kwargs)

        pages: List[str] = []
        for page in getattr(response, "pages", []) or []:
            md = getattr(page, "markdown", "") or ""
            idx = getattr(page, "index", len(pages))
            pages.append(f"<!-- page {idx} -->\n{md}".strip())

        if not pages:
            raise ValueError(f"Mistral OCR returned no pages for {path.name}.")
        logger.info("Mistral OCR read %d page(s) from %s", len(pages), path.name)
        return "\n\n".join(pages)
