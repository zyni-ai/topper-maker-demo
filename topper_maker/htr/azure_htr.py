"""HTR engine backed by Azure Document Intelligence (Read API).

Requires env vars:
  AZURE_DOC_INTEL_ENDPOINT
  AZURE_DOC_INTEL_KEY
"""

from __future__ import annotations

import io
import logging
import os

from PIL import Image

from topper_maker.htr.base import BaseHTR, HTRResult, RegionType, TextBlock

logger = logging.getLogger(__name__)


class AzureHTR(BaseHTR):
    """Uses Azure Document Intelligence prebuilt-read for handwriting recognition."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:
            raise ImportError(
                "Install azure-ai-documentintelligence: "
                "pip install azure-ai-documentintelligence"
            ) from exc

        resolved_endpoint = endpoint or os.environ.get("AZURE_DOC_INTEL_ENDPOINT", "")
        resolved_key = api_key or os.environ.get("AZURE_DOC_INTEL_KEY", "")

        if not resolved_endpoint or not resolved_key:
            raise EnvironmentError(
                "Azure credentials missing. Set AZURE_DOC_INTEL_ENDPOINT and "
                "AZURE_DOC_INTEL_KEY in your .env file."
            )

        self._client = DocumentIntelligenceClient(
            endpoint=resolved_endpoint,
            credential=AzureKeyCredential(resolved_key),
        )

    def extract(self, image: Image.Image, page_number: int = 1) -> HTRResult:
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        image_bytes = self._to_bytes(image)
        img_w, img_h = image.size

        poller = self._client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(bytes_source=image_bytes),
        )
        result = poller.result()

        blocks: list[TextBlock] = []

        for page in result.pages or []:
            # Build a per-page word→confidence lookup (consume by occurrence order).
            # Using a list per word to handle repeated words on the same page correctly.
            word_conf: dict[str, list[float]] = {}
            for word in page.words or []:
                word_conf.setdefault(word.content, []).append(word.confidence or 0.85)

            for line in page.lines or []:
                confidence = self._line_confidence(line.content, word_conf)
                bbox = self._normalise_bbox(line.polygon or [], img_w, img_h)
                blocks.append(
                    TextBlock(
                        text=line.content,
                        confidence=confidence,
                        region_type=RegionType.TEXT,
                        bbox=bbox,
                    )
                )

        return HTRResult(
            page_number=page_number,
            blocks=blocks,
            engine="azure",
            raw_response={},
        )

    @staticmethod
    def _to_bytes(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _line_confidence(line_content: str, word_conf: dict[str, list[float]]) -> float:
        """Average confidence of words in this line using the page word table.

        Uses exact word-by-word lookup and consumes the first occurrence of each
        word to handle duplicates correctly.  The old substring approach
        (w.content in line.content) produced false positives for short words like
        'a', 'I', 'V' and always read from pages[0] regardless of the current page.
        """
        words = line_content.split()
        confidences: list[float] = []
        for word in words:
            if word in word_conf and word_conf[word]:
                confidences.append(word_conf[word].pop(0))
        if confidences:
            return sum(confidences) / len(confidences)
        return 0.85

    @staticmethod
    def _normalise_bbox(polygon: list, img_w: int, img_h: int) -> list[float]:
        if not polygon or img_w == 0 or img_h == 0:
            return [0.0, 0.0, 1.0, 1.0]
        xs = polygon[0::2]
        ys = polygon[1::2]
        return [
            min(xs) / img_w,
            min(ys) / img_h,
            max(xs) / img_w,
            max(ys) / img_h,
        ]
