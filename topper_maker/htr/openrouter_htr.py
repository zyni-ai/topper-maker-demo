"""HTR engine backed by OpenRouter (OpenAI-compatible API).

Supports any vision-capable model available on OpenRouter, e.g.:
  anthropic/claude-sonnet-4-5
  google/gemini-flash-1.5
  openai/gpt-4o

Requires env vars:
  OPENROUTER_API_KEY
  OPENROUTER_MODEL  (optional, defaults to anthropic/claude-sonnet-4-5)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time

import openai
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from PIL import Image

from topper_maker.htr.base import BaseHTR, HTRResult, RegionType, TextBlock

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"

# Retries on transient API errors (429, 5xx, network).
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubled each retry

# Downscale to this long-edge pixel count before encoding.
# Keeps base64 payload ≤ ~4 MB while preserving detail at 300 DPI.
_MAX_IMAGE_LONG_EDGE = 3000

_SYSTEM_PROMPT = """You are an expert Handwritten Text Recognition (HTR) engine \
specialised in Indian school exam answer sheets (Karnataka PU board style).

Your ONLY job is to faithfully transcribe what is written on the page.
Do NOT interpret, grade, or summarise — just read and report.

Rules:
1. Transcribe EXACTLY what is written, preserving spelling, punctuation, and line breaks.
2. For mathematical expressions use LaTeX: inline → $...$, display → $$...$$
3. If a word or character is genuinely illegible, write [ILLEGIBLE].
4. Mark struck-through text as [STRUCK: <text>].
5. Mark overwritten text as [OVERWRITTEN: <old>→<new>].
6. Mark text inserted above/below a line as [INSERTED: <text>].
7. Classify each contiguous content block as one of: text | math | diagram | table
8. Treat pages labelled "Rough Work" or with only scratch calculations as region_type "rough".
9. If a page appears blank (no ink), return a single block: {"text": "[BLANK PAGE]", ...}

Return ONLY valid JSON — no markdown fences, no prose before or after.

Schema:
{
  "blocks": [
    {
      "text": "<transcribed content>",
      "confidence": <float 0.0-1.0>,
      "region_type": "text|math|diagram|table|rough",
      "bbox": [x0_norm, y0_norm, x1_norm, y1_norm]
    }
  ]
}

confidence = your certainty about the transcription (1.0 = certain, 0.0 = completely illegible).
bbox values are normalised to [0, 1] relative to image width/height, top-left origin.
"""


class OpenRouterHTR(BaseHTR):
    """Vision HTR via any model available on OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = 8192,
        base_url: str | None = None,
        site_url: str = "https://github.com/sharath-s-rao/topper-maker-evaluation-pipeline",
        site_name: str = "Topper Maker",
        usage_tracker=None,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        if not resolved_key:
            raise EnvironmentError(
                "No LLM API key set. Set LLM_API_KEY (or OPENROUTER_API_KEY) in your .env file."
            )
        resolved_base_url = (
            base_url or os.environ.get("LLM_BASE_URL") or _OPENROUTER_BASE_URL
        )
        # `usage: {include: true}` (cost accounting) and the attribution headers are
        # OpenRouter-specific; other providers (e.g. NVIDIA NIM) 400 on the usage param.
        self._is_openrouter = "openrouter.ai" in resolved_base_url

        self.model = model or os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
        self.max_tokens = max_tokens
        # Optional shared accountant so HTR cost rolls into the evaluation total.
        self._usage_tracker = usage_tracker

        # HTTP-Referer / X-Title are OpenRouter-specific attribution headers; only send
        # them when targeting OpenRouter so other OpenAI-compatible providers (e.g.
        # NVIDIA NIM) aren't handed headers they don't use.
        default_headers = (
            {"HTTP-Referer": site_url, "X-Title": site_name}
            if "openrouter.ai" in resolved_base_url
            else {}
        )
        # wrap_openai instruments the client so the real LLM call is traced as a
        # nested run in LangSmith: system+user prompt, the rendered page image
        # (from the image_url block), raw model response, token counts, and cost.
        self._client = wrap_openai(
            OpenAI(
                base_url=resolved_base_url,
                api_key=resolved_key,
                default_headers=default_headers,
            )
        )

    @staticmethod
    def _trace_inputs(inputs: dict) -> dict:
        """Strip the raw PIL image from the traced inputs.

        LangSmith would otherwise serialise it to an unreadable ``<PIL.Image…>``
        repr. The rendered image is captured on the nested wrapped-LLM run via
        the ``image_url`` content block, so we only keep scalar context here.
        """
        image = inputs.get("image")
        return {
            "page_number": inputs.get("page_number"),
            "image_size": list(image.size) if hasattr(image, "size") else None,
        }

    @traceable(name="htr_page", run_type="chain", process_inputs=_trace_inputs)
    def extract(self, image: Image.Image, page_number: int = 1) -> HTRResult:
        """Extract text from one page image."""
        image_b64 = self._encode_image(image)

        # max_tokens is omitted when None so the model uses its full remaining-context
        # budget rather than a forced cap that can truncate before the JSON is emitted.
        create_kwargs: dict = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Transcribe every handwritten element on this exam "
                                "answer sheet page. Return ONLY the JSON object."
                            ),
                        },
                    ],
                },
            ],
            # Force JSON output from token 1 — stops a model emitting reasoning
            # prose before the JSON, which wastes output tokens and breaks parsing.
            "response_format": {"type": "json_object"},
        }
        if self.max_tokens is not None:
            create_kwargs["max_tokens"] = self.max_tokens
        # Ask OpenRouter to return token + cost accounting; other providers 400 on it.
        if self._is_openrouter:
            create_kwargs["extra_body"] = {"usage": {"include": True}}

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(**create_kwargs)
                if self._usage_tracker is not None:
                    self._usage_tracker.record_response(response, self.model)
                break
            except (
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
            ) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "OpenRouter API error on page %d (attempt %d/%d): %s — retrying in %.1fs",
                        page_number, attempt + 1, _MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "OpenRouter API failed on page %d after %d attempts: %s",
                        page_number, _MAX_RETRIES, exc,
                    )
                    raise

        raw_text = response.choices[0].message.content or ""  # type: ignore[union-attr]
        finish_reason = response.choices[0].finish_reason  # type: ignore[union-attr]
        if finish_reason == "length":
            logger.warning(
                "Page %d: response truncated (finish_reason=length).",
                page_number,
            )
        logger.debug("Page %d raw response: %s", page_number, raw_text[:300])

        parsed = self._parse_response(raw_text)
        blocks = []
        for b in parsed.get("blocks", []):
            _c = b.get("confidence")
            confidence = float(_c) if _c is not None else 0.5
            blocks.append(TextBlock(
                text=b.get("text", ""),
                confidence=confidence,
                region_type=self._parse_region_type(b.get("region_type", "text")),
                bbox=b.get("bbox", [0.0, 0.0, 1.0, 1.0]),
            ))

        return HTRResult(
            page_number=page_number,
            blocks=blocks,
            engine=f"openrouter/{self.model}",
            raw_response=parsed,
            truncated=(finish_reason == "length"),
        )

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """JPEG-encode with downscaling; keeps payload ≤ ~4 MB."""
        img = image.copy()
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > _MAX_IMAGE_LONG_EDGE:
            scale = _MAX_IMAGE_LONG_EDGE / long_edge
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=92)
        return base64.standard_b64encode(buffer.getvalue()).decode()

    @staticmethod
    def _parse_response(raw: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Truncation recovery: try to find and close a partial blocks array.
        match = re.search(r'"blocks"\s*:\s*(\[.*)', cleaned, re.DOTALL)
        if match:
            partial = match.group(1)
            # Strip trailing comma or incomplete last entry, then close the structure.
            partial = re.sub(r",?\s*\{[^}]*$", "", partial).rstrip(",").strip()
            if not partial.endswith("]"):
                partial += "]"
            attempt = '{"blocks": ' + partial + "}"
            try:
                result = json.loads(attempt)
                logger.info(
                    "Recovered %d blocks from truncated JSON response.",
                    len(result.get("blocks", [])),
                )
                return result
            except json.JSONDecodeError:
                pass

        logger.warning("JSON parse failed entirely. Raw (first 500 chars): %s", raw[:500])
        return {"blocks": []}

    @staticmethod
    def _parse_region_type(value: str) -> RegionType:
        mapping = {
            "text": RegionType.TEXT,
            "math": RegionType.MATH,
            "diagram": RegionType.DIAGRAM,
            "table": RegionType.TABLE,
            "rough": RegionType.UNKNOWN,
        }
        return mapping.get(value.lower(), RegionType.UNKNOWN)
