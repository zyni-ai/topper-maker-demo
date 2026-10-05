"""Image helpers for building multimodal LLM content and cropping diagrams."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Tuple

from PIL import Image


def detect_image_format_from_base64(base64_data: str) -> str:
    """Best-effort MIME detection from magic bytes; defaults to image/jpeg."""
    try:
        decoded = base64.b64decode(base64_data[:32])
    except Exception:
        return "image/jpeg"
    if decoded[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if decoded[:2] == b"\xff\xd8":
        return "image/jpeg"
    if decoded[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if decoded[:4] == b"RIFF" and decoded[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def image_block_from_url(image_url: str) -> Dict[str, Any]:
    """OpenAI/OpenRouter-compatible image content block from a URL."""
    return {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}


def image_block_from_base64(base64_data: str) -> Dict[str, Any]:
    """OpenAI/OpenRouter-compatible image content block from base64 data."""
    if base64_data.startswith("data:"):
        url = base64_data
    else:
        media_type = detect_image_format_from_base64(base64_data)
        url = f"data:{media_type};base64,{base64_data}"
    return {"type": "image_url", "image_url": {"url": url, "detail": "high"}}


def pil_to_base64(image: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    """Encode a PIL image to base64 (no data-URI prefix)."""
    if fmt.upper() == "JPEG" and image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=quality)
    return base64.standard_b64encode(buffer.getvalue()).decode()


def crop_normalised_bbox(
    image: Image.Image,
    bbox: List[float],
    padding_frac: float = 0.0,
) -> Image.Image:
    """Crop a region given a normalised ``[x0, y0, x1, y1]`` box in 0..1 coords.

    Padding is added symmetrically as a fraction of the image's width/height and
    the result is clamped to the image bounds. Vision-LLM bboxes are approximate,
    so a small pad avoids clipping the figure (the padding fraction is configurable
    and tuned empirically — see :class:`EvaluationConfig`).

    Falls back to the full image if the box is degenerate.
    """
    w, h = image.size
    x0, y0, x1, y1 = _sanitise_bbox(bbox)

    pad_x = padding_frac
    pad_y = padding_frac
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(1.0, x1 + pad_x)
    y1 = min(1.0, y1 + pad_y)

    left, top, right, bottom = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    if right <= left or bottom <= top:
        return image  # degenerate box → return full page rather than an empty crop
    return image.crop((left, top, right, bottom))


def bbox_area_frac(bbox: List[float]) -> float:
    """Area of a normalised bbox as a fraction of the page (0..1)."""
    x0, y0, x1, y1 = _sanitise_bbox(bbox)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _sanitise_bbox(bbox: List[float]) -> Tuple[float, float, float, float]:
    """Coerce a bbox to a valid ordered 0..1 tuple; default to full frame."""
    if not bbox or len(bbox) != 4:
        return 0.0, 0.0, 1.0, 1.0
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return 0.0, 0.0, 1.0, 1.0
    # Clamp to [0,1] and order corners.
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    return x0, y0, x1, y1
