"""Locate each question's handwritten answer on the scanned pages and crop it.

The HTR step records a bounding box for every text block. After the mapper decides which
question an answer belongs to, we match that answer's text back to the blocks that produced
it and crop their union from the page image, so a reviewer can see the student's own writing.
Best-effort: a question with no confident match simply gets no crop.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any, Dict, List

from PIL import Image

from topper_maker.evaluation.common.image_utils import crop_normalised_bbox, pil_to_base64

_MIN_OVERLAP = 0.6  # share of a block's words that must appear in the answer


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_answer_crops(
    pages: List[Dict[str, Any]], answers: list, padding: float = 0.01
) -> Dict[int, List[Dict[str, Any]]]:
    """Return {question_id: [{"page": n, "image_b64": jpeg-base64}, ...]}."""
    answer_words = {a.question_id: set(_words(a.user_answer or "")) for a in answers}
    # question_id -> page_number -> [bbox, ...]
    hits: Dict[int, Dict[int, List[List[float]]]] = {}

    for page in pages:
        for block in page.get("text_blocks", []):
            bw = _words(block["text"])
            if not bw:
                continue
            best_qid, best = None, 0.0
            for qid, aw in answer_words.items():
                score = sum(w in aw for w in bw) / len(bw)
                if score > best:
                    best_qid, best = qid, score
            if best_qid is not None and best >= _MIN_OVERLAP:
                hits.setdefault(best_qid, {}).setdefault(page["page_number"], []).append(block["bbox"])

    by_page = {p["page_number"]: p for p in pages}
    crops: Dict[int, List[Dict[str, Any]]] = {}
    for qid, per_page in hits.items():
        for page_no, boxes in sorted(per_page.items()):
            b64 = by_page[page_no].get("page_jpeg")
            if not b64:
                continue
            union = [
                min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes),
            ]
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            crop = crop_normalised_bbox(img, union, padding_frac=padding)
            crops.setdefault(qid, []).append({"page": page_no, "image_b64": pil_to_base64(crop)})
    return crops


# ---------------------------------------------------------------------------------------
# Vision-based locator (primary). Text matching above stays as the fallback.
#
# Vision models place a question *label* ("Q3", "4.") vertically far more reliably than they
# draw a tight box around handwriting, and answers are written as horizontal bands. So we
# ask only for vertical ranges and crop the FULL page width, which can't clip a line.
# ---------------------------------------------------------------------------------------
from pydantic import BaseModel, Field  # noqa: E402

from topper_maker.evaluation.common.concurrency import gather_bounded  # noqa: E402
from topper_maker.evaluation.common.image_utils import image_block_from_base64  # noqa: E402

_LOCATE_SYSTEM = """You look at ONE scanned page of a student's handwritten answer sheet and \
report where each question's answer sits on the page.

For every question that has any of its answer on this page, return one segment:
- question_id: the id given in the list (NOT the number the student wrote).
- y_start / y_end: vertical extent of that answer on the page, as fractions of page height
  (0 = top edge, 1 = bottom edge). Start just above the first line of the answer (include the
  question number the student wrote) and end just below its last line.
Rules:
- Handwriting at the very top of the page before any question number is the continuation of the
  previous page's answer: report it under the question it continues, if you can tell which.
- Segments must not overlap; consecutive answers should abut. Ignore headers, margins, rough work.
- Omit questions that are not on this page. Return {"segments": []} if the page is blank."""


class _Segment(BaseModel):
    question_id: int
    y_start: float = Field(..., ge=0, le=1)
    y_end: float = Field(..., ge=0, le=1)


class _Segments(BaseModel):
    segments: List[_Segment] = Field(default_factory=list)


async def locate_answer_crops(client, config, questions, pages, answers) -> Dict[int, List[Dict[str, Any]]]:
    """Vision-locate each answer's vertical band per page and crop it at full page width.

    Falls back to text matching for any question the vision pass did not place.
    """
    answered = {a.question_id for a in answers if (a.user_answer or "").strip()}
    qlist = "\n".join(
        f"- id={q.id} (printed number {q.question_number}): {q.question_statement[:120]}"
        for q in questions if q.id in answered
    )

    async def _one(page: Dict[str, Any]):
        b64 = page.get("page_jpeg")
        if not b64 or page.get("text_content") == "[BLANK PAGE]":
            return page["page_number"], []
        try:
            res = await client.complete_json(
                model=config.mapping_model,
                system_prompt=_LOCATE_SYSTEM,
                user_content=[
                    {"type": "text", "text": f"QUESTIONS THAT MAY APPEAR:\n{qlist}\n\nThis is page {page['page_number']}."},
                    image_block_from_base64(b64),
                ],
                schema_model=_Segments,
                max_tokens=config.htr_max_tokens,
                label="answer_locate",
            )
            return page["page_number"], res.segments
        except Exception:  # noqa: BLE001 - per-page best effort
            return page["page_number"], []

    results = await gather_bounded([(lambda p=p: _one(p)) for p in pages], config.max_concurrent_htr)
    by_page = {p["page_number"]: p for p in pages}
    crops: Dict[int, List[Dict[str, Any]]] = {}
    for page_no, segments in sorted(results, key=lambda r: r[0]):
        img = None
        for seg in segments:
            if seg.question_id not in answered or seg.y_end - seg.y_start < 0.01:
                continue
            if img is None:
                img = Image.open(io.BytesIO(base64.b64decode(by_page[page_no]["page_jpeg"])))
            # Full width, slightly loose vertically: a roomy crop beats a clipped one.
            top = int(max(0.0, seg.y_start - 0.015) * img.height)
            bottom = int(min(1.0, seg.y_end + 0.015) * img.height)
            crop = img.crop((0, top, img.width, bottom))
            crops.setdefault(seg.question_id, []).append({"page": page_no, "image_b64": pil_to_base64(crop)})

    # Fallback for answered questions the vision pass could not place.
    missing = answered - set(crops)
    if missing:
        crops.update(build_answer_crops(pages, [a for a in answers if a.question_id in missing]))
    return crops
