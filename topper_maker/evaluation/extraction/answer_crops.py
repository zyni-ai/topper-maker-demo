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
