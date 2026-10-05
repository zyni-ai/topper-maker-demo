"""Question paper + rubric generation from study material (simple RAG).

Flow: extract text -> chunk -> embed -> pick topics -> for each question slot retrieve the
most relevant chunks -> LLM writes the question AND its marking rubric in one go.

The output is a list of ``QuestionItem`` (with ``rubric_points``) which is exactly what the
evaluation pipeline consumes, so the rubric made here is the rubric used to grade.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pymupdf
from openai import OpenAI

from topper_maker.evaluation.schemas.question import QuestionItem, RubricPoint

EMBED_MODEL = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")
GEN_MODEL = os.getenv("GEN_MODEL", "anthropic/claude-sonnet-4-5")

# (section_id, label, marks, instruction for the model)
QUESTION_TYPES = {
    "mcq": ("part_a", "Part A - Multiple choice", 1,
            "A multiple-choice question with four options labelled A) B) C) D) written inside "
            "question_statement. expected_answer is the correct option, e.g. 'B) Mitochondria'."),
    "short": ("part_b", "Part B - Short answer", 2,
              "A short-answer question answerable in 2-3 sentences."),
    "long": ("part_c", "Part C - Long answer", 5,
             "A long-answer / explain / numerical question needing a structured answer."),
}

BLOOM_CYCLE = ["remember", "understand", "apply", "analyze"]


@dataclass
class Chunk:
    text: str
    page: int


# ---------------------------------------------------------------- material -> chunks
def extract_text(name: str, data: bytes) -> List[tuple[int, str]]:
    """Return [(page_no, text)] from a PDF / txt / md upload."""
    if name.lower().endswith(".pdf"):
        doc = pymupdf.open(stream=data, filetype="pdf")
        return [(i + 1, p.get_text()) for i, p in enumerate(doc)]
    return [(1, data.decode("utf-8", errors="ignore"))]


def chunk_pages(pages: List[tuple[int, str]], size: int = 1200, overlap: int = 150) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page, text in pages:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if len(piece) > 80:
                chunks.append(Chunk(piece, page))
            start += size - overlap
    return chunks


# ---------------------------------------------------------------- retrieval
class Index:
    """Tiny in-memory vector index. Falls back to keyword overlap if embeddings fail."""

    def __init__(self, client: OpenAI, chunks: List[Chunk]):
        self.client, self.chunks = client, chunks
        self.vectors: Optional[np.ndarray] = None
        try:
            self.vectors = self._embed([c.text for c in chunks])
        except Exception:  # embeddings unavailable for this key/provider -> keyword fallback
            self.vectors = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):
            r = self.client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 64])
            out += [d.embedding for d in r.data]
        v = np.array(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def search(self, query: str, k: int = 3) -> List[Chunk]:
        if self.vectors is not None:
            q = self._embed([query])[0]
            idx = np.argsort(-(self.vectors @ q))[:k]
        else:
            words = set(re.findall(r"\w+", query.lower()))
            scores = [len(words & set(re.findall(r"\w+", c.text.lower()))) for c in self.chunks]
            idx = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in idx]


# ---------------------------------------------------------------- LLM helpers
def _json_call(client: OpenAI, prompt: str, retries: int = 2) -> dict | list:
    last = None
    for _ in range(retries + 1):
        r = client.chat.completions.create(
            model=GEN_MODEL, temperature=0.4,
            messages=[{"role": "system", "content": "You are a careful exam setter. Reply with JSON only."},
                      {"role": "user", "content": prompt}])
        text = r.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last = e
            prompt += f"\n\nYour previous reply was not valid JSON ({e}). Return valid JSON only."
    raise ValueError(f"Model did not return valid JSON: {last}")


def pick_topics(client: OpenAI, chunks: List[Chunk], n: int) -> List[str]:
    step = max(1, len(chunks) // 12)
    sample = "\n---\n".join(c.text[:500] for c in chunks[::step][:12])
    data = _json_call(client, f"From this study material, list {n} distinct, examinable topics "
                              f"(short phrases) spread across the material. "
                              f'Return JSON: {{"topics": ["..."]}}\n\nMATERIAL:\n{sample}')
    topics = [t for t in data.get("topics", []) if isinstance(t, str)]
    return (topics * (n // max(len(topics), 1) + 1))[:n] if topics else ["the material"] * n


def _make_question(client: OpenAI, index: Index, qid: int, qtype: str, topic: str,
                   bloom: str, subject: str) -> QuestionItem:
    section, _, marks, how = QUESTION_TYPES[qtype]
    context = "\n\n".join(c.text for c in index.search(topic, k=3))
    prompt = f"""Subject: {subject}. Write ONE exam question about "{topic}", using ONLY the context below.
Type: {how}
Marks: {marks}. Cognitive level (Bloom): {bloom}.

Also write the marking rubric a board examiner would use. Rules:
- rubric_points are value points; their marks MUST add up to exactly {marks}.
- Each point: key (short snake_case), description (what the student must show),
  marks (number, multiples of 0.5), keywords (acceptable terms), depends_on (keys of earlier points
  that must be earned first - use for numericals, e.g. substitution depends on formula; else []).
- {"Use exactly one rubric point worth 1 mark: the correct option." if qtype == "mcq" else "Use 2-5 points."}
- expected_answer is the full model answer.

Return JSON: {{"question_statement": "...", "expected_answer": "...", "rubric_points":
[{{"key": "...", "description": "...", "marks": 1, "keywords": ["..."], "depends_on": []}}]}}

CONTEXT:
{context}"""
    d = _json_call(client, prompt)
    points = [RubricPoint(key=p["key"], description=p["description"], marks=float(p["marks"]),
                          keywords=p.get("keywords", []), depends_on=p.get("depends_on", []))
              for p in d["rubric_points"]]
    total = sum(p.marks for p in points)
    if abs(total - marks) > 1e-6:  # enforce marks add up, whatever the model did
        for p in points:
            p.marks = round(p.marks * marks / total * 2) / 2 or 0.5
        points[-1].marks = max(0.5, marks - sum(p.marks for p in points[:-1]))
    return QuestionItem(id=qid, question_number=qid, question_statement=d["question_statement"],
                        max_score=float(marks), expected_answer=d["expected_answer"],
                        rubric_points=points, topic=topic, section_id=section)


def generate_paper(client: OpenAI, material: List[Chunk], subject: str, counts: dict[str, int],
                   progress: Callable[[str], None] = lambda s: None) -> List[QuestionItem]:
    """counts e.g. {"mcq": 5, "short": 3, "long": 2}. Returns questions in paper order."""
    progress("Indexing material...")
    index = Index(client, material)
    total = sum(counts.values())
    progress("Choosing topics...")
    topics = pick_topics(client, material, total)

    slots, qid = [], 1
    for qtype in ("mcq", "short", "long"):
        for _ in range(counts.get(qtype, 0)):
            slots.append((qid, qtype, topics[qid - 1], BLOOM_CYCLE[qid % len(BLOOM_CYCLE)]))
            qid += 1

    progress(f"Writing {total} questions with rubrics...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(lambda s: _make_question(client, index, s[0], s[1], s[2], s[3], subject), slots))
