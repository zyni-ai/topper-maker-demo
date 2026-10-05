"""Paper/rubric compiler.

Turns a **question-paper PDF** and an **answer-key PDF** into a reviewable
:class:`CompiledPaper` whose ``questions`` + ``optional_sections`` drop straight into
an :class:`~topper_maker.evaluation.schemas.state.EvaluationRequest`. This is the
input-side front door the evaluation engine was missing: it produces the marking
context (statements + rubrics + choice groups) the engine needs to grade a student.

Two vision passes (paper, then key) read each document into structured form; a pure
:func:`~topper_maker.evaluation.authoring.reconcile.reconcile` step joins them by
question number and surfaces every ambiguity as a warning. The output is meant to be
eyeballed/edited by a human before any student is graded — a rubric-extraction error
would otherwise silently corrupt every mark.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import List, TypeVar

from topper_maker.evaluation.authoring.reconcile import reconcile
from topper_maker.evaluation.authoring.schemas import (
    CompiledPaper,
    KeyExtraction,
    PaperExtraction,
)

_T = TypeVar("_T", PaperExtraction, KeyExtraction)
from topper_maker.evaluation.common.image_utils import image_block_from_base64, pil_to_base64
from topper_maker.evaluation.common.llm_client import OpenRouterClient
from topper_maker.evaluation.config import EvaluationConfig
from topper_maker.ingestion.pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)

# Printed papers read fine well below scan DPI; keep the payload small.
_RENDER_DPI = 150
# None = no cap: the model is bounded only by its context window. A forced cap can
# truncate a reasoning model before it emits the JSON (finish_reason=length). Page
# grouping (below) is what actually bounds the per-call output size.
_MAX_TOKENS = None
# Split vision calls into groups of this many pages so the output token budget is never
# exhausted mid-JSON (#82). A KSEAB section rarely spans more than 5-6 pages.
_PAGES_PER_CHUNK = 6
# Cap total pages rendered; over-cap papers are flagged for the human reviewer.
_MAX_RENDER_PAGES = 20

_PAPER_SYSTEM = """You are reading a school/board EXAM QUESTION PAPER. Extract every
numbered question from the QUESTION BODY into structured data.

Critical rules:
- Read questions ONLY from the actual numbered question body. IGNORE any "General
  Instructions" / summary block at the top — its section/numbering description is often
  inconsistent with the real questions. Trust the printed question numbers and sections.
- Capture each question's full statement verbatim. Render ALL mathematics, equations,
  symbols, units and chemical formulae in LaTeX (inline $...$, display $$...$$); do not
  use plain-text or unicode for math. Preserve MCQ options and table contents in the
  statement.
- max_marks: only set it if marks are printed for that question (e.g. "[4 marks]").
- section_label / section_title: the Section or Part the question sits under
  (e.g. label "A", title "Objective Questions").
- question_type: best-effort one of mcq, match, fill_blank, very_short, short,
  numerical, long, derivation.
- has_figure: true only if the question itself prints a figure to interpret.
- choice_group: when two (or more) questions are alternatives joined by "OR" (the
  student answers only one), give them the SAME short choice_group label (e.g. "C-OR",
  "E-OR"). Compulsory questions must have choice_group = null. If an "OR" alternative
  reuses the same printed number, still record both as separate questions with the same
  choice_group.
- section_optional_counts: if a section heading (or a printed instruction next to the
  section) says something like "Answer any 5 questions" or "Attempt any four", record
  that count here as {section_label: N} (e.g. {"B": 5}). Only set this for sections
  with an explicit "answer any N" / "attempt any N" instruction. Leave empty ({}) for
  fully compulsory sections. Do NOT confuse "answer any N" section rules with individual
  "OR" choices between pairs of questions (those use choice_group instead)."""

_KEY_SYSTEM = """You are reading an EXAM ANSWER KEY / EVALUATION RUBRIC. Extract the
marking scheme into structured data.

Render ALL mathematics, equations, symbols, units and chemical formulae in LaTeX
(inline $...$, display $$...$$) everywhere they appear — in expected_answer, in every
rubric point description, and in keywords. Do not use plain-text or unicode for math.

For each question:
- answer_kind = "objective" ONLY for a single-answer question worth one holistic
  comparison — a single-choice MCQ or a single fill-in-the-blank. Put the correct
  answer in expected_answer and leave rubric_points empty.
- answer_kind = "rubric" whenever the answer has SEVERAL independently-marked parts.
  This includes descriptive / numerical / derivation / diagram questions AND
  multi-component objective questions:
    * MATCH-THE-FOLLOWING: emit one rubric point per pair (key "pair_1", "pair_2", …;
      description like "A → q"; marks = the per-pair mark stated in the key, usually 1;
      allow_partial=false). Do NOT collapse a 5-pair match into one expected_answer.
    * MULTI-BLANK fill-in (more than one blank, each separately marked): emit one
      rubric point per blank (key "blank_1", …; marks = per-blank mark).
  break the marking scheme into value points in rubric_points. For each point set:
    * key: a short stable id (e.g. "statement", "formula", "substitution", "answer",
      "diagram"). Keys must be unique within the question.
    * description: what the student must demonstrate. Render any math/equations/units/
      chemical formulae in LaTeX (inline $...$, display $$...$$).
    * marks: marks for that point (the points should sum to the question's max marks).
    * keywords: acceptable key terms/phrases if the key lists them.
    * is_diagram: true for a point awarded for a drawn figure / labelled diagram.
    * allow_partial: true if the key says partial/step credit may be given for the point.
    * depends_on: for numericals, the substitution point depends_on the formula point,
      and the final-answer point depends_on the substitution point. Encode that chain.
- max_marks: the question's total marks if stated.

Also extract:
- section_distribution: the mark-distribution table (section label, question range,
  marks each, total).
- global_partial_rules: general partial-marking guidelines that apply across questions.
- penalties: common-mistake deductions.
- teacher_notes: any free-form examiner instructions.

Be faithful to the key; do not invent marks or points that are not present."""


class PaperCompiler:
    """Compiles a question paper + answer key into evaluation-ready inputs."""

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        client: OpenRouterClient | None = None,
    ) -> None:
        self.config = config or EvaluationConfig.from_env()
        self._client = client or OpenRouterClient(
            api_key=self.config.effective_llm_api_key or "",
            base_url=self.config.llm_base_url,
            site_url=self.config.site_url,
            site_name=self.config.site_name,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_s,
        )
        self._renderer = PDFExtractor(target_dpi=_RENDER_DPI, max_pages=_MAX_RENDER_PAGES)

        # Optional Mistral OCR front-end: read printed PDFs to markdown+LaTeX before
        # structured extraction. Enabled only when configured AND a key is present;
        # otherwise we transparently fall back to the vision-LLM image path.
        self._ocr = None
        if self.config.use_mistral_ocr and self.config.mistral_api_key:
            from topper_maker.evaluation.authoring.mistral_ocr import MistralOCR

            self._ocr = MistralOCR(
                api_key=self.config.mistral_api_key,
                model=self.config.mistral_ocr_model,
            )
            logger.info("Paper compiler using Mistral OCR (%s) front-end.", self.config.mistral_ocr_model)

    # -- Public API --------------------------------------------------------------

    def compile_sync(self, paper_pdf: str | Path, key_pdf: str | Path) -> CompiledPaper:
        """Synchronous convenience wrapper around :meth:`compile`."""
        import asyncio

        return asyncio.run(self.compile(paper_pdf, key_pdf))

    async def compile(self, paper_pdf: str | Path, key_pdf: str | Path) -> CompiledPaper:
        logger.info("Compiling paper=%s key=%s", paper_pdf, key_pdf)

        extra_warnings: List[str] = []
        paper, key = await asyncio.gather(
            self._extract_document(
                paper_pdf, _PAPER_SYSTEM, PaperExtraction, "compile_paper",
                "question paper", extra_warnings,
            ),
            self._extract_document(
                key_pdf, _KEY_SYSTEM, KeyExtraction, "compile_key",
                "answer key", extra_warnings,
            ),
        )

        logger.info("Paper read: %d question(s)", len(paper.questions))
        if not paper.questions:
            extra_warnings.append(
                "No questions were extracted from the question paper — the read may have "
                "failed or truncated. Review the source PDF and re-run before grading."
            )
        logger.info("Key read: %d question(s)", len(key.questions))
        if not key.questions:
            extra_warnings.append(
                "No answers were extracted from the answer key — every question will fall back "
                "to a placeholder. Review the source PDF and re-run before grading."
            )

        compiled = reconcile(paper, key)
        # Surface compiler-level (rendering/extraction) issues alongside reconciliation ones.
        compiled.warnings = [*extra_warnings, *compiled.warnings]
        logger.info("Compiled %s", json.dumps(compiled.summary()))
        for w in compiled.warnings:
            logger.warning("compile warning: %s", w)
        return compiled

    # -- Internals ---------------------------------------------------------------

    async def _extract_document(
        self,
        pdf_path: str | Path,
        system_prompt: str,
        schema_model: type[_T],
        label: str,
        doc_label: str,
        warnings: List[str],
    ) -> _T:
        """Extract structured data from a document.

        Uses Mistral OCR (single call) when available; otherwise renders pages as
        images and processes them in _PAGES_PER_CHUNK-page chunks to stay within
        the 8192 output-token budget (#82).
        """
        if self._ocr is not None:
            try:
                markdown = self._ocr.read_markdown(pdf_path, max_pages=_MAX_RENDER_PAGES)
                text = (
                    f"Extract the {doc_label} from this OCR text (LaTeX math already formatted):"
                    f"\n\n{markdown}"
                )
                return await self._client.complete_json(
                    model=self.config.vision_model,
                    system_prompt=system_prompt,
                    user_content=text,
                    schema_model=schema_model,
                    max_tokens=_MAX_TOKENS,
                    label=label,
                )
            except Exception as exc:  # noqa: BLE001 - degrade to the vision path
                logger.warning(
                    "Mistral OCR failed for %s (%s); falling back to vision LLM.", doc_label, exc
                )
                warnings.append(
                    f"Mistral OCR failed for the {doc_label} ({exc}); read with the vision "
                    "model instead — verify the extraction."
                )

        images = self._render(pdf_path, doc_label, warnings)
        return await self._extract_images_chunked(images, system_prompt, schema_model, label)

    async def _extract_images_chunked(
        self,
        images: List[str],
        system_prompt: str,
        schema_model: type[_T],
        label: str,
    ) -> _T:
        """Process pages in _PAGES_PER_CHUNK-page batches concurrently and merge results."""
        chunk_size = _PAGES_PER_CHUNK
        chunks = [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]

        async def _call_chunk(chunk_images: List[str], chunk_idx: int) -> _T:
            page_start = chunk_idx * chunk_size + 1
            page_end = page_start + len(chunk_images) - 1
            instruction = (
                f"Extract from pages {page_start}–{page_end} of this document. "
                "If a section heading visible earlier gives context for these questions, "
                "apply it here."
            )
            return await self._client.complete_json(
                model=self.config.vision_model,
                system_prompt=system_prompt,
                user_content=self._content(instruction, chunk_images),
                schema_model=schema_model,
                max_tokens=_MAX_TOKENS,
                label=f"{label}_chunk{chunk_idx + 1}",
            )

        if not chunks:
            return schema_model()  # type: ignore[call-arg]  # no pages → empty extraction
        partials: List[_T] = list(
            await asyncio.gather(*[_call_chunk(ch, i) for i, ch in enumerate(chunks)])
        )
        return _merge_extractions(partials)

    def _render(self, pdf_path: str | Path, label: str, warnings: List[str]) -> List[str]:
        result = self._renderer.extract(pdf_path)
        if not result.pages:
            errs = "; ".join(result.errors) or "no pages rendered"
            raise ValueError(f"Could not render {pdf_path}: {errs}")
        if result.total_pages > len(result.pages):
            warnings.append(
                f"The {label} has {result.total_pages} pages but only the first "
                f"{len(result.pages)} were read (cap={_MAX_RENDER_PAGES}); questions beyond "
                "that page were not compiled."
            )
        return [pil_to_base64(p.image, fmt="JPEG", quality=85) for p in result.pages]

    @staticmethod
    def _content(instruction: str, images: List[str]) -> list:
        blocks: list = [{"type": "text", "text": instruction}]
        blocks.extend(image_block_from_base64(b) for b in images)
        return blocks


def _merge_extractions(parts: list) -> object:
    """Merge partial PaperExtraction or KeyExtraction objects from page-range chunks."""
    if not parts:
        raise ValueError("No extraction parts to merge.")
    if len(parts) == 1:
        return parts[0]

    if isinstance(parts[0], PaperExtraction):
        questions = [q for p in parts for q in p.questions]
        return PaperExtraction(
            exam_title=next((p.exam_title for p in parts if p.exam_title), None),
            subject=next((p.subject for p in parts if p.subject), None),
            class_type=next((p.class_type for p in parts if p.class_type), None),
            total_marks=next((p.total_marks for p in parts if p.total_marks), None),
            questions=questions,
        )

    if isinstance(parts[0], KeyExtraction):
        questions = [q for p in parts for q in p.questions]
        # Dedup list-of-strings fields; earlier chunks are more likely to have global metadata.
        def _dedup(items):
            return list(dict.fromkeys(items))

        return KeyExtraction(
            section_distribution=[sd for p in parts for sd in p.section_distribution],
            questions=questions,
            global_partial_rules=_dedup(r for p in parts for r in p.global_partial_rules),
            penalties=_dedup(pen for p in parts for pen in p.penalties),
            teacher_notes=[n for p in parts for n in p.teacher_notes],
        )

    raise TypeError(f"Unknown extraction type: {type(parts[0])}")
