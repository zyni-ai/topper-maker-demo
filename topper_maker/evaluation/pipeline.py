"""Evaluation pipeline orchestrator.

Stages: validate → extract (HTR) → moderate → map answers → evaluate → aggregate
→ route review. Each stage is timed for observability and isolated so a recoverable
failure degrades to human-review routing rather than crashing the run.

Human-review routing (decisions from issue #9):
- low per-page HTR confidence or truncated (partial) extraction → whole booklet to review
- rotated/illegible pages → flag those pages, booklet to review
- blank pages → surfaced distinctly; unattempted questions on a booklet that has
  blank pages are marked BLANK_PAGE_SUSPECTED (a scan may have eaten the answer)
- a moderation *error* (couldn't check) routes to review rather than rejecting,
  while a genuine moderation *flag* rejects the sheet (is_valid=False)
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import Dict, List, Optional

from langsmith import traceable

from topper_maker.evaluation.config import EvaluationConfig
from topper_maker.evaluation.common.llm_client import OpenRouterClient
from topper_maker.evaluation.common.usage import UsageTracker, usage_or_none
from topper_maker.evaluation.extraction.htr_content_extractor import HTRContentExtractor
from topper_maker.evaluation.guardrails.input_validator import validate_input
from topper_maker.evaluation.guardrails.moderation import build_moderator
from topper_maker.evaluation.mapping.answer_mapper import AnswerMapper
from topper_maker.evaluation.schemas.feedback import AnswerStatus, IndividualFeedback
from topper_maker.evaluation.schemas.question import QuestionItem
from topper_maker.evaluation.schemas.state import (
    EvaluationRequest,
    EvaluationResponse,
    PageQuality,
    ReviewReason,
)
from topper_maker.evaluation.schemas.supervision import SupervisionRecord
from topper_maker.evaluation.scoring.evaluator import AnswerEvaluator
from topper_maker.evaluation.scoring.marks_aggregator import (
    apply_counted_flags,
    compute_total_marks,
)
from topper_maker.evaluation.scoring.supervisor import SupervisorArbitrator
from topper_maker.evaluation.extraction.answer_crops import build_answer_crops
from topper_maker.evaluation.storage.s3_uploader import S3Uploader

logger = logging.getLogger(__name__)


class _Timer:
    """Accumulates per-stage wall-clock timings in milliseconds."""

    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    def stage(self, name: str) -> "_StageTimer":
        return _StageTimer(self, name)


class _StageTimer:
    def __init__(self, timer: _Timer, name: str) -> None:
        self._timer = timer
        self._name = name

    def __enter__(self) -> "_StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self._timer.timings[self._name] = round((time.perf_counter() - self._start) * 1000, 1)


class EvaluationPipeline:
    """End-to-end rubric-driven evaluation of one scanned answer sheet."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig.from_env()
        # UsageTracker is created fresh per evaluate() call so concurrent evaluations
        # don't cross-contaminate cost accounting (issue #62).
        self._client = OpenRouterClient(
            api_key=self.config.effective_llm_api_key or "",
            base_url=self.config.llm_base_url,
            site_url=self.config.site_url,
            site_name=self.config.site_name,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_s,
            usage_tracker=None,
        )
        self._extractor = HTRContentExtractor(self.config, usage_tracker=None)
        self._moderator = build_moderator(self.config, self._client)
        self._mapper = AnswerMapper(self._client, self.config)
        self._evaluator = AnswerEvaluator(self._client, self.config)
        self._s3 = S3Uploader(self.config)
        # Supervisor is only wired when both a supervisor_model is set AND there are
        # multiple candidate eval_models; a single model needs no arbitration.
        self._supervisor: Optional[SupervisorArbitrator] = (
            SupervisorArbitrator(self._client, self.config, self.config.eval_models)
            if self.config.supervisor_model and len(self.config.eval_models) > 1
            else None
        )

    # -- Public API --------------------------------------------------------------

    def evaluate_sync(self, request: EvaluationRequest) -> EvaluationResponse:
        """Synchronous convenience wrapper around :meth:`evaluate`.

        The HTTP clients are closed inside the loop ``asyncio.run`` manages so their
        httpx connection pools are drained while that loop is still alive. Otherwise
        the pools are only torn down later by the garbage collector on the (now
        closed) loop, which on Windows raises ``RuntimeError('Event loop is closed')``
        (issue #138).
        """
        import asyncio

        async def _run() -> EvaluationResponse:
            try:
                return await self.evaluate(request)
            finally:
                await self.aclose()

        return asyncio.run(_run())

    async def aclose(self) -> None:
        """Close the HTTP clients this pipeline owns (LLM client + moderator)."""
        await self._client.aclose()
        await self._moderator.aclose()

    @traceable(name="evaluate_answer_sheet", run_type="chain")
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        rid = request.request_id or request.user_test_id
        timer = _Timer()
        usage = UsageTracker()
        self._client._usage_tracker = usage
        self._extractor._htr._usage_tracker = usage
        logger.info("[%s] Evaluation start: %s", rid, request.answer_sheet_url)

        # 1. Input guardrails ----------------------------------------------------
        with timer.stage("validate"):
            validation = await validate_input(
                request.answer_sheet_url,
                max_file_size_mb=self.config.max_file_size_mb,
                max_pages=self.config.max_pages,
                download_timeout_s=self.config.download_timeout_s,
                max_retries=self.config.max_retries,
                allow_local_paths=self.config.allow_local_paths,
            )
        if not validation.valid:
            logger.warning("[%s] Rejected at validation: %s", rid, validation.error)
            return EvaluationResponse(
                is_valid=False,
                rejection_reason=validation.error,
                request_id=rid,
                stage_timings_ms=timer.timings,
            )

        # 2. HTR extraction ------------------------------------------------------
        with timer.stage("extract"):
            extraction = await self._extractor.extract(validation.pdf_bytes)
        if not extraction.success:
            logger.error("[%s] Extraction failed: %s", rid, extraction.error)
            return EvaluationResponse(
                is_valid=False,
                rejection_reason=f"Extraction failed: {extraction.error}",
                request_id=rid,
                stage_timings_ms=timer.timings,
            )

        extracted_text = "\n".join(p.get("text_content", "") for p in extraction.pages)

        # 3. Moderation ----------------------------------------------------------
        with timer.stage("moderate"):
            moderation = await self._moderator.moderate(extracted_text)
        moderation_errored = "moderation_error" in moderation.categories
        if moderation.is_flagged and not moderation_errored:
            logger.warning("[%s] Rejected by moderation: %s", rid, moderation.categories)
            return EvaluationResponse(
                is_valid=False,
                rejection_reason="Content flagged by moderation.",
                moderation_categories=moderation.categories,
                moderation_scores=moderation.scores,
                request_id=rid,
                stage_timings_ms=timer.timings,
            )

        # 4. Persist analysis (non-fatal) ---------------------------------------
        with timer.stage("upload_analysis"):
            analysis_url = await self._s3.upload_analysis(
                request, extracted_text, _moderation_dict(moderation)
            )

        # 5. Map answers ---------------------------------------------------------
        with timer.stage("map"):
            mapped = await self._mapper.map_answers(request.questions_list, extraction.pages)

        # 6. Upload student images, get UUID codes ------------------------------
        with timer.stage("upload_images"):
            image_codes_map = await self._s3.upload_images(request, mapped.all_images)

        # 7. Evaluate (one pass per candidate model, concurrently) -----------------
        with timer.stage("evaluate"):
            candidate_feedbacks = await self._evaluate_candidates(
                request.questions_list, mapped, image_codes_map
            )

        # 7b. Supervisor arbitration — only when >1 candidate + supervisor configured
        supervision_record: Optional[SupervisionRecord] = None
        candidate_responses_map: Optional[Dict[str, List[IndividualFeedback]]] = None

        if self._supervisor and len(candidate_feedbacks) > 1:
            with timer.stage("supervise"):
                responses, supervision_record = await self._supervisor.arbitrate(
                    request.questions_list, mapped, candidate_feedbacks, image_codes_map
                )
            candidate_responses_map = {
                model: fbs
                for model, fbs in zip(self.config.eval_models, candidate_feedbacks)
            }
        else:
            responses = candidate_feedbacks[0]

        # 8. Aggregate marks -----------------------------------------------------
        with timer.stage("aggregate"):
            marks = compute_total_marks(
                responses, request.questions_list, request.optional_sections
            )
            apply_counted_flags(responses, marks.counted_question_ids)

        # 9. Review routing — _route_review is the single authority for the booklet
        #    review decision; per-question needs_review flags are also aggregated there.
        self._mark_blank_suspected(responses, extraction.page_quality)
        review_reasons, flagged_pages, needs_review = self._route_review(
            extraction.page_quality,
            responses,
            moderation_errored,
            mapped.mapping_failed,
            bool(mapped.unmapped_question_ids),
        )

        # Supplement review reasons with supervisor-specific signals.
        if supervision_record:
            if supervision_record.total_deviation > self.config.supervisor_deviation_threshold:
                review_reasons.append(ReviewReason.MODEL_DISAGREEMENT)
            if supervision_record.has_unresolved:
                review_reasons.append(ReviewReason.SUPERVISOR_UNRESOLVED)
                needs_review = True

        response = EvaluationResponse(
            is_valid=True,
            responses=responses,
            total_marks=marks.total_marks,
            max_marks=marks.max_marks,
            percentage=marks.percentage,
            section_summaries=marks.section_summaries,
            needs_human_review=needs_review,
            review_reasons=sorted(set(review_reasons), key=lambda r: r.value),
            flagged_pages=sorted(set(flagged_pages)),
            page_quality=extraction.page_quality,
            has_processing_errors=any(r.has_evaluation_error for r in responses),
            stage_timings_ms=timer.timings,
            request_id=rid,
            analysis_url=analysis_url,
            moderation_categories=moderation.categories if moderation_errored else None,
            cost_usd=usage.total_cost_usd if usage.records else None,
            usage=usage_or_none(usage),
            supervision=supervision_record,
            candidate_responses=candidate_responses_map,
            answer_crops=_safe_crops(extraction.pages, mapped.answers),
        )
        logger.info(
            "[%s] Done: %.1f/%.1f (%.1f%%), review=%s, reasons=%s",
            rid,
            marks.total_marks,
            marks.max_marks,
            marks.percentage,
            needs_review,
            [r.value for r in response.review_reasons],
        )
        return response

    # -- Review routing helpers --------------------------------------------------

    def _route_review(
        self,
        page_quality: List[PageQuality],
        responses: List[IndividualFeedback],
        moderation_errored: bool,
        mapping_failed: bool = False,
        unmapped_answers: bool = False,
    ) -> tuple[List[ReviewReason], List[int], bool]:
        reasons: List[ReviewReason] = []
        flagged_pages: List[int] = []

        for pq in page_quality:
            if pq.is_blank:
                reasons.append(ReviewReason.BLANK_PAGE_SCAN_ERROR)
                flagged_pages.append(pq.page_number)
                continue
            # "unknown" is whitelisted on purpose: the detector reports it when it
            # genuinely cannot determine the text axis (diagram-heavy / sparse pages),
            # so flagging it would false-positive on most figure pages. A populated
            # "rotated_90" (quarter-turn) is now reachable for real bulk-scan rotations.
            if pq.orientation not in ("upright", "unknown"):
                reasons.append(ReviewReason.ROTATED_PAGE)
                flagged_pages.append(pq.page_number)
            if pq.truncated:
                reasons.append(ReviewReason.PARTIAL_EXTRACTION)
                flagged_pages.append(pq.page_number)
            if not pq.is_blank and pq.num_text_blocks == 0:
                # HTR returned no blocks on a non-blank page — parse failure or exception
                # captured by _safe_process (#49, #51). Treat as partial extraction.
                reasons.append(ReviewReason.PARTIAL_EXTRACTION)
                flagged_pages.append(pq.page_number)
            elif pq.num_text_blocks > 0 and pq.mean_confidence < self.config.low_confidence_review_threshold:
                reasons.append(ReviewReason.LOW_HTR_CONFIDENCE)
                flagged_pages.append(pq.page_number)

        if any(r.has_evaluation_error for r in responses):
            reasons.append(ReviewReason.EVALUATION_ERROR)
        if any(r.low_trust_short_answer for r in responses):
            reasons.append(ReviewReason.UNVERIFIED_SHORT_ANSWER)
        if any(r.diagram_expected_missing for r in responses):
            reasons.append(ReviewReason.DIAGRAM_EXPECTED_NOT_FOUND)
        if moderation_errored:
            reasons.append(ReviewReason.MODERATION_ERROR)
        if mapping_failed:
            reasons.append(ReviewReason.MAPPING_ERROR)
        if unmapped_answers:
            reasons.append(ReviewReason.UNMAPPED_ANSWER)

        # Aggregate per-question flags into the booklet-level decision (#77).
        needs_review = bool(reasons) or any(r.needs_review for r in responses)
        return reasons, flagged_pages, needs_review

    @staticmethod
    def _mark_blank_suspected(
        responses: List[IndividualFeedback], page_quality: List[PageQuality]
    ) -> None:
        """If the booklet has blank pages, an unattempted answer may be a scan error.

        Surface those distinctly from genuine non-attempts (issue #9) and flag them
        for review.
        """
        if not any(pq.is_blank for pq in page_quality):
            return
        for r in responses:
            if r.answer_status == AnswerStatus.UNATTEMPTED:
                r.answer_status = AnswerStatus.BLANK_PAGE_SUSPECTED
                r.needs_review = True
                r.feedback = (
                    "No answer was found, and this booklet has a blank page that may be "
                    "a scan error. Please verify before finalising."
                )


    async def _evaluate_candidates(
        self,
        questions_list: List[QuestionItem],
        mapped,
        image_codes_map: Dict[int, str],
    ) -> List[List[IndividualFeedback]]:
        """Run one evaluation pass per model in config.eval_models, concurrently.

        Returns a list of feedback lists — one per candidate model, in the same
        order as config.eval_models.  Single-model case returns a list of length 1
        (identical to the original single-evaluator behaviour).
        """
        async def _run_one(model: str) -> List[IndividualFeedback]:
            model_cfg = dataclasses.replace(
                self.config,
                vision_model=model,
                text_model=model,
            )
            evaluator = AnswerEvaluator(self._client, model_cfg)
            return await evaluator.evaluate(questions_list, mapped, image_codes_map)

        return list(
            await asyncio.gather(*(_run_one(m) for m in self.config.eval_models))
        )


def _safe_crops(pages, answers):
    try:
        return build_answer_crops(pages, answers)
    except Exception as exc:  # noqa: BLE001 - crops are a nicety, never fail the evaluation
        logger.warning("Answer cropping failed (continuing): %s", exc)
        return {}


def _moderation_dict(moderation) -> dict:
    return {
        "is_flagged": moderation.is_flagged,
        "categories": moderation.categories,
        "scores": moderation.scores,
        "error": moderation.error,
    }
