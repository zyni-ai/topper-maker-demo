"""Central configuration for the evaluation pipeline.

All tunable knobs live here so behaviour can be changed via environment variables
without touching code. Values flagged "experimental" are the subject of the
next-phase benchmarking studies (preprocessing strategy, bbox padding) and are
deliberately exposed as config rather than hard-coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _get_opt_int(name: str, default: int | None) -> int | None:
    """Like :func:`_get_int` but supports an unbounded ("no cap") default.

    An env value of ``0`` or empty is read as "no cap" (``None``) so the model is
    bounded only by its context window; any positive integer sets an explicit ceiling.
    """
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    value = int(raw)
    return value if value > 0 else None


def _get_str_list(name: str, default: List[str]) -> List[str]:
    """Parse a comma-separated env var into a list of non-empty strings."""
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return [s.strip() for s in raw.split(",") if s.strip()]


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class EvaluationConfig:
    """Tunable configuration for :class:`EvaluationPipeline`."""

    # --- Provider / endpoint ----------------------------------------------------
    # Any OpenAI-compatible endpoint. Defaults to OpenRouter for back-compat; set
    # LLM_BASE_URL + LLM_API_KEY to target another provider, e.g. NVIDIA NIM at
    # https://integrate.api.nvidia.com/v1. Model IDs stay provider-namespaced
    # (e.g. "minimaxai/minimax-m3" on NVIDIA, "anthropic/claude-sonnet-4-5" on OpenRouter).
    openrouter_api_key: str | None = None  # kept for back-compat; falls back into llm_api_key
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str | None = None  # when unset, falls back to openrouter_api_key

    # --- Models -----------------------------------------------------------------
    htr_model: str = "anthropic/claude-sonnet-4-5"      # transcription
    vision_model: str = "anthropic/claude-sonnet-4-5"   # diagram-aware evaluation
    text_model: str = "anthropic/claude-haiku-4-5"      # text evaluation (faster/cheaper)
    moderation_model: str = "anthropic/claude-haiku-4-5"
    mapping_model: str = "anthropic/claude-sonnet-4-5"  # answer→question mapping

    # --- Supervisor / multi-model evaluation (issue #133) -----------------------
    # eval_models: candidate set for stage 7 (evaluate). When more than one model is
    # listed, the evaluate stage runs each concurrently and hands all result-lists to
    # the SupervisorArbitrator. A single model → original behaviour (no arbitration).
    eval_models: List[str] = field(
        default_factory=lambda: ["anthropic/claude-sonnet-4-5"]
    )
    # supervisor_model: when set alongside >1 eval_models, arbitrates candidates into
    # one authoritative List[IndividualFeedback]. None disables supervision entirely.
    supervisor_model: Optional[str] = None
    # supervisor_deviation_threshold: per-question and per-total score gap (in marks)
    # above which the critical (full re-evaluate) path fires. Tunable, not hard-coded.
    supervisor_deviation_threshold: float = 5.0

    # --- Moderation backend (issue #123) ----------------------------------------
    # "openrouter" (default): LLM classifier over OpenRouter (costs tokens, JSON-
    # parse/truncation failure modes). "openai": OpenAI's dedicated, free
    # ``/moderations`` endpoint — structured scores, no parse failures, needs
    # OPENAI_API_KEY. Both are fail-closed; the OpenRouter classifier stays as a
    # fallback. NOTE: the OpenAI backend sends transcribed student text to OpenAI —
    # see the data-governance section before using on real (minor) student data.
    moderation_backend: str = "openrouter"
    openai_api_key: str | None = None
    openai_moderation_model: str = "omni-moderation-latest"

    # --- Paper/rubric compiler (authoring) --------------------------------------
    # Use Mistral OCR to read the printed paper + key into markdown (LaTeX math)
    # before structured extraction. Falls back to the vision LLM when no key is set.
    use_mistral_ocr: bool = True
    mistral_api_key: str | None = None
    mistral_ocr_model: str = "mistral-ocr-latest"

    # --- Rendering / extraction -------------------------------------------------
    target_dpi: int = 300

    # Output-token ceilings for the OpenRouter completion calls. Default ``None`` means
    # "no cap" — the model is bounded only by its context window. A forced cap was
    # starving reasoning models (e.g. MiniMax-M, MiMo): their hidden reasoning trace
    # counts against max_tokens but is NOT returned in the content, so a low cap
    # truncates the response before any JSON is emitted (finish_reason=length, empty
    # body). Set the matching env var to a positive integer to re-impose a cost ceiling;
    # ``0`` or unset means unbounded. See docs/discussion_max_tokens_truncation.md.
    htr_max_tokens: int | None = None         # HTR transcription + answer mapping
    eval_max_tokens: int | None = None        # rubric / holistic-vision / holistic-batch scoring
    moderation_max_tokens: int | None = None  # content-safety classifier

    # Preprocessing toggle. EXPERIMENTAL: for handwritten scans, raw images may
    # beat deskew+CLAHE (aggressive denoising can erase faint strikethroughs).
    # Phase-2 A/B test decides the default; configurable until then.
    apply_preprocessing: bool = False

    # Diagram crop padding as a fraction of image dimension, added on each side.
    # EXPERIMENTAL: tuned empirically in phase 2. Vision-LLM bboxes are approximate,
    # so a small pad avoids clipping the figure.
    bbox_padding_frac: float = 0.04
    min_diagram_area_frac: float = 0.005  # ignore diagram boxes smaller than this
    # A diagram bbox at/above this fraction of the page is treated as un-localised:
    # almost always a missing/degenerate box the model returned as the whole frame
    # (which crops the entire page). We still keep the crop so the vision judge sees
    # the work, but emit a warning so the page is visible to a reviewer (issue #124).
    max_diagram_area_frac: float = 0.98

    # --- Concurrency ------------------------------------------------------------
    max_concurrent_htr: int = 4       # parallel page transcriptions
    max_concurrent_eval: int = 4      # parallel question evaluations
    eval_batch_size: int = 5          # text questions per evaluation LLM call

    # --- Guardrails -------------------------------------------------------------
    max_file_size_mb: float = 50.0
    max_pages: int = 40
    download_timeout_s: int = 30
    allow_local_paths: bool = False  # set True only in demo/test environments

    # --- Quality thresholds -----------------------------------------------------
    # Below this mean per-page HTR confidence, route the WHOLE booklet to review.
    low_confidence_review_threshold: float = 0.70

    # Short objective answers (one/two-word fill-blanks, single-word MCQ) are
    # vulnerable to confident HTR misreads that clear the page-level confidence
    # thresholds (issue #37). When such a question is not fully correct, flag it for
    # human verification rather than silently scoring it wrong.
    flag_short_objective_for_review: bool = True
    short_objective_max_marks: float = 2.0   # only questions at/under this many marks
    short_objective_max_words: int = 4       # only expected answers this short

    # --- Retry ------------------------------------------------------------------
    max_retries: int = 3
    retry_base_delay_s: float = 2.0

    # --- Storage (optional) -----------------------------------------------------
    s3_bucket: str | None = None
    s3_region: str = "ap-south-1"
    enable_s3: bool = True  # if bucket unset, S3 steps no-op gracefully
    s3_upload_concurrency: int = 5

    # --- OpenRouter attribution headers -----------------------------------------
    site_url: str = "https://github.com/sharath-s-rao/topper-maker-evaluation-pipeline"
    site_name: str = "Topper Maker"

    @property
    def effective_llm_api_key(self) -> str | None:
        """The key to authenticate with, preferring the generic LLM_API_KEY."""
        return self.llm_api_key or self.openrouter_api_key

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
            llm_api_key=os.getenv("LLM_API_KEY"),
            htr_model=os.getenv("HTR_MODEL", cls.htr_model),
            vision_model=os.getenv("EVAL_VISION_MODEL", cls.vision_model),
            text_model=os.getenv("EVAL_TEXT_MODEL", cls.text_model),
            moderation_model=os.getenv("MODERATION_MODEL", cls.moderation_model),
            moderation_backend=os.getenv("MODERATION_BACKEND", cls.moderation_backend),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_moderation_model=os.getenv(
                "OPENAI_MODERATION_MODEL", cls.openai_moderation_model
            ),
            mapping_model=os.getenv("MAPPING_MODEL", cls.mapping_model),
            eval_models=_get_str_list("EVAL_MODELS", ["anthropic/claude-sonnet-4-5"]),
            supervisor_model=os.getenv("SUPERVISOR_MODEL") or None,
            supervisor_deviation_threshold=_get_float(
                "SUPERVISOR_DEVIATION_THRESHOLD", cls.supervisor_deviation_threshold
            ),
            use_mistral_ocr=_get_bool("USE_MISTRAL_OCR", cls.use_mistral_ocr),
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            mistral_ocr_model=os.getenv("MISTRAL_OCR_MODEL", cls.mistral_ocr_model),
            target_dpi=_get_int("TARGET_DPI", cls.target_dpi),
            htr_max_tokens=_get_opt_int("HTR_MAX_TOKENS", cls.htr_max_tokens),
            eval_max_tokens=_get_opt_int("EVAL_MAX_TOKENS", cls.eval_max_tokens),
            moderation_max_tokens=_get_opt_int(
                "MODERATION_MAX_TOKENS", cls.moderation_max_tokens
            ),
            apply_preprocessing=_get_bool("APPLY_PREPROCESSING", cls.apply_preprocessing),
            bbox_padding_frac=_get_float("BBOX_PADDING_FRAC", cls.bbox_padding_frac),
            min_diagram_area_frac=_get_float(
                "MIN_DIAGRAM_AREA_FRAC", cls.min_diagram_area_frac
            ),
            max_diagram_area_frac=_get_float(
                "MAX_DIAGRAM_AREA_FRAC", cls.max_diagram_area_frac
            ),
            max_concurrent_htr=_get_int("MAX_CONCURRENT_HTR", cls.max_concurrent_htr),
            max_concurrent_eval=_get_int("MAX_CONCURRENT_EVAL", cls.max_concurrent_eval),
            eval_batch_size=_get_int("EVAL_BATCH_SIZE", cls.eval_batch_size),
            max_file_size_mb=_get_float("MAX_FILE_SIZE_MB", cls.max_file_size_mb),
            max_pages=_get_int("MAX_ANSWER_SHEET_PAGES", cls.max_pages),
            low_confidence_review_threshold=_get_float(
                "LOW_CONFIDENCE_REVIEW_THRESHOLD", cls.low_confidence_review_threshold
            ),
            flag_short_objective_for_review=_get_bool(
                "FLAG_SHORT_OBJECTIVE_FOR_REVIEW", cls.flag_short_objective_for_review
            ),
            short_objective_max_marks=_get_float(
                "SHORT_OBJECTIVE_MAX_MARKS", cls.short_objective_max_marks
            ),
            short_objective_max_words=_get_int(
                "SHORT_OBJECTIVE_MAX_WORDS", cls.short_objective_max_words
            ),
            max_retries=_get_int("MAX_RETRIES", cls.max_retries),
            s3_bucket=os.getenv("AWS_S3_BUCKET_NAME"),
            s3_region=os.getenv("AWS_REGION", cls.s3_region),
            enable_s3=_get_bool("ENABLE_S3", cls.enable_s3),
        )
