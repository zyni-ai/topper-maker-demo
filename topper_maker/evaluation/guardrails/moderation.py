"""Content moderation for transcribed answer-sheet text.

Two interchangeable backends implement the same ``moderate(text) -> ModerationResult``
contract; pick one via ``EvaluationConfig.moderation_backend`` (env
``MODERATION_BACKEND``):

- ``openrouter`` (:class:`Moderator`): OpenRouter has no dedicated moderation
  endpoint, so we run an LLM-based classifier over the text and parse JSON. Costs
  tokens per booklet, splits text into ≤12 k-char chunks, and reasoning-heavy
  models can burn the token budget and trip the fail-closed path.
- ``openai`` (:class:`OpenAIModerator`): OpenAI's dedicated, free ``/moderations``
  endpoint (``omni-moderation-latest``). Purpose-built, returns structured category
  scores, no JSON-parsing/truncation failure mode. Needs ``OPENAI_API_KEY``.

Policy (both backends): **fail-closed**. If the moderation call errors out after
retries, the content is flagged (category ``moderation_error``) so nothing
inappropriate is silently processed. The pipeline then routes the booklet for human
review rather than rejecting it outright.

For exam answer sheets this is a safety net against abusive or inappropriate content
a student may have written.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from pydantic import BaseModel, Field

from topper_maker.evaluation.common.llm_client import OpenRouterClient

logger = logging.getLogger(__name__)

_MODERATION_CHAR_LIMIT = 12_000

# OpenAI's moderation endpoint accepts much larger inputs than a chat classifier and
# has no token-budget/JSON-truncation failure mode, so we chunk far more loosely and
# send every chunk in a single batched request.
_OPENAI_MODERATION_CHAR_LIMIT = 30_000

_SYSTEM_PROMPT = """You are a content-safety classifier for school exam answer sheets.
You are given text transcribed from a student's handwritten answers.

Flag the content ONLY if it contains material that is clearly inappropriate for a
school context: hate speech, explicit sexual content, graphic violence, self-harm
content, harassment, or threats. Ordinary exam answers — including incorrect ones,
crude diagrams, or frustration ("I don't know this") — are NOT violations.

Return a JSON object with:
- is_flagged: boolean
- categories: list of short category strings (empty if not flagged)
- scores: object mapping each flagged category to a confidence 0.0-1.0 (empty if not flagged)
"""

# Categories produced by the classifier are free-form short strings; these are the
# expected ones but the schema does not constrain them.
_CATEGORY_HINT = "hate, sexual, violence, self_harm, harassment, threat"


class _ModerationLLMResponse(BaseModel):
    is_flagged: bool = Field(...)
    categories: List[str] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)


@dataclass
class ModerationResult:
    is_flagged: bool
    categories: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    error: str | None = None


def _split_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of at most chunk_size characters on word boundaries."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Break at last whitespace within the window to avoid mid-word splits.
        split = text.rfind(" ", start, end)
        if split <= start:
            split = end
        chunks.append(text[start:split])
        start = split
    return chunks


class Moderator:
    """OpenRouter LLM-classifier, fail-closed content moderator.

    Long texts are split into ≤12 k-char chunks and each is moderated independently.
    Any flagged or errored chunk causes the whole sheet to be flagged (fail-closed).
    """

    def __init__(
        self, client: OpenRouterClient, model: str, max_tokens: int | None = None
    ) -> None:
        self._client = client
        self._model = model
        # None = no cap: reasoning models spend hidden reasoning tokens against this
        # budget, so a tight cap truncates the response before any JSON is emitted.
        self._max_tokens = max_tokens

    async def moderate(self, text: str) -> ModerationResult:
        if not text or not text.strip():
            return ModerationResult(is_flagged=False)

        chunks = _split_into_chunks(text, _MODERATION_CHAR_LIMIT)
        merged_categories: List[str] = []
        merged_scores: Dict[str, float] = {}
        errors: List[str] = []

        for idx, chunk in enumerate(chunks):
            label = f"moderation_chunk_{idx + 1}of{len(chunks)}"
            try:
                result = await self._client.complete_json(
                    model=self._model,
                    system_prompt=f"{_SYSTEM_PROMPT}\nCommon categories: {_CATEGORY_HINT}.",
                    user_content=f"Transcribed answer-sheet text:\n\n{chunk}",
                    schema_model=_ModerationLLMResponse,
                    max_tokens=self._max_tokens,
                    label=label,
                )
            except Exception as exc:  # noqa: BLE001 - fail-closed on any failure
                logger.error("Moderation chunk %d/%d failed; failing closed. %s", idx + 1, len(chunks), exc)
                errors.append(str(exc))
                continue

            if result.is_flagged:
                logger.warning("Content flagged in chunk %d/%d: %s", idx + 1, len(chunks), result.categories)
                merged_categories.extend(c for c in result.categories if c not in merged_categories)
                for cat, score in result.scores.items():
                    merged_scores[cat] = max(merged_scores.get(cat, 0.0), score)

        if errors:
            merged_categories = list(dict.fromkeys(["moderation_error"] + merged_categories))
            return ModerationResult(
                is_flagged=True,
                categories=merged_categories,
                scores=merged_scores,
                error="; ".join(errors),
            )

        return ModerationResult(
            is_flagged=bool(merged_categories),
            categories=merged_categories,
            scores=merged_scores,
        )

    async def aclose(self) -> None:
        """No-op: this backend shares the pipeline's :class:`OpenRouterClient`.

        The shared client is closed by its owner (the pipeline), so closing it here
        too would double-close it. The method exists only so callers can close any
        moderator uniformly (issue #138).
        """
        return None


class OpenAIModerator:
    """OpenAI ``/moderations`` (``omni-moderation-latest``), fail-closed moderator.

    Uses OpenAI's dedicated, free moderation endpoint instead of an LLM classifier:
    no token cost, structured category flags + scores, and no JSON-parse/truncation
    failure mode. All chunks of a sheet are sent in a single batched request; if the
    call errors the whole sheet is flagged (fail-closed), matching :class:`Moderator`.
    """

    def __init__(self, client: "AsyncOpenAI", model: str) -> None:  # noqa: F821
        self._client = client
        self._model = model

    @classmethod
    def from_api_key(cls, api_key: str, model: str) -> "OpenAIModerator":
        """Build an instance with a fresh ``AsyncOpenAI`` client (lazy SDK import)."""
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set; cannot use the OpenAI moderation backend."
            )
        from openai import AsyncOpenAI

        return cls(AsyncOpenAI(api_key=api_key), model)

    async def moderate(self, text: str) -> ModerationResult:
        if not text or not text.strip():
            return ModerationResult(is_flagged=False)

        chunks = _split_into_chunks(text, _OPENAI_MODERATION_CHAR_LIMIT)
        try:
            response = await self._client.moderations.create(
                model=self._model,
                input=chunks,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed on any failure
            logger.error("OpenAI moderation failed; failing closed. %s", exc)
            return ModerationResult(
                is_flagged=True,
                categories=["moderation_error"],
                error=str(exc),
            )

        merged_categories: List[str] = []
        merged_scores: Dict[str, float] = {}
        for result in response.results:
            if not getattr(result, "flagged", False):
                continue
            categories = _as_dict(result.categories)
            scores = _as_dict(result.category_scores)
            for category, is_on in categories.items():
                if not is_on:
                    continue
                if category not in merged_categories:
                    merged_categories.append(category)
                score = scores.get(category)
                if score is not None:
                    merged_scores[category] = max(merged_scores.get(category, 0.0), float(score))

        if merged_categories:
            logger.warning("Content flagged by OpenAI moderation: %s", merged_categories)

        return ModerationResult(
            is_flagged=bool(merged_categories),
            categories=merged_categories,
            scores=merged_scores,
        )

    async def aclose(self) -> None:
        """Close this backend's own ``AsyncOpenAI`` (and its httpx pool).

        Unlike :class:`Moderator`, this backend owns a separate client built in
        :meth:`from_api_key`, so it must be closed on the live loop to avoid the
        Windows ``Event loop is closed`` teardown error (issue #138). Defensive:
        a client without an async ``close`` (or a double close) is ignored.
        """
        close = getattr(self._client, "close", None)
        if close is None:
            return
        try:
            await close()
        except Exception as exc:  # noqa: BLE001 - teardown must never raise
            logger.debug("OpenAI moderation client close failed (ignored): %s", exc)


def _as_dict(obj) -> Dict[str, object]:
    """Normalise an OpenAI categories/scores object (Pydantic model) to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def build_moderator(config, openrouter_client: OpenRouterClient):
    """Construct the moderator selected by ``config.moderation_backend``.

    ``openrouter`` (default) → :class:`Moderator` (reuses the shared OpenRouter
    client). ``openai`` → :class:`OpenAIModerator` (needs ``OPENAI_API_KEY``). An
    unknown backend value raises rather than silently picking one.
    """
    backend = (getattr(config, "moderation_backend", "openrouter") or "openrouter").strip().lower()
    if backend == "openai":
        return OpenAIModerator.from_api_key(
            config.openai_api_key or "", config.openai_moderation_model
        )
    if backend == "openrouter":
        return Moderator(
            openrouter_client,
            config.moderation_model,
            max_tokens=getattr(config, "moderation_max_tokens", None),
        )
    raise ValueError(
        f"Unknown MODERATION_BACKEND={backend!r}; expected 'openai' or 'openrouter'."
    )
