"""Async OpenRouter LLM client with JSON-structured output and vision support.

Design decisions
----------------
- **OpenRouter only** (issue #8). One key, the OpenAI-compatible SDK, free model
  choice. Data-retention/training is disabled at the OpenRouter account level.
- **Structured output via prompt + parse + validate + retry**, rather than native
  ``response_format`` / tool-calling. This is the most portable approach across the
  many models OpenRouter fronts (native JSON modes vary by provider). We strongly
  instruct JSON, strip code fences, parse, and validate against a Pydantic model;
  on failure we re-prompt with the validation error, up to ``max_retries``.
- **Typed surface**: ``complete_json`` returns a validated Pydantic instance so
  callers never touch raw strings.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from pydantic import BaseModel, ValidationError

from topper_maker.evaluation.common.retry import retry_async
from topper_maker.evaluation.common.usage import UsageTracker

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TModel = TypeVar("TModel", bound=BaseModel)


def _is_openrouter(base_url: str) -> bool:
    """True if the endpoint is OpenRouter (used to gate its attribution headers)."""
    return "openrouter.ai" in base_url

# Transient API errors worth retrying. Restricted to connection/rate/server failures
# so permanent errors (401, 400, context-length) do not waste 3 retries × 4 attempts.
def _retryable_exceptions():
    try:
        import openai
        return (
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
    except ImportError:
        return (Exception,)

_RETRYABLE = _retryable_exceptions()


# Matches EITHER a valid JSON escape (kept as-is) OR a lone backslash (doubled).
# Listing valid escapes first means an already-escaped "\\" is consumed as one token,
# so a following letter (e.g. "\\section") is not mistaken for a stray backslash.
_JSON_ESCAPE_OR_BACKSLASH = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')


def _repair_json_escapes(text: str) -> str:
    """Double stray backslashes so ``json.loads`` accepts unescaped LaTeX/math.

    Models often emit ``\\sqrt``, ``\\alpha``, ``\\Delta`` inside JSON strings without
    escaping the backslash — invalid JSON ("Invalid \\escape"). Valid escapes (``\\n``,
    ``\\"``, ``\\\\``, ``\\uXXXX`` …) are preserved; only backslashes that don't start
    one are doubled. (A LaTeX command that collides with a JSON escape letter — ``\\t``
    in ``\\times``, ``\\f`` in ``\\frac`` — is still read as the control char; that is an
    inherent JSON ambiguity, but it no longer crashes the whole batch.)
    """
    return _JSON_ESCAPE_OR_BACKSLASH.sub(
        lambda m: m.group(0) if len(m.group(0)) > 1 else r"\\", text
    )


# Detects control chars that JSON interpretes from a valid escape (\\t → TAB, \\f → FF,
# \\b → BS, \\r → CR, \\n → LF) when immediately followed by lowercase letters —
# the signature of a LaTeX command like \\theta, \\frac, \\beta, \\nabla, \\rho.
_CTRL_PLUS_LETTER = re.compile(r"[\t\f\b\r\n](?=[a-z])")

# In raw JSON text: doubles the backslash before t/f/b/r/n when followed by [a-z].
# \\theta → \\\\theta (so json.loads gives \\theta instead of <TAB>heta).
_LATEX_COLLISION_RE = re.compile(r"\\([tfbnr])(?=[a-z])")


def _has_latex_corruption(obj) -> bool:
    """Walk parsed JSON; True if any string has a control char immediately before a letter."""
    if isinstance(obj, str):
        return bool(_CTRL_PLUS_LETTER.search(obj))
    if isinstance(obj, dict):
        return any(_has_latex_corruption(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_latex_corruption(v) for v in obj)
    return False


def _repair_latex_collisions(text: str) -> str:
    """Double backslash before t/f/b/r/n in raw JSON text when followed by a letter."""
    return _LATEX_COLLISION_RE.sub(r"\\\\\1", text)


def _recover_partial_json(text: str) -> dict | None:
    """Extract complete JSON objects from a truncated response.

    When a model hits its output token limit mid-array, the raw content is valid JSON
    up to some point then abruptly stops. This is not always a small hard cap: even with a
    large output ceiling we still observe ``finish_reason=length`` on whole-booklet
    HTR/mapping calls, because a single-shot JSON over an 11-page booklet (plus the model's
    own reasoning tokens) exhausts even a large ``max_tokens`` budget. The recovery below
    applies whenever that happens. This scanner
    walks the text character-by-character, tracking string context so LaTeX curly braces
    inside string values don't confuse the bracket counter, and collects every top-level
    value that closed cleanly before the truncation. Returns ``{"key": [...complete...]}``
    for any array-valued key found, or None if nothing could be recovered.
    """
    # Find the outermost object's key and array value
    m = re.search(r'"(\w+)"\s*:\s*\[', text)
    if not m:
        return None

    array_key = m.group(1)
    pos = m.end()
    complete_objects: list = []

    while pos < len(text):
        # Skip whitespace and commas between elements
        while pos < len(text) and text[pos] in ' \t\n\r,':
            pos += 1
        if pos >= len(text) or text[pos] != '{':
            break

        # Walk forward to find the matching closing brace, respecting strings
        depth = 0
        in_string = False
        escape_next = False
        start = pos
        end = None
        for i in range(pos, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end is None:
            break  # object was not closed — truncated here

        obj_str = text[start:end]
        for candidate in (obj_str, _repair_json_escapes(obj_str)):
            try:
                complete_objects.append(json.loads(candidate))
                break
            except json.JSONDecodeError:
                pass
        pos = end

    if not complete_objects:
        return None
    return {array_key: complete_objects}


class LLMError(RuntimeError):
    """Raised when an LLM call cannot produce valid output after all retries."""


class OpenRouterClient:
    """Thin async wrapper over the OpenAI SDK pointed at OpenRouter."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _OPENROUTER_BASE_URL,
        site_url: str = "",
        site_name: str = "",
        max_retries: int = 3,
        base_delay: float = 2.0,
        usage_tracker: "UsageTracker | None" = None,
    ) -> None:
        if not api_key:
            raise LLMError(
                "No LLM API key set (LLM_API_KEY / OPENROUTER_API_KEY); cannot create client."
            )
        # Imported lazily so importing this module never requires the SDK/credentials.
        from openai import AsyncOpenAI

        # HTTP-Referer / X-Title are OpenRouter-specific attribution headers; only send
        # them when actually targeting OpenRouter so other providers (e.g. NVIDIA NIM)
        # aren't handed headers they don't use.
        is_openrouter = _is_openrouter(base_url)
        default_headers = (
            {"HTTP-Referer": site_url, "X-Title": site_name} if is_openrouter else {}
        )
        # The `usage: {include: true}` request param (cost accounting) is also
        # OpenRouter-specific; other providers reject it as an unsupported parameter.
        self._send_openrouter_usage = is_openrouter
        # wrap_openai traces each call as a nested LLM run with the prompt, any
        # image content, the raw response, and token usage / cost.
        self._client = wrap_openai(
            AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers=default_headers,
            )
        )
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._usage_tracker = usage_tracker

    # -- Public API --------------------------------------------------------------

    @traceable(run_type="chain")
    async def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: str | List[Dict[str, Any]],
        schema_model: Type[TModel],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        label: str = "llm_json",
    ) -> TModel:
        """Call the model and return a validated instance of ``schema_model``.

        ``user_content`` may be a plain string or a list of multimodal content
        blocks (text + image_url) for vision calls.

        Raises:
            LLMError: if no valid JSON matching the schema is produced after retries.
        """
        schema_hint = self._schema_instruction(schema_model)
        base_system = f"{system_prompt}\n\n{schema_hint}"

        async def _attempt(correction: str = "") -> TModel:
            user = self._with_correction(user_content, correction)
            raw = await self._raw_completion(
                model=model,
                system_prompt=base_system,
                user_content=user,
                max_tokens=max_tokens,
                temperature=temperature,
                label=label,
            )
            payload = self._extract_json(raw)
            return schema_model.model_validate(payload)

        # First try (with its own network-level retries), then up to two
        # self-correction rounds where we feed the validation error back in.
        correction = ""
        last_err: Exception | None = None
        for round_idx in range(3):
            try:
                return await _attempt(correction)
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_err = exc
                logger.warning(
                    "%s: invalid JSON on round %d: %s", label, round_idx + 1, exc
                )
                correction = (
                    "\n\nYour previous response was not valid JSON matching the schema. "
                    f"Error: {exc}. Return ONLY the corrected JSON object."
                )
        raise LLMError(f"{label}: failed to obtain valid structured output: {last_err}")

    async def aclose(self) -> None:
        """Close the underlying ``AsyncOpenAI`` (and its httpx connection pool).

        Must be awaited while the event loop that opened the connections is still
        running. ``evaluate_sync`` runs each evaluation on a throwaway loop via
        ``asyncio.run``; without an explicit close the httpx pool is only torn down
        later, by the garbage collector, on an already-closed loop — which on the
        Windows Proactor loop raises ``RuntimeError('Event loop is closed')`` from a
        fire-and-forget finalizer Task (issue #138). Closing here drains the pool on
        the live loop so GC has nothing left to tear down. Idempotent and defensive:
        a client without an async ``close`` (or a double close) is ignored.
        """
        close = getattr(self._client, "close", None)
        if close is None:
            return
        try:
            await close()
        except Exception as exc:  # noqa: BLE001 - teardown must never raise
            logger.debug("OpenRouter client close failed (ignored): %s", exc)

    @traceable(run_type="chain")
    async def complete_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: str | List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        label: str = "llm_text",
    ) -> str:
        """Return the raw text completion (used for lightweight classification)."""
        return await self._raw_completion(
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=temperature,
            label=label,
        )

    # -- Internals ---------------------------------------------------------------

    @traceable(run_type="chain")
    async def _raw_completion(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: str | List[Dict[str, Any]],
        max_tokens: Optional[int],
        temperature: float,
        label: str,
    ) -> str:
        async def _call() -> Any:
            # max_tokens is omitted entirely when None so the model is bounded only by
            # its own remaining-context window. A forced cap was starving reasoning
            # models: their hidden reasoning trace counts against max_tokens but is not
            # returned in the content, so a low cap truncates the response before any
            # JSON is emitted (finish_reason=length, empty body). Letting the provider
            # default to full context is the fix; callers may still pass a cap to bound
            # cost. See docs/discussion_max_tokens_truncation.md.
            kwargs: Dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            # Ask OpenRouter to return token + cost accounting; other providers reject
            # this param, so only send it to OpenRouter.
            if self._send_openrouter_usage:
                kwargs["extra_body"] = {"usage": {"include": True}}
            return await self._client.chat.completions.create(**kwargs)

        response = await retry_async(
            _call,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            retryable_exceptions=_RETRYABLE,
            operation_name=label,
        )
        if self._usage_tracker is not None:
            self._usage_tracker.record_response(response, model)
        choice = response.choices[0]
        content = choice.message.content or ""
        if choice.finish_reason == "length":
            logger.warning(
                "%s: response truncated (finish_reason=length) — attempting partial recovery.",
                label,
            )
        return content

    @staticmethod
    def _with_correction(
        user_content: str | List[Dict[str, Any]], correction: str
    ) -> str | List[Dict[str, Any]]:
        if not correction:
            return user_content
        if isinstance(user_content, str):
            return user_content + correction
        return [*user_content, {"type": "text", "text": correction}]

    @staticmethod
    def _schema_instruction(schema_model: Type[BaseModel]) -> str:
        schema = json.dumps(schema_model.model_json_schema(), indent=2)
        return (
            "Return ONLY a single valid JSON object that conforms to this JSON Schema. "
            "No markdown fences, no prose before or after.\n"
            f"JSON Schema:\n{schema}"
        )

    @staticmethod
    def _extract_json(raw: str) -> Dict[str, Any]:
        """Strip fences and parse the first JSON object in the text.

        Models routinely emit math/LaTeX (e.g. ``\\sqrt``, ``\\alpha``, ``\\Delta``)
        inside JSON string values without escaping the backslash, which is invalid
        JSON. Each candidate is therefore retried with such stray escapes repaired
        before we give up.
        """
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        # Try the full text first, then fall back to the outermost {...} span.
        candidates = [cleaned]
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            candidates.append(match.group(0))

        last_err: json.JSONDecodeError | None = None
        for candidate in candidates:
            # For each candidate, try as-is, then with stray backslash escapes repaired.
            for text in (candidate, _repair_json_escapes(candidate)):
                try:
                    parsed = json.loads(text)
                    # #47: a successful parse can silently corrupt LaTeX commands whose
                    # leading letters collide with JSON escape sequences (\\t→TAB,
                    # \\f→FF, \\b→BS, \\r→CR, \\n→LF). Detect and re-parse.
                    if _has_latex_corruption(parsed):
                        logger.warning(
                            "Detected possible LaTeX/JSON-escape collision in response "
                            "(e.g. \\\\theta parsed as TAB+heta); applying targeted repair."
                        )
                        try:
                            return json.loads(_repair_latex_collisions(text))
                        except json.JSONDecodeError:
                            pass  # repaired text also invalid — return original parse
                    return parsed
                except json.JSONDecodeError as exc:
                    last_err = exc

        # Truncation recovery: scan for complete JSON objects within a partially-written
        # array. Handles a model that hits its output token limit mid-response.
        recovered = _recover_partial_json(cleaned)
        if recovered is not None:
            logger.info("Recovered partial JSON from truncated response (%d top-level key(s)).", len(recovered))
            return recovered

        assert last_err is not None  # candidates is non-empty, so a parse was attempted
        raise last_err
