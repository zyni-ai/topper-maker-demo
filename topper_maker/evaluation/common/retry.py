"""Async retry with exponential backoff.

Ported from the reference evaluation engine and kept dependency-free so it can
wrap any awaitable (LLM calls, HTTP downloads, S3 puts).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Tuple, Type

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 2.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_EXPONENTIAL_BASE = 2.0


async def retry_async(
    operation: Callable[..., Awaitable[Any]],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    operation_name: str = "operation",
    **kwargs: Any,
) -> Any:
    """Run ``operation`` with exponential backoff, re-raising the last error.

    Args:
        operation: An async callable.
        max_retries: Number of retries *after* the first attempt.
        base_delay: Initial backoff in seconds, doubled each attempt (capped at max_delay).
        retryable_exceptions: Exceptions that trigger a retry; others propagate immediately.
        operation_name: Label for log lines.

    Returns:
        Whatever ``operation`` returns.

    Raises:
        The last exception if every attempt fails.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await operation(*args, **kwargs)
        except retryable_exceptions as exc:  # noqa: PERF203 - retry loop is intentional
            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (DEFAULT_EXPONENTIAL_BASE**attempt), max_delay)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    operation_name,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "All %d attempts failed for %s: %s",
                    max_retries + 1,
                    operation_name,
                    exc,
                )
    assert last_exc is not None  # for type-checkers; loop always sets it on failure
    raise last_exc
