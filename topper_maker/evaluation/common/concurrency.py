"""Bounded-concurrency helper for fanning out async work.

The pipeline transcribes pages and evaluates questions in parallel, but we must
cap in-flight LLM calls to stay within provider rate limits and keep memory
bounded. ``gather_bounded`` runs a list of coroutine factories with a semaphore.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List, TypeVar

T = TypeVar("T")


async def gather_bounded(
    factories: List[Callable[[], Awaitable[T]]],
    limit: int,
) -> List[T]:
    """Run coroutine factories concurrently, at most ``limit`` at a time.

    Args:
        factories: Zero-arg callables each returning a fresh awaitable. Factories
            (not bare coroutines) are used so nothing starts until a slot is free.
        limit: Maximum number of concurrent tasks (clamped to >= 1).

    Returns:
        Results in the same order as ``factories``. Exceptions propagate (the
        caller decides whether to treat per-item failure as fatal).
    """
    if not factories:
        return []
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _run(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    return await asyncio.gather(*[_run(f) for f in factories])
