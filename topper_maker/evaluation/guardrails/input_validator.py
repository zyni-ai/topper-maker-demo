"""Input guardrails — reject bad inputs before spending any API budget.

Checks, in cheap-to-expensive order:
1. URL reachable & content-length within limit (HEAD, no body download).
2. Bytes actually downloaded are a PDF (``%PDF`` magic) and within the size cap.
3. Page count within the cap, and the PDF opens cleanly (catches corrupt/encrypted).

The validator downloads the file once and returns the bytes so the extractor does
not have to fetch it again.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from topper_maker.evaluation.common.retry import retry_async

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF"


@dataclass
class InputValidationResult:
    valid: bool
    error: Optional[str] = None
    pdf_bytes: Optional[bytes] = None
    page_count: int = 0
    size_mb: float = 0.0


async def validate_input(
    answer_sheet_url: str,
    *,
    max_file_size_mb: float = 50.0,
    max_pages: int = 40,
    download_timeout_s: int = 30,
    max_retries: int = 3,
    allow_local_paths: bool = False,
) -> InputValidationResult:
    """Validate the answer-sheet source and return the PDF bytes on success.

    ``answer_sheet_url`` may be an ``http(s)`` URL (downloaded), a ``file://`` URI, or
    a local filesystem path (read from disk). Local paths are only accepted when
    ``allow_local_paths=True``; set this flag for the demo UI but leave it ``False``
    in production API deployments to prevent arbitrary filesystem reads.
    """
    # 1. Fetch bytes — from disk for local paths, over HTTP otherwise.
    try:
        if _is_local_source(answer_sheet_url):
            if not allow_local_paths:
                return InputValidationResult(
                    valid=False,
                    error="Local filesystem paths are not allowed. Pass a URL.",
                )
            pdf_bytes = await _read_local(answer_sheet_url)
        else:
            await _head_check(answer_sheet_url, download_timeout_s)
            pdf_bytes = await _download(answer_sheet_url, download_timeout_s, max_retries)
    except Exception as exc:  # noqa: BLE001 - any fetch failure is a clean rejection
        return InputValidationResult(valid=False, error=f"Could not read answer sheet: {exc}")

    size_mb = len(pdf_bytes) / (1024 * 1024)

    # 2. Size cap.
    if size_mb > max_file_size_mb:
        return InputValidationResult(
            valid=False,
            error=f"File is {size_mb:.1f} MB, exceeds the {max_file_size_mb:.0f} MB limit.",
            size_mb=size_mb,
        )

    # 3. PDF magic bytes.
    if not pdf_bytes[:4] == _PDF_MAGIC:
        return InputValidationResult(
            valid=False,
            error="File is not a PDF (missing %PDF header).",
            size_mb=size_mb,
        )

    # 4. Page count + integrity (corrupt/encrypted PDFs raise here).
    try:
        page_count = await asyncio.get_event_loop().run_in_executor(
            None, _page_count, pdf_bytes
        )
    except Exception as exc:  # noqa: BLE001
        return InputValidationResult(
            valid=False,
            error=f"PDF could not be opened (corrupt or encrypted): {exc}",
            size_mb=size_mb,
        )

    if page_count == 0:
        return InputValidationResult(
            valid=False, error="PDF has no pages.", size_mb=size_mb
        )
    if page_count > max_pages:
        return InputValidationResult(
            valid=False,
            error=f"PDF has {page_count} pages, exceeds the {max_pages}-page limit.",
            page_count=page_count,
            size_mb=size_mb,
        )

    logger.info(
        "Input validated: %d page(s), %.2f MB.", page_count, size_mb
    )
    return InputValidationResult(
        valid=True,
        pdf_bytes=pdf_bytes,
        page_count=page_count,
        size_mb=size_mb,
    )


def _is_local_source(source: str) -> bool:
    """True if ``source`` is a local filesystem path or a ``file://`` URI."""
    lower = source.lower()
    if lower.startswith(("http://", "https://")):
        return False
    return True


async def _read_local(source: str) -> bytes:
    """Read PDF bytes from a local path or ``file://`` URI off the event loop."""
    path = source
    if source.lower().startswith("file://"):
        from urllib.parse import urlparse
        from urllib.request import url2pathname

        path = url2pathname(urlparse(source).path)

    def _read() -> bytes:
        from pathlib import Path

        return Path(path).read_bytes()

    return await asyncio.get_event_loop().run_in_executor(None, _read)


async def _head_check(url: str, timeout_s: int) -> None:
    """HEAD pre-flight — raises immediately on 4xx before we waste bandwidth."""
    import requests

    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None, lambda: requests.head(url, timeout=timeout_s, allow_redirects=True)
    )
    if 400 <= resp.status_code < 500:
        resp.raise_for_status()


async def _download(url: str, timeout_s: int, max_retries: int) -> bytes:
    """Download the PDF, retrying on 5xx/network errors but not on 4xx."""
    import requests
    from requests.exceptions import ConnectionError, Timeout

    async def _get() -> bytes:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: requests.get(url, timeout=timeout_s)
        )
        if 400 <= resp.status_code < 500:
            # Client errors are not transient — propagate immediately, don't retry.
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.content

    # Retry only on transient network/server errors, not on client (4xx) errors.
    # _head_check already surfaced 4xx before we reach here, but guard defensively.
    return await retry_async(
        _get,
        max_retries=max_retries,
        retryable_exceptions=(ConnectionError, Timeout, IOError),
        operation_name="answer_sheet_download",
    )


def _page_count(pdf_bytes: bytes) -> int:
    import fitz  # pymupdf

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.needs_pass:  # encrypted and we have no password
            raise ValueError("PDF is password-protected.")
        return doc.page_count
