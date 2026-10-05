"""S3 persistence — optional and non-fatal.

Uploads two artefacts, mirroring the reference engine's layout:
1. ``answer_sheet.json`` — extracted text + moderation result (audit trail).
2. Student diagram images, each under a fresh UUID, returning an
   ``{global_index: "uuid.ext"}`` map used to populate ``user_answer_image_codes``.

S3 is best-effort: if the bucket is unconfigured or boto3 is missing, methods log a
warning and return empty results so the pipeline still completes. boto3 is imported
lazily so it is not a hard dependency for non-S3 deployments.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from topper_maker.evaluation.common.concurrency import gather_bounded
from topper_maker.evaluation.common.retry import retry_async
from topper_maker.evaluation.config import EvaluationConfig

logger = logging.getLogger(__name__)


class S3Uploader:
    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config
        self._enabled = bool(config.enable_s3 and config.s3_bucket)
        if config.enable_s3 and not config.s3_bucket:
            logger.warning("S3 enabled but AWS_S3_BUCKET_NAME is unset; uploads will no-op.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _safe(value: str) -> str:
        """Normalize an untrusted string for use as an S3 key segment."""
        return re.sub(r"[^A-Za-z0-9_\-]", "_", value)

    def _base_path(self, req) -> str:
        return (
            f"academic/evaluation/{self._safe(req.class_type)}/{self._safe(req.skill)}/"
            f"submissions/{self._safe(req.user_test_id)}"
        )

    async def upload_analysis(self, req, extracted_text: str, moderation: dict) -> Optional[str]:
        """Upload the extraction + moderation analysis JSON; return its URL or None."""
        if not self._enabled:
            return None
        try:
            client = self._client()
            key = f"{self._base_path(req)}/answer_sheet.json"
            body = json.dumps(
                {
                    "user_test_id": req.user_test_id,
                    "user_id": req.user_id,
                    "extracted_text": extracted_text,
                    "moderation": moderation,
                },
                indent=2,
            ).encode("utf-8")
            await retry_async(
                self._put,
                client,
                key,
                body,
                "application/json",
                max_retries=self._config.max_retries,
                operation_name="s3_upload_analysis",
            )
            return self._url(key)
        except Exception as exc:  # noqa: BLE001 - non-fatal
            logger.warning("Analysis upload failed (continuing): %s", exc)
            return None

    async def upload_images(self, req, images: List[Dict[str, Any]]) -> Dict[int, str]:
        """Upload each image under a UUID concurrently; return {global_index: 'uuid.ext'}."""
        index_to_uuid: Dict[int, str] = {}
        if not self._enabled or not images:
            return index_to_uuid
        try:
            client = self._client()
        except Exception as exc:  # noqa: BLE001
            logger.warning("S3 client unavailable (continuing without image upload): %s", exc)
            return index_to_uuid

        base = f"{self._base_path(req)}/images"

        async def _upload_one(img: Dict[str, Any]) -> Optional[tuple]:
            gidx = img.get("global_index")
            b64 = img.get("base64", "")
            if gidx is None or not b64:
                return None
            try:
                ext = "jpg"
                img_uuid = str(uuid.uuid4())
                key = f"{base}/{img_uuid}.{ext}"
                await retry_async(
                    self._put,
                    client,
                    key,
                    base64.b64decode(b64),
                    "image/jpeg",
                    max_retries=self._config.max_retries,
                    operation_name=f"s3_upload_image_{gidx}",
                )
                return gidx, f"{img_uuid}.{ext}"
            except Exception as exc:  # noqa: BLE001 - skip the bad image, keep going
                logger.warning("Image upload failed for index %s: %s", gidx, exc)
                return None

        results = await gather_bounded(
            [lambda i=img: _upload_one(i) for img in images],
            limit=self._config.s3_upload_concurrency,
        )
        for r in results:
            if r is not None:
                gidx, code = r
                index_to_uuid[gidx] = code
        logger.info("Uploaded %d/%d images to S3.", len(index_to_uuid), len(images))
        return index_to_uuid

    # -- internals ---------------------------------------------------------------

    def _client(self):
        import boto3

        return boto3.client("s3", region_name=self._config.s3_region)

    async def _put(self, client, key: str, body: bytes, content_type: str) -> None:
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: client.put_object(
                Bucket=self._config.s3_bucket, Key=key, Body=body, ContentType=content_type
            ),
        )

    def _url(self, key: str) -> str:
        return f"https://{self._config.s3_bucket}.s3.{self._config.s3_region}.amazonaws.com/{key}"
