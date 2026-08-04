"""Disk-backed response cache for model API calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from choicebench.identity import integrity_digest
from choicebench.infra.atomic_io import atomic_write_json

from choicebench.clients.types import (
    ModelRequest,
    ModelResponse,
    UsageInfo,
    SUCCESS_STATUS,
)

logger = logging.getLogger(__name__)


def _cache_key(request: ModelRequest) -> str:
    """Compute a stable SHA-256 cache key for a model request.

    Only deterministic generation parameters are included — not trace
    metadata (question_id, run_id, etc.), which vary per request but
    don't affect the model output.
    """
    key_data: dict = {
        "provider": request.provider,
        "model_name": request.model_name,
        "prompt": request.payload,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "seed": request.seed,
        "request_logprobs": request.request_logprobs,
    }
    fingerprint = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(fingerprint.encode()).hexdigest()


class ResponseCache:
    """Disk-backed JSON cache for successful model responses.

    Files are stored as {cache_dir}/{key[:2]}/{key}.json — a two-level
    directory structure that avoids excessive files in a single directory
    for large caches.
    """

    def __init__(self, cache_dir: Path, namespace: str | None = None) -> None:
        self._dir = cache_dir
        self._namespace = namespace or Path(cache_dir).name
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """Return cached payload dict, or None on miss."""
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with p.open() as f:
                record = json.load(f)
            if record.get("schema_version") != "choicebench.response-cache.v1":
                return None
            if record.get("namespace") != self._namespace:
                return None
            payload = record.get("payload")
            expected = integrity_digest({"namespace": self._namespace, "cache_key": key, "payload": payload})
            if record.get("payload_digest") != expected:
                logger.warning("Ignoring corrupt response cache entry %s", p)
                return None
            return payload
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, payload: dict) -> None:
        """Write payload dict to cache atomically."""
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            atomic_write_json(p, {
                "schema_version": "choicebench.response-cache.v1",
                "namespace": self._namespace,
                "payload": payload,
                "payload_digest": integrity_digest({
                    "namespace": self._namespace, "cache_key": key, "payload": payload,
                }),
            })
        except OSError as exc:
            logger.warning("Cache write failed for key %s: %s", key[:8], exc)


class CachingClientWrapper:
    """Wraps a BaseClient to transparently cache successful responses.

    Cache hits return immediately without making an API call. Only
    successful (status=SUCCESS) responses are written to the cache.

    The wrapper exposes ``provider``, ``model_name``, ``generate``, and
    ``generate_batch`` — the full interface that runners expect on a client.

    Args:
        client: The underlying provider client.
        cache:  A ResponseCache instance.
    """

    def __init__(self, client, cache: ResponseCache) -> None:
        self._client = client
        self._cache = cache
        self.provider = client.provider
        self.model_name = client.model_name

    async def generate(self, request: ModelRequest) -> ModelResponse:
        key = _cache_key(request)
        cached = self._cache.get(key)

        if cached is not None:
            logger.debug(
                "Cache hit for %s (key=%s…)", request.model_name, key[:8]
            )
            return self._build_cached_response(request, cached)

        response = await self._client.generate(request)

        if response.is_success():
            self._cache.put(key, self._serialize_response(response))

        return response

    async def generate_batch(
        self, requests: list[ModelRequest]
    ) -> list[ModelResponse]:
        return list(
            await asyncio.gather(*[self.generate(r) for r in requests])
        )

    @staticmethod
    def _serialize_response(response: ModelResponse) -> dict:
        usage = None
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        payload: dict[str, object] = {
            "raw_text": response.raw_text,
            "finish_reason": response.finish_reason,
            "usage": usage,
            "logprobs": response.logprobs,
        }
        return payload

    @staticmethod
    def _build_cached_response(
        request: ModelRequest, payload: dict
    ) -> ModelResponse:
        usage = None
        if payload.get("usage"):
            u = payload["usage"]
            usage = UsageInfo(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )
        return ModelResponse(
            provider=request.provider,
            model_name=request.model_name,
            status=SUCCESS_STATUS,
            latency_seconds=0.0,
            raw_text=payload["raw_text"],
            finish_reason=payload.get("finish_reason"),
            usage=usage,
            error=None,
            timestamp_utc=None,
            logprobs=payload.get("logprobs"),
        )
