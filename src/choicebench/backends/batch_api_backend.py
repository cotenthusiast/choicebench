# src/choicebench/backends/batch_api_backend.py

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Awaitable, Callable

from choicebench.backends.api_backend import APIBackend
from choicebench.clients.base import BaseClient
from choicebench.clients.types import (
    BATCH_COMPLETED,
    BATCH_FAILED,
    ErrorInfo,
    FAILURE_STATUS,
    ModelRequest,
    ModelResponse,
)
from choicebench.infra.cache import CachingClientWrapper, _cache_key


class BatchAPIBackend(APIBackend):
    """APIBackend variant backed by a provider's true Batch API instead of
    concurrent per-request dispatch.

    Implements the exact same generate_batch(prompts) -> list[ModelResponse]
    contract as APIBackend, so every runner that already has a working
    run_many_async() path (it calls backend.generate_batch()) gains batch
    execution for free by swapping the backend -- zero runner-level changes.

    The underlying provider client must additionally implement:
      - async submit_batch(requests: list[ModelRequest]) -> str (a batch id)
      - async poll_batch(batch_id: str) -> str (one of
        clients.types.BATCH_IN_PROGRESS/BATCH_COMPLETED/BATCH_FAILED)
      - async fetch_batch_results(batch_id, requests) -> list[ModelResponse],
        reassembled into the SAME order as `requests` (provider batch APIs
        return results in arbitrary order -- reassembly is the client's job,
        via each request's custom_id)

    Resumable across a process restart: before submitting, the exact set of
    uncached requests is content-hashed to a batch-state file under
    batch_state_dir. A restart recomputes the same hash, finds the existing
    file, and polls the SAME batch_id instead of submitting a duplicate job.
    The state file is deleted once the batch reaches either terminal
    outcome (success or failure), so a genuinely new prompt set never
    collides and a retry after failure submits fresh.

    Caching integrates the same way APIBackend's does: a request already in
    the response cache is never included in the batch at all, and a
    successful batch result is written to the same cache other backends
    share, on the same content-addressed key.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        client: BaseClient,
        cache_dir: Path,
        temperature: float,
        max_tokens: int,
        seed: int,
        batch_state_dir: Path,
        concurrency_limit: int = 10,
        cache_identity: str | None = None,
        poll_interval_seconds: float = 30.0,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(
            provider=provider, model_name=model_name, client=client, cache_dir=cache_dir,
            temperature=temperature, max_tokens=max_tokens, seed=seed,
            concurrency_limit=concurrency_limit, cache_identity=cache_identity,
        )
        self._batch_state_dir = Path(batch_state_dir)
        self._batch_state_dir.mkdir(parents=True, exist_ok=True)
        self._poll_interval_seconds = poll_interval_seconds
        self._sleep_fn = sleep_fn

    async def generate_batch(self, prompts: list[str]) -> list[ModelResponse]:
        """Execute all prompts via one provider Batch API job (cache misses
        only), returned in the same order the prompts were given."""
        requests = [self._make_request(p) for p in prompts]
        cache = self._client._cache

        results: list[ModelResponse | None] = [None] * len(prompts)
        uncached_indices: list[int] = []
        for i, request in enumerate(requests):
            cached = cache.get(_cache_key(request))
            if cached is not None:
                results[i] = CachingClientWrapper._build_cached_response(request, cached)
            else:
                uncached_indices.append(i)

        if uncached_indices:
            uncached_requests = [requests[i] for i in uncached_indices]
            batch_id = await self._resolve_or_submit_batch(uncached_requests)
            fetched = await self._poll_until_terminal(batch_id, uncached_requests)
            for local_i, response in zip(uncached_indices, fetched):
                results[local_i] = response
                if response.is_success():
                    cache.put(
                        _cache_key(requests[local_i]),
                        CachingClientWrapper._serialize_response(response),
                    )
            self._clear_batch_state(uncached_requests)

        return results

    async def _resolve_or_submit_batch(self, requests: list[ModelRequest]) -> str:
        state_path = self._batch_state_path(requests)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            return state["batch_id"]
        batch_id = await self._raw_client.submit_batch(requests)
        state_path.write_text(json.dumps({
            "batch_id": batch_id,
            "provider": self._provider,
            "model_name": self._model_name,
            "n_requests": len(requests),
        }))
        return batch_id

    async def _poll_until_terminal(
        self, batch_id: str, requests: list[ModelRequest],
    ) -> list[ModelResponse]:
        while True:
            status = await self._raw_client.poll_batch(batch_id)
            if status == BATCH_COMPLETED:
                return await self._raw_client.fetch_batch_results(batch_id, requests)
            if status == BATCH_FAILED:
                return [
                    ModelResponse(
                        provider=request.provider,
                        model_name=request.model_name,
                        status=FAILURE_STATUS,
                        latency_seconds=0.0,
                        error=ErrorInfo(
                            "BatchJobFailed",
                            f"Batch job {batch_id} did not complete successfully.",
                            True,
                            "batch_submission",
                        ),
                    )
                    for request in requests
                ]
            await self._sleep_fn(self._poll_interval_seconds)

    def _batch_state_path(self, requests: list[ModelRequest]) -> Path:
        keys = sorted(_cache_key(request) for request in requests)
        fingerprint = json.dumps(
            {"provider": self._provider, "model_name": self._model_name, "request_keys": keys},
            sort_keys=True,
        )
        state_key = hashlib.sha256(fingerprint.encode()).hexdigest()
        return self._batch_state_dir / f"{state_key}.json"

    def _clear_batch_state(self, requests: list[ModelRequest]) -> None:
        self._batch_state_path(requests).unlink(missing_ok=True)
