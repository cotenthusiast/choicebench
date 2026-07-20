# experiments/visible_llm_matcher/repairs/group1_historical_arc/harness/local_backend_adapter.py
#
# Adapts choicebench.backends.hf_backend.HuggingFaceBackend (synchronous,
# returns a plain str, no caching layer) to the async
# generate_single_async(prompt) -> ModelResponse interface every repair_one_*
# function in this harness already calls (built against APIBackend). This
# keeps baseline_repair.py/cyclic_repair.py/text_extraction_repair.py/
# two_stage_repair.py backend-agnostic — they never import HuggingFaceBackend
# or APIBackend directly, only call backend.generate_single_async().
#
# No caching layer exists for local inference (HuggingFaceBackend has none,
# by design — every call is a genuine forward pass), so there is no
# "accidental cache hit" risk to guard against here the way there is for API
# repairs; assert_fresh_call's latency>0 sanity check is still applied as a
# basic "did this actually run" guard.

from __future__ import annotations

import asyncio
import time

from choicebench.backends.hf_backend import HuggingFaceBackend
from choicebench.clients.types import (
    FAILURE_STATUS,
    SUCCESS_STATUS,
    ErrorInfo,
    ModelResponse,
)
from choicebench.identity import redact_text


class LocalBackendAsyncAdapter:
    """Wraps a loaded HuggingFaceBackend so harness repair functions can
    call it exactly like an APIBackend."""

    def __init__(self, backend: HuggingFaceBackend) -> None:
        self._backend = backend

    @property
    def provider(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    async def generate_single_async(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        loop = asyncio.get_event_loop()
        try:
            raw_text = await loop.run_in_executor(None, self._backend.generate, prompt)
        except Exception as exc:  # noqa: BLE001 - surfaced as a failure ModelResponse, not raised
            return ModelResponse(
                provider=self.provider,
                model_name=self.model_name,
                status=FAILURE_STATUS,
                latency_seconds=time.perf_counter() - start,
                raw_text=None,
                error=ErrorInfo(
                    error_type=type(exc).__name__,
                    message=redact_text(exc) if str(exc) else "Unexpected exception with no message.",
                    retryable=False,
                    stage="generation",
                ),
            )
        return ModelResponse(
            provider=self.provider,
            model_name=self.model_name,
            status=SUCCESS_STATUS,
            latency_seconds=time.perf_counter() - start,
            raw_text=raw_text,
        )
