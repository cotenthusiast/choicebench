# src/mcq_eval/clients/base.py

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod

from mcq_eval.clients.types import (
    ValidationError,
    ProviderCallError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    FAILURE_STATUS,
    ErrorInfo,
    ModelRequest,
    ModelResponse,
)
from mcq_eval.config.models import MAX_RETRIES, TIMEOUT

logger = logging.getLogger(__name__)

# Exponential backoff constants (seconds)
_BACKOFF_BASE = 5.0
_BACKOFF_CAP = 300.0


class BaseClient(ABC):
    """Abstract base client for provider-backed model calls.

    Handles request validation, bounded concurrency, per-call minimum delay,
    and exponential-backoff retry on transient errors. Concrete subclasses
    implement only the provider-specific raw call.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        timeout: int = TIMEOUT,
        concurrency_limit: int = 10,
        max_retries: int = MAX_RETRIES,
        min_delay_seconds: float = 0.0,
    ) -> None:
        """Initialize shared client configuration.

        Args:
            provider:            Provider name string.
            model_name:          Exact model name string.
            timeout:             Per-request timeout in seconds.
            concurrency_limit:   Max in-flight requests at once.
            max_retries:         Retry attempts on transient errors.
            min_delay_seconds:   Minimum gap between consecutive calls to
                                 this model. Proactively spaces requests to
                                 stay under per-minute rate limits.
        """
        self.provider = provider
        self.model_name = model_name
        self.timeout = timeout
        self.concurrency_limit = concurrency_limit
        self.max_retries = max_retries
        self.min_delay_seconds = min_delay_seconds

        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self._delay_lock = asyncio.Lock()
        self._last_call_time: float = 0.0

    async def _enforce_min_delay(self) -> None:
        """Sleep if necessary to maintain the configured minimum call gap.

        Uses a lock so that concurrent coroutines are serialized through
        the spacing check, ensuring at most 1/min_delay_seconds calls/sec.
        """
        if self.min_delay_seconds <= 0.0:
            return
        async with self._delay_lock:
            now = asyncio.get_event_loop().time()
            wait = self.min_delay_seconds - (now - self._last_call_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_time = asyncio.get_event_loop().time()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Execute one standardized model request with retry and backoff.

        Validation is not retried. Transient provider errors (rate limits,
        timeouts, generic call errors) are retried up to ``max_retries``
        times with exponential backoff and ±25% jitter.

        Args:
            request: Standardized model request to execute.

        Returns:
            A standardized model response (success or structured failure).
        """
        start_time = time.perf_counter()

        # Validation runs once — no retry.
        try:
            request.validate()
            self._validate_request_compatibility(request)
        except Exception as exc:
            error = self._normalize_exception(exc, "request_validation")
            return self._build_failure_response(
                request, error, time.perf_counter() - start_time
            )

        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                delay = min(
                    _BACKOFF_BASE * (2 ** (attempt - 1)), _BACKOFF_CAP
                )
                delay *= random.uniform(0.75, 1.25)
                logger.warning(
                    "[%s] Retrying (attempt %d/%d) after %.1fs backoff",
                    self.model_name,
                    attempt,
                    self.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

            try:
                await self._enforce_min_delay()
                async with self.semaphore:
                    response = await self._generate_provider_response(request)

                response.latency_seconds = time.perf_counter() - start_time
                response.validate()
                return response

            except ProviderRateLimitError as exc:
                logger.warning(
                    "[%s] Rate limit hit (attempt %d/%d): %s",
                    self.model_name,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                last_exc = exc

            except ProviderTimeoutError as exc:
                logger.warning(
                    "[%s] Timeout (attempt %d/%d): %s",
                    self.model_name,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                last_exc = exc

            except ProviderCallError as exc:
                logger.warning(
                    "[%s] Call error (attempt %d/%d): %s",
                    self.model_name,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                last_exc = exc

            except Exception as exc:
                # Non-retryable (config error, validation error, unknown).
                error = self._normalize_exception(exc, "provider_call")
                return self._build_failure_response(
                    request, error, time.perf_counter() - start_time
                )

        # Exhausted all retry attempts.
        error = self._normalize_exception(last_exc, "provider_call")
        return self._build_failure_response(
            request, error, time.perf_counter() - start_time
        )

    async def generate_batch(
        self,
        requests: list[ModelRequest],
    ) -> list[ModelResponse]:
        """Execute multiple requests concurrently, respecting concurrency limit.

        Args:
            requests: Standardized model requests to execute.

        Returns:
            List of standardized model responses, one per input request.
        """
        coroutines = [self.generate(request) for request in requests]
        return list(await asyncio.gather(*coroutines))

    @abstractmethod
    async def _generate_provider_response(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Execute the provider-specific raw model call.

        Subclasses translate a standardized request into the provider's API
        format, perform the call, and return a standardized response or raise
        a typed provider exception.

        Args:
            request: Validated model request for this provider/model pair.

        Returns:
            A standardized model response.
        """

    def _validate_request_compatibility(self, request: ModelRequest) -> None:
        """Validate that the request matches this client's provider and model."""
        if request.provider != self.provider:
            raise ProviderConfigurationError(
                f"Provider mismatch: request={request.provider}, client={self.provider}"
            )
        if request.model_name != self.model_name:
            raise ProviderConfigurationError(
                f"Model mismatch: request={request.model_name}, client={self.model_name}"
            )

    def _build_failure_response(
        self,
        request: ModelRequest,
        error: ErrorInfo,
        latency_seconds: float,
    ) -> ModelResponse:
        """Build a standardized failed response object."""
        response = ModelResponse(
            provider=request.provider,
            model_name=request.model_name,
            status=FAILURE_STATUS,
            latency_seconds=latency_seconds,
            metadata=request.metadata,
            error=error,
            raw_text=None,
            finish_reason=None,
            usage=None,
            timestamp_utc=None,
        )
        response.validate()
        return response

    def _normalize_exception(
        self,
        exc: Exception,
        stage: str,
    ) -> ErrorInfo:
        """Convert a raw exception into standardized error information."""
        if isinstance(exc, ValidationError):
            return ErrorInfo(type(exc).__name__, str(exc), False, stage)
        if isinstance(exc, ProviderRateLimitError):
            return ErrorInfo(type(exc).__name__, str(exc), True, stage)
        if isinstance(exc, ProviderTimeoutError):
            return ErrorInfo(type(exc).__name__, str(exc), True, stage)
        if isinstance(exc, ProviderConfigurationError):
            return ErrorInfo(type(exc).__name__, str(exc), False, stage)
        if isinstance(exc, ProviderCallError):
            return ErrorInfo(type(exc).__name__, str(exc), True, stage)
        return ErrorInfo(
            type(exc).__name__,
            str(exc) if str(exc) else "Unexpected exception with no message.",
            False,
            stage,
        )
