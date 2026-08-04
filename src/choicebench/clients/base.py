# src/choicebench/clients/base.py

from __future__ import annotations

import asyncio
import logging
import random
import time
import math
from numbers import Real
from abc import ABC, abstractmethod
from choicebench.identity import redact_text

from choicebench.clients.types import (
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
from choicebench.config.providers import MAX_RETRIES, TIMEOUT

logger = logging.getLogger(__name__)

# Exponential backoff constants (seconds)
_BACKOFF_BASE = 5.0
_BACKOFF_CAP = 300.0


def map_api_status_error(exc) -> Exception:
    """Map an OpenAI/Groq-shaped APIStatusError to our exception taxonomy.

    Shared by every OpenAI-compatible Chat Completions client (groq, together,
    vllm) — each catches its own SDK's APIStatusError class, then delegates
    the identical status-code bucketing here.
    """
    message = str(exc)
    if exc.status_code == 429:
        return ProviderRateLimitError(message)
    if exc.status_code in {400, 401, 403, 404, 422}:
        return ProviderConfigurationError(message)
    return ProviderCallError(message)


class BaseClient(ABC):
    """Abstract base client for provider-backed model calls.

    Handles request validation, bounded concurrency, and exponential-backoff
    retry on transient errors. Concrete subclasses implement only the
    provider-specific raw call.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        timeout: int = TIMEOUT,
        concurrency_limit: int = 10,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        """Initialize shared client configuration.

        Args:
            provider:            Provider name string.
            model_name:          Exact model name string.
            timeout:             Per-request timeout in seconds.
            concurrency_limit:   Max in-flight requests at once.
            max_retries:         Retry attempts on transient errors.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, Real) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number.")
        if isinstance(concurrency_limit, bool) or not isinstance(concurrency_limit, int) or concurrency_limit <= 0:
            raise ValueError("concurrency_limit must be a positive integer.")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        self.provider = provider
        self.model_name = model_name
        self.timeout = timeout
        self.concurrency_limit = concurrency_limit
        self.max_retries = max_retries

        self.semaphore = asyncio.Semaphore(concurrency_limit)

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
                    redact_text(exc),
                )
                last_exc = exc

            except ProviderTimeoutError as exc:
                logger.warning(
                    "[%s] Timeout (attempt %d/%d): %s",
                    self.model_name,
                    attempt + 1,
                    self.max_retries + 1,
                    redact_text(exc),
                )
                last_exc = exc

            except ProviderCallError as exc:
                logger.warning(
                    "[%s] Call error (attempt %d/%d): %s",
                    self.model_name,
                    attempt + 1,
                    self.max_retries + 1,
                    redact_text(exc),
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
            return ErrorInfo(type(exc).__name__, redact_text(exc), False, stage)
        if isinstance(exc, ProviderRateLimitError):
            return ErrorInfo(type(exc).__name__, redact_text(exc), True, stage)
        if isinstance(exc, ProviderTimeoutError):
            return ErrorInfo(type(exc).__name__, redact_text(exc), True, stage)
        if isinstance(exc, ProviderConfigurationError):
            return ErrorInfo(type(exc).__name__, redact_text(exc), False, stage)
        if isinstance(exc, ProviderCallError):
            return ErrorInfo(type(exc).__name__, redact_text(exc), True, stage)
        return ErrorInfo(
            type(exc).__name__,
            redact_text(exc) if str(exc) else "Unexpected exception with no message.",
            False,
            stage,
        )
