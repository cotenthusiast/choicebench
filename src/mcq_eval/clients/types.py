# src/mcq_eval/clients/types.py

from numbers import Real

from mcq_eval.config.providers import (
    MAX_TOKENS,
    SEED,
    SUPPORTED_MODELS_BY_PROVIDER,
    TEMPERATURE,
)

SUCCESS_STATUS = "success"
FAILURE_STATUS = "failure"
VALID_STATUS = {SUCCESS_STATUS, FAILURE_STATUS}


class ValidationError(Exception):
    """Base exception for validation failures in client request/response types."""
    pass


class RequestValidationError(ValidationError):
    """Raised when a ModelRequest contains invalid, missing, or inconsistent values."""
    pass


class ResponseValidationError(ValidationError):
    """Raised when a ModelResponse contains invalid, missing, or internally inconsistent values."""
    pass


class ProviderConfigurationError(Exception):
    """Raised when a provider or model configuration is unsupported, missing, or incompatible with the current client setup."""
    pass


class ProviderCallError(Exception):
    """Base exception for failures that occur while communicating with an external model provider."""
    pass


class ProviderTimeoutError(ProviderCallError):
    """Raised when a provider request exceeds the allowed timeout."""
    pass


class ProviderRateLimitError(ProviderCallError):
    """Raised when a provider rejects a request because the rate limit or quota has been exceeded."""
    pass


class ProviderResponseError(ProviderCallError):
    """Raised when a provider returns a malformed, incomplete, or otherwise unusable response."""
    pass


class UsageInfo:
    """Standardized token-usage information returned by a provider."""

    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class ErrorInfo:
    """Standardized error record for a failed model request."""

    def __init__(
        self,
        error_type: str,
        message: str,
        retryable: bool,
        stage: str,
    ) -> None:
        self.error_type = error_type
        self.message = message
        self.retryable = retryable
        self.stage = stage


class ModelRequest:
    """Provider-agnostic request object for one model call.

    This object stores the target provider/model, the request payload,
    and generation settings.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        payload: str,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        seed: int | None = SEED,
        request_logprobs: bool = False,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.payload = payload
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.request_logprobs = request_logprobs

    def validate(self) -> None:
        """Validate that the request contains supported values and metadata."""
        if self.provider not in SUPPORTED_MODELS_BY_PROVIDER:
            raise RequestValidationError(
                f"Provider '{self.provider}' is not supported."
            )

        allowed_models = SUPPORTED_MODELS_BY_PROVIDER[self.provider]
        if self.model_name not in allowed_models:
            raise RequestValidationError(
                f"Model '{self.model_name}' is not supported for provider '{self.provider}'."
            )

        if not isinstance(self.payload, str) or not self.payload.strip():
            raise RequestValidationError("payload must be a non-empty string.")

        if isinstance(self.temperature, bool) or not isinstance(self.temperature, Real):
            raise RequestValidationError("temperature must be a numeric value.")

        if self.temperature < 0.0 or self.temperature > 2.0:
            raise RequestValidationError(
                "temperature must be between 0.0 and 2.0."
            )

        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise RequestValidationError("max_tokens must be a positive integer.")

        if self.max_tokens <= 0:
            raise RequestValidationError("max_tokens must be a positive integer.")

        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise RequestValidationError("seed must be an integer or None.")


class ModelResponse:
    """Standardized response object for one model call.

    This object records the outcome of a request, including execution
    status, timing, and the raw model output.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        status: str,
        latency_seconds: float,
        raw_text: str | None = None,
        finish_reason: str | None = None,
        usage: UsageInfo | None = None,
        error: ErrorInfo | None = None,
        timestamp_utc: str | None = None,
        logprobs: list | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.status = status
        self.latency_seconds = latency_seconds
        self.raw_text = raw_text
        self.finish_reason = finish_reason
        self.usage = usage
        self.error = error
        self.timestamp_utc = timestamp_utc
        self.logprobs = logprobs

    def validate(self) -> None:
        """Validate that the response contains supported values and metadata."""
        if self.provider not in SUPPORTED_MODELS_BY_PROVIDER:
            raise ResponseValidationError(
                f"Provider '{self.provider}' is not supported."
            )

        allowed_models = SUPPORTED_MODELS_BY_PROVIDER[self.provider]
        if self.model_name not in allowed_models:
            raise ResponseValidationError(
                f"Model '{self.model_name}' is not supported for provider '{self.provider}'."
            )

        if self.status not in VALID_STATUS:
            raise ResponseValidationError(
                f"status must be one of {sorted(VALID_STATUS)}."
            )

        if isinstance(self.latency_seconds, bool) or not isinstance(self.latency_seconds, Real):
            raise ResponseValidationError(
                "latency_seconds must be a non-negative numeric value."
            )

        if self.latency_seconds < 0:
            raise ResponseValidationError(
                "latency_seconds must be a non-negative numeric value."
            )

        if self.is_success():
            if not isinstance(self.raw_text, str) or not self.raw_text.strip():
                raise ResponseValidationError(
                    "Successful responses must include non-empty raw_text."
                )
            if self.error is not None:
                raise ResponseValidationError(
                    "Successful responses must not include error info."
                )
        else:
            if not isinstance(self.error, ErrorInfo):
                raise ResponseValidationError(
                    "Failure responses must include an ErrorInfo instance."
                )

        if self.usage is not None and not isinstance(self.usage, UsageInfo):
            raise ResponseValidationError(
                "usage must be a UsageInfo instance or None."
            )

        if self.finish_reason is not None:
            if not isinstance(self.finish_reason, str) or not self.finish_reason.strip():
                raise ResponseValidationError(
                    "finish_reason must be a non-empty string or None."
                )

        if self.timestamp_utc is not None:
            if not isinstance(self.timestamp_utc, str) or not self.timestamp_utc.strip():
                raise ResponseValidationError(
                    "timestamp_utc must be a non-empty string or None."
                )

    def is_success(self) -> bool:
        """Return True if the response status indicates success."""
        return self.status == SUCCESS_STATUS
