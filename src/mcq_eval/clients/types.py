# src/mcq_eval/clients/types.py

from numbers import Real

from mcq_eval.config.models import (
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
    """Raised when a ModelRequest or its attached metadata contains invalid, missing, or inconsistent values."""
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


# -----------------------------------------------------------------------------------------------


class RequestMetadata:
    """Trace metadata attached to one experiment request/response pair.

    This object identifies a single benchmark item across async execution,
    logging, saving, and evaluation.
    """

    def __init__(
        self,
        question_id: str,
        split_name: str,
        method_name: str,
        subject: str,
        run_id: str,
        prompt_version: str,
        perturbation_name: str | None,
        sample_index: int,
    ) -> None:
        self.question_id = question_id
        self.split_name = split_name
        self.method_name = method_name
        self.subject = subject
        self.run_id = run_id
        self.prompt_version = prompt_version
        self.perturbation_name = perturbation_name
        self.sample_index = sample_index

    def validate(self) -> None:
        """Validate that all required metadata fields are present and valid."""
        required_str_fields = {
            "question_id": self.question_id,
            "split_name": self.split_name,
            "method_name": self.method_name,
            "subject": self.subject,
            "run_id": self.run_id,
            "prompt_version": self.prompt_version,
        }

        for field_name, value in required_str_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise RequestValidationError(
                    f"{field_name} must be a non-empty string."
                )

        if self.perturbation_name is not None:
            if (
                not isinstance(self.perturbation_name, str)
                or not self.perturbation_name.strip()
            ):
                raise RequestValidationError(
                    "perturbation_name must be None or a non-empty string."
                )

        if isinstance(self.sample_index, bool) or not isinstance(self.sample_index, int) or self.sample_index < 0:
            raise RequestValidationError(
                "sample_index must be a non-negative integer."
            )


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
    generation settings, and trace metadata needed to track the call.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        payload: str,
        metadata: RequestMetadata,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        seed: int | None = SEED,
        request_logprobs: bool = False,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.payload = payload
        self.metadata = metadata
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

        if not isinstance(self.metadata, RequestMetadata):
            raise RequestValidationError(
                "metadata must be a RequestMetadata instance."
            )

        self.metadata.validate()


class ModelResponse:
    """Standardized response object for one model call.

    This object records the outcome of a request, including basic execution
    status, timing, and the metadata needed to map the response back to the
    original benchmark item.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        status: str,
        latency_seconds: float,
        metadata: RequestMetadata,
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
        self.metadata = metadata
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

        if not isinstance(self.metadata, RequestMetadata):
            raise ResponseValidationError(
                "metadata must be a RequestMetadata instance."
            )

        self.metadata.validate()

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
