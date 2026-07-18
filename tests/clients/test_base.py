# tests/clients/test_base.py

import pytest

from choicebench.clients.base import BaseClient
from choicebench.clients.types import (
    ErrorInfo,
    FAILURE_STATUS,
    ModelRequest,
    ModelResponse,
    ProviderConfigurationError,
    ProviderTimeoutError,
    SUCCESS_STATUS,
    ValidationError,
)


class DummyClient(BaseClient):
    def __init__(
        self, provider: str, model_name: str, behavior: str = "success",
        timeout: int = 30, concurrency_limit: int = 10, max_retries: int = 3,
    ) -> None:
        super().__init__(
            provider=provider, model_name=model_name, timeout=timeout,
            concurrency_limit=concurrency_limit, max_retries=max_retries,
        )
        self.behavior = behavior

    async def _generate_provider_response(self, request):
        if self.behavior == "success":
            return ModelResponse(
                provider=request.provider, model_name=request.model_name,
                status=SUCCESS_STATUS, latency_seconds=0.0,
                raw_text="B", finish_reason="stop", usage=None, error=None, timestamp_utc=None,
            )
        if self.behavior == "timeout":
            raise ProviderTimeoutError("Request timed out.")
        if self.behavior == "generic_error":
            raise RuntimeError("Unexpected provider failure.")
        if self.behavior == "invalid_response":
            return ModelResponse(
                provider=request.provider, model_name=request.model_name,
                status=SUCCESS_STATUS, latency_seconds=0.0,
                raw_text="", finish_reason="stop", usage=None, error=None, timestamp_utc=None,
            )
        if self.behavior == "failure_response":
            return ModelResponse(
                provider=request.provider, model_name=request.model_name,
                status=FAILURE_STATUS, latency_seconds=0.0,
                raw_text=None, finish_reason=None, usage=None,
                error=ErrorInfo(
                    error_type="ProviderCallError", message="Forced failure response.",
                    retryable=True, stage="provider_call",
                ),
                timestamp_utc=None,
            )
        raise RuntimeError(f"Unknown dummy behavior: {self.behavior}")


@pytest.fixture
def dummy_success_client() -> DummyClient:
    return DummyClient(provider="openai", model_name="gpt-4.1-mini", behavior="success")

@pytest.fixture
def dummy_timeout_client() -> DummyClient:
    return DummyClient(provider="openai", model_name="gpt-4.1-mini", behavior="timeout")

@pytest.fixture
def dummy_generic_error_client() -> DummyClient:
    return DummyClient(provider="openai", model_name="gpt-4.1-mini", behavior="generic_error")

@pytest.fixture
def dummy_invalid_response_client() -> DummyClient:
    return DummyClient(provider="openai", model_name="gpt-4.1-mini", behavior="invalid_response")

@pytest.fixture
def dummy_failure_response_client() -> DummyClient:
    return DummyClient(provider="openai", model_name="gpt-4.1-mini", behavior="failure_response")

@pytest.fixture
def dummy_mismatch_client() -> DummyClient:
    return DummyClient(provider="gemini", model_name="gemini-2.5-flash", behavior="success")


class TestGenerate:
    """Tests for BaseClient.generate."""

    @pytest.mark.asyncio
    async def test_returns_valid_success_response_for_valid_request(self, dummy_success_client, valid_request):
        response = await dummy_success_client.generate(valid_request)
        assert response.is_success()
        assert response.error is None
        assert response.model_name == valid_request.model_name
        assert response.provider == valid_request.provider
        assert response.latency_seconds >= 0

    @pytest.mark.asyncio
    async def test_returns_failed_response_for_provider_model_mismatch(self, dummy_mismatch_client, valid_request):
        response = await dummy_mismatch_client.generate(valid_request)
        assert not response.is_success()
        assert response.status == FAILURE_STATUS
        assert response.error is not None
        assert response.error.error_type == "ProviderConfigurationError"
        assert response.error.stage == "request_validation"
        assert response.error.retryable is False
        assert response.latency_seconds >= 0

    @pytest.mark.asyncio
    async def test_returns_failed_response_for_provider_timeout(self, dummy_timeout_client, valid_request):
        response = await dummy_timeout_client.generate(valid_request)
        assert not response.is_success()
        assert response.status == FAILURE_STATUS
        assert response.error.error_type == "ProviderTimeoutError"
        assert response.error.stage == "provider_call"
        assert response.error.retryable is True
        assert response.latency_seconds >= 0

    @pytest.mark.asyncio
    async def test_returns_failed_response_for_unknown_generic_exception(self, dummy_generic_error_client, valid_request):
        response = await dummy_generic_error_client.generate(valid_request)
        assert not response.is_success()
        assert response.error.error_type == "RuntimeError"
        assert response.error.stage == "provider_call"
        assert response.error.retryable is False

    @pytest.mark.asyncio
    async def test_returns_failed_response_for_invalid_provider_response(self, dummy_invalid_response_client, valid_request):
        response = await dummy_invalid_response_client.generate(valid_request)
        assert not response.is_success()
        assert response.error.error_type == "ResponseValidationError"
        assert response.error.stage == "provider_call"
        assert response.error.retryable is False


class TestGenerateBatch:
    """Tests for BaseClient.generate_batch."""

    @pytest.mark.asyncio
    async def test_returns_one_response_per_request(self, dummy_success_client, valid_request):
        requests = [
            ModelRequest(
                provider=valid_request.provider, model_name=valid_request.model_name,
                payload=f"Question {i}",
                temperature=valid_request.temperature, max_tokens=valid_request.max_tokens, seed=valid_request.seed,
            )
            for i in range(3)
        ]
        responses = await dummy_success_client.generate_batch(requests)
        assert len(responses) == len(requests)
        for response in responses:
            assert response.is_success()


class TestBuildFailureResponse:
    """Tests for BaseClient._build_failure_response."""

    def test_returns_valid_failure_model_response(self, dummy_success_client, valid_request):
        error = ErrorInfo(
            error_type="ProviderTimeoutError", message="Request timed out.",
            retryable=True, stage="provider_call",
        )
        response = dummy_success_client._build_failure_response(
            request=valid_request, error=error, latency_seconds=0.5,
        )
        assert response.status == FAILURE_STATUS
        assert not response.is_success()
        assert response.error is error
        assert response.provider == valid_request.provider
        assert response.model_name == valid_request.model_name
        assert response.raw_text is None
        assert response.latency_seconds == 0.5
        response.validate()


class TestNormalizeException:
    """Tests for BaseClient._normalize_exception."""

    @pytest.mark.parametrize(
        "exc,stage,expected_error_type,expected_retryable",
        [
            (ValidationError("bad request"), "request_validation", "ValidationError", False),
            (ProviderTimeoutError("timed out"), "provider_call", "ProviderTimeoutError", True),
            (ProviderConfigurationError("mismatch"), "provider_call", "ProviderConfigurationError", False),
            (RuntimeError("boom"), "provider_call", "RuntimeError", False),
            (ValueError("bad value"), "provider_call", "ValueError", False),
        ],
    )
    def test_maps_exceptions_correctly(self, dummy_success_client, exc, stage, expected_error_type, expected_retryable):
        error = dummy_success_client._normalize_exception(exc, stage)
        assert error.error_type == expected_error_type
        assert error.message == str(exc)
        assert error.retryable is expected_retryable
        assert error.stage == stage


@pytest.mark.parametrize("kwargs", [
    {"timeout": 0}, {"timeout": float("nan")}, {"timeout": True},
    {"concurrency_limit": 0}, {"concurrency_limit": True},
    {"max_retries": -1}, {"max_retries": True},
])
def test_constructor_rejects_hazardous_numeric_values(kwargs):
    with pytest.raises(ValueError):
        DummyClient(provider="openai", model_name="m", **kwargs)
