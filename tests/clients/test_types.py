# tests/clients/test_types.py

import pytest

from choicebench.clients.types import (
    RequestValidationError,
    ResponseValidationError,
)


class TestModelRequestValidation:
    """Tests for ModelRequest.validate."""

    def test_passes_for_valid_request(self, valid_request):
        valid_request.validate()

    @pytest.mark.parametrize("payload", ["", 123, None])
    def test_fails_for_invalid_payload(self, valid_request, payload):
        valid_request.payload = payload
        with pytest.raises(RequestValidationError):
            valid_request.validate()

    @pytest.mark.parametrize(
        "temperature,max_tokens,seed",
        [
            ("hot", 500, 42),
            (-0.1, 500, 42),
            (2.1, 500, 42),
            (0.0, 0, 42),
            (0.0, -1, 42),
            (0.0, 1.5, 42),
            (0.0, 500, "seed"),
        ],
    )
    def test_fails_for_invalid_generation_parameters(self, valid_request, temperature, max_tokens, seed):
        valid_request.temperature = temperature
        valid_request.max_tokens = max_tokens
        valid_request.seed = seed
        with pytest.raises(RequestValidationError):
            valid_request.validate()

class TestModelResponseValidation:
    """Tests for ModelResponse.validate."""

    def test_passes_for_valid_success_response(self, successful_response):
        successful_response.validate()

    def test_passes_for_valid_failure_response(self, failed_response):
        failed_response.validate()

    @pytest.mark.parametrize(
        "status,latency_seconds",
        [
            ("bad_status", 0.2),
            ("success", -1.0),
            ("success", "fast"),
        ],
    )
    def test_fails_for_invalid_status_or_latency(
        self, successful_response, status, latency_seconds,
    ):
        successful_response.status = status
        successful_response.latency_seconds = latency_seconds
        with pytest.raises(ResponseValidationError):
            successful_response.validate()

    def test_fails_for_broken_success_invariants(self, successful_response):
        successful_response.raw_text = ""
        with pytest.raises(ResponseValidationError):
            successful_response.validate()

    def test_fails_for_broken_failure_invariants(self, failed_response):
        failed_response.error = None
        with pytest.raises(ResponseValidationError):
            failed_response.validate()

    @pytest.mark.parametrize(
        "usage,finish_reason,timestamp_utc",
        [
            (object(), "stop", "2026-03-13T21:00:00Z"),
            (None, "", "2026-03-13T21:00:00Z"),
            (None, 123, "2026-03-13T21:00:00Z"),
            (None, "stop", ""),
            (None, "stop", 123),
        ],
    )
    def test_fails_for_invalid_optional_fields(self, successful_response, usage, finish_reason, timestamp_utc):
        successful_response.timestamp_utc = timestamp_utc
        successful_response.usage = usage
        successful_response.finish_reason = finish_reason
        with pytest.raises(ResponseValidationError):
            successful_response.validate()

    def test_is_success_returns_true_for_success(self, successful_response):
        assert successful_response.is_success()

    def test_is_success_returns_false_for_failure(self, failed_response):
        assert not failed_response.is_success()
