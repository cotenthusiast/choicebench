# tests/runners/conftest.py

import pytest

from twoprompt.backends.base import BaseBackend
from twoprompt.clients.base import BaseClient
from twoprompt.clients.types import (
    ModelRequest,
    ModelResponse,
    UsageInfo,
    SUCCESS_STATUS,
    FAILURE_STATUS,
    ErrorInfo,
    RequestMetadata,
)


class MockClient(BaseClient):
    """A fake client that returns predetermined responses for testing.

    Supports queueing multiple responses for multi-call runners
    like TwoStageRunner and PermutationRunner.
    """

    def __init__(
            self,
            responses: list[ModelResponse] | None = None,
            provider: str = "openai",
            model_name: str = "gpt-4.1-mini",
    ) -> None:
        super().__init__(
            provider=provider,
            model_name=model_name,
        )
        self._responses = list(responses) if responses else []
        self._call_count = 0
        self.requests_received: list[ModelRequest] = []

    async def _generate_provider_response(
            self,
            request: ModelRequest,
    ) -> ModelResponse:
        self.requests_received.append(request)
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            return response
        raise RuntimeError("MockClient has no more queued responses.")


class MockBackend(BaseBackend):
    """Test double: queues a sequence of fixed text responses for backend.generate().

    Mirrors MockClient's queueing behavior (one entry consumed per call,
    raises once exhausted) but implements BaseBackend's generate(prompt) -> str
    contract instead of the client-level ModelRequest/ModelResponse contract.

    Each queued entry is either a str (returned as the generated text) or an
    Exception instance (raised — ExperimentRunner._call_backend_generate
    catches it and turns it into a failure ModelResponse, so
    result["error_type"] becomes the exception's class name, matching the
    old MockClient + _make_failure_response behavior).
    """

    def __init__(
            self,
            responses: list[str | Exception] | None = None,
            provider: str = "openai",
            model_name: str = "gpt-4.1-mini",
            supports_logprobs: bool = False,
    ) -> None:
        self._provider = provider
        self._model_name = model_name
        self._supports_logprobs = supports_logprobs
        self._responses = list(responses) if responses else []
        self._call_count = 0
        self.requests_received: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def supports_logprobs(self) -> bool:
        return self._supports_logprobs

    def generate(self, prompt: str, **kwargs) -> str:
        self.requests_received.append(prompt)
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            if isinstance(response, Exception):
                raise response
            return response
        raise RuntimeError("MockBackend has no more queued responses.")


def _make_success_response(
        raw_text: str,
        metadata: RequestMetadata,
        provider: str = "openai",
        model_name: str = "gpt-4.1-mini",
) -> ModelResponse:
    """Helper to build a successful ModelResponse."""
    return ModelResponse(
        provider=provider,
        model_name=model_name,
        status=SUCCESS_STATUS,
        latency_seconds=0.25,
        metadata=metadata,
        raw_text=raw_text,
        finish_reason="stop",
        usage=UsageInfo(
            prompt_tokens=50,
            completion_tokens=5,
            total_tokens=55,
        ),
        error=None,
        timestamp_utc="2026-03-25T12:00:00Z",
    )


def _make_failure_response(
        metadata: RequestMetadata,
        provider: str = "openai",
        model_name: str = "gpt-4.1-mini",
) -> ModelResponse:
    """Helper to build a failed ModelResponse."""
    return ModelResponse(
        provider=provider,
        model_name=model_name,
        status=FAILURE_STATUS,
        latency_seconds=0.40,
        metadata=metadata,
        raw_text=None,
        finish_reason=None,
        usage=None,
        error=ErrorInfo(
            error_type="ProviderTimeoutError",
            message="Request timed out.",
            retryable=True,
            stage="provider_call",
        ),
        timestamp_utc=None,
    )


@pytest.fixture
def runner_metadata() -> RequestMetadata:
    """Metadata used for constructing mock responses in runner tests."""
    return RequestMetadata(
        question_id="4865890d7f0efae8",
        split_name="robustness",
        method_name="baseline",
        subject="computer_security",
        run_id="test_run_001",
        prompt_version="v1",
        perturbation_name=None,
        sample_index=0,
    )


@pytest.fixture
def runner_question_row() -> dict[str, object]:
    """A normalized question row with all fields runners expect."""
    return {
        "question_id": "4865890d7f0efae8",
        "subject": "computer_security",
        "question_text": "Which protocol is primarily used to securely browse websites?",
        "choice_a": "FTP",
        "choice_b": "HTTP",
        "choice_c": "HTTPS",
        "choice_d": "SMTP",
        "correct_option": "C",
        "correct_answer_text": "HTTPS",
    }


@pytest.fixture
def success_response_c(runner_metadata: RequestMetadata) -> ModelResponse:
    """A successful response returning letter C."""
    return _make_success_response("C", runner_metadata)


@pytest.fixture
def success_response_a(runner_metadata: RequestMetadata) -> ModelResponse:
    """A successful response returning letter A."""
    return _make_success_response("A", runner_metadata)


@pytest.fixture
def success_response_b(runner_metadata: RequestMetadata) -> ModelResponse:
    """A successful response returning letter B."""
    return _make_success_response("B", runner_metadata)


@pytest.fixture
def success_response_free_text(runner_metadata: RequestMetadata) -> ModelResponse:
    """A successful response returning a free-text answer."""
    return _make_success_response("HTTPS", runner_metadata)


@pytest.fixture
def failure_response(runner_metadata: RequestMetadata) -> ModelResponse:
    """A failed response for testing error handling."""
    return _make_failure_response(runner_metadata)


@pytest.fixture
def mock_client_success_c(success_response_c: ModelResponse) -> MockClient:
    """A mock client that always returns letter C."""
    return MockClient(responses=[success_response_c])


@pytest.fixture
def canonical_options() -> dict[str, str]:
    """The canonical option mapping for the sample question."""
    return {
        "A": "FTP",
        "B": "HTTP",
        "C": "HTTPS",
        "D": "SMTP",
    }