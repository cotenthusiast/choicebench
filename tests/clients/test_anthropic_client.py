# tests/clients/test_anthropic_client.py

import httpx
import pytest
import anthropic
from types import SimpleNamespace
from unittest.mock import AsyncMock

from choicebench.clients.anthropic_client import AnthropicClient
from choicebench.clients.types import (
    BATCH_COMPLETED,
    BATCH_IN_PROGRESS,
    ModelRequest,
    ProviderResponseError,
    ProviderCallError,
    ProviderRateLimitError,
    ProviderConfigurationError,
)


@pytest.fixture
def anthropic_client() -> AnthropicClient:
    return AnthropicClient(model_name="claude-opus-4-5", api_key="test-key")


@pytest.fixture
def model_request() -> ModelRequest:
    return ModelRequest(
        provider="anthropic", model_name="claude-opus-4-5",
        payload="test prompt", temperature=0.2, max_tokens=128,
    )


@pytest.fixture
def make_anthropic_response():
    def _make_response(
        *, text="hello", stop_reason="end_turn",
        input_tokens=10, output_tokens=5, include_usage=True,
    ):
        content = [SimpleNamespace(text=text, type="text")]
        usage = None
        if include_usage:
            usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
        return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)
    return _make_response


@pytest.fixture
def mock_create(anthropic_client: AnthropicClient) -> AsyncMock:
    mock = AsyncMock()
    anthropic_client.client.messages.create = mock
    return mock


class TestAnthropicClientGenerateProviderResponse:
    """Tests for AnthropicClient._generate_provider_response."""

    pytestmark = pytest.mark.asyncio

    async def test_returns_model_response_on_success(self, anthropic_client, model_request, make_anthropic_response, mock_create) -> None:
        mock_create.return_value = make_anthropic_response()
        response = await anthropic_client._generate_provider_response(model_request)
        assert response.raw_text == "hello"
        assert response.provider == "anthropic"
        assert response.finish_reason == "end_turn"

    async def test_raises_provider_response_error_when_text_is_none(self, anthropic_client, model_request, make_anthropic_response, mock_create) -> None:
        mock_create.return_value = make_anthropic_response(text=None)
        with pytest.raises(ProviderResponseError):
            await anthropic_client._generate_provider_response(model_request)

    async def test_raises_provider_response_error_when_text_is_whitespace(self, anthropic_client, model_request, make_anthropic_response, mock_create) -> None:
        mock_create.return_value = make_anthropic_response(text="   ")
        with pytest.raises(ProviderResponseError):
            await anthropic_client._generate_provider_response(model_request)

    async def test_allows_missing_usage(self, anthropic_client, model_request, make_anthropic_response, mock_create) -> None:
        mock_create.return_value = make_anthropic_response(include_usage=False)
        response = await anthropic_client._generate_provider_response(model_request)
        assert response.usage is None

    async def test_usage_totals_are_summed(self, anthropic_client, model_request, make_anthropic_response, mock_create) -> None:
        mock_create.return_value = make_anthropic_response(input_tokens=10, output_tokens=5)
        response = await anthropic_client._generate_provider_response(model_request)
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    async def test_raises_provider_rate_limit_error_for_429(self, anthropic_client, model_request, mock_create) -> None:
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        http_response = httpx.Response(429, request=request)
        mock_create.side_effect = anthropic.RateLimitError("rate limit", response=http_response, body={"error": {"message": "rate limit"}})
        with pytest.raises(ProviderRateLimitError):
            await anthropic_client._generate_provider_response(model_request)

    async def test_raises_provider_configuration_error_for_404(self, anthropic_client, model_request, mock_create) -> None:
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        http_response = httpx.Response(404, request=request)
        mock_create.side_effect = anthropic.NotFoundError("not found", response=http_response, body={"error": {"message": "not found"}})
        with pytest.raises(ProviderConfigurationError):
            await anthropic_client._generate_provider_response(model_request)

    async def test_raises_provider_call_error_for_500(self, anthropic_client, model_request, mock_create) -> None:
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        http_response = httpx.Response(500, request=request)
        mock_create.side_effect = anthropic.InternalServerError("server error", response=http_response, body={"error": {"message": "server error"}})
        with pytest.raises(ProviderCallError):
            await anthropic_client._generate_provider_response(model_request)

    async def test_uses_env_api_key(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-test-key")
        import importlib
        import choicebench.config.providers as prov
        importlib.reload(prov)
        client = AnthropicClient(model_name="claude-opus-4-5")
        assert client.client.api_key == "env-test-key"

    async def test_uses_async_client(self, anthropic_client) -> None:
        from anthropic import AsyncAnthropic
        assert isinstance(anthropic_client.client, AsyncAnthropic)


class _FakeBatchResultsDecoder:
    """Async-iterable double for AsyncBatches.results()'s AsyncJSONLDecoder."""

    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for item in self._items:
            yield item


def _succeeded_result(custom_id: str, text: str, stop_reason="end_turn"):
    message = SimpleNamespace(
        content=[SimpleNamespace(text=text, type="text")],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    result = SimpleNamespace(type="succeeded", message=message)
    return SimpleNamespace(custom_id=custom_id, result=result)


def _errored_result(custom_id: str, message: str):
    error_object = SimpleNamespace(message=message)
    error_response = SimpleNamespace(error=error_object)
    result = SimpleNamespace(type="errored", error=error_response)
    return SimpleNamespace(custom_id=custom_id, result=result)


def _expired_result(custom_id: str):
    result = SimpleNamespace(type="expired")
    return SimpleNamespace(custom_id=custom_id, result=result)


@pytest.fixture
def batch_requests() -> list[ModelRequest]:
    return [
        ModelRequest(provider="anthropic", model_name="claude-opus-4-5", payload=f"question {i}",
                     temperature=0.0, max_tokens=64)
        for i in range(3)
    ]


class TestAnthropicClientSubmitBatch:
    pytestmark = pytest.mark.asyncio

    async def test_submits_one_request_per_prompt_with_stringified_index_custom_id(
        self, anthropic_client, batch_requests,
    ):
        create = AsyncMock(return_value=SimpleNamespace(id="msgbatch_abc"))
        anthropic_client.client.messages.batches.create = create

        batch_id = await anthropic_client.submit_batch(batch_requests)

        assert batch_id == "msgbatch_abc"
        sent_requests = create.call_args.kwargs["requests"]
        assert [r["custom_id"] for r in sent_requests] == ["0", "1", "2"]
        assert sent_requests[0]["params"]["model"] == "claude-opus-4-5"
        assert sent_requests[0]["params"]["messages"] == [{"role": "user", "content": "question 0"}]
        assert sent_requests[0]["params"]["max_tokens"] == 64
        assert sent_requests[0]["params"]["temperature"] == 0.0


class TestAnthropicClientPollBatch:
    pytestmark = pytest.mark.asyncio

    async def test_ended_maps_to_completed(self, anthropic_client):
        anthropic_client.client.messages.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(processing_status="ended")
        )
        assert await anthropic_client.poll_batch("msgbatch_abc") == BATCH_COMPLETED

    async def test_in_progress_maps_to_in_progress(self, anthropic_client):
        anthropic_client.client.messages.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(processing_status="in_progress")
        )
        assert await anthropic_client.poll_batch("msgbatch_abc") == BATCH_IN_PROGRESS


class TestAnthropicClientFetchBatchResults:
    pytestmark = pytest.mark.asyncio

    async def test_parses_succeeded_results(self, anthropic_client, batch_requests):
        decoder = _FakeBatchResultsDecoder([
            _succeeded_result("0", "A"), _succeeded_result("1", "B"), _succeeded_result("2", "C"),
        ])
        anthropic_client.client.messages.batches.results = AsyncMock(return_value=decoder)

        results = await anthropic_client.fetch_batch_results("msgbatch_abc", batch_requests)

        assert [r.raw_text for r in results] == ["A", "B", "C"]
        assert all(r.is_success() for r in results)

    async def test_reassembles_by_custom_id_not_stream_order(self, anthropic_client, batch_requests):
        decoder = _FakeBatchResultsDecoder([
            _succeeded_result("2", "C"), _succeeded_result("0", "A"), _succeeded_result("1", "B"),
        ])
        anthropic_client.client.messages.batches.results = AsyncMock(return_value=decoder)

        results = await anthropic_client.fetch_batch_results("msgbatch_abc", batch_requests)

        assert [r.raw_text for r in results] == ["A", "B", "C"]

    async def test_errored_result_becomes_a_failure_response(self, anthropic_client, batch_requests):
        decoder = _FakeBatchResultsDecoder([
            _succeeded_result("0", "A"),
            _errored_result("1", "invalid request"),
            _succeeded_result("2", "C"),
        ])
        anthropic_client.client.messages.batches.results = AsyncMock(return_value=decoder)

        results = await anthropic_client.fetch_batch_results("msgbatch_abc", batch_requests)

        assert results[0].is_success()
        assert not results[1].is_success()
        assert "invalid request" in results[1].error.message
        assert results[2].is_success()

    async def test_expired_result_becomes_a_failure_response(self, anthropic_client, batch_requests):
        decoder = _FakeBatchResultsDecoder([
            _succeeded_result("0", "A"), _expired_result("1"), _succeeded_result("2", "C"),
        ])
        anthropic_client.client.messages.batches.results = AsyncMock(return_value=decoder)

        results = await anthropic_client.fetch_batch_results("msgbatch_abc", batch_requests)

        assert not results[1].is_success()
        assert "expired" in results[1].error.message.lower()

    async def test_a_missing_custom_id_becomes_a_failure_response_not_a_crash(self, anthropic_client, batch_requests):
        decoder = _FakeBatchResultsDecoder([_succeeded_result("0", "A"), _succeeded_result("2", "C")])
        anthropic_client.client.messages.batches.results = AsyncMock(return_value=decoder)

        results = await anthropic_client.fetch_batch_results("msgbatch_abc", batch_requests)

        assert results[0].is_success() and results[2].is_success()
        assert not results[1].is_success()
