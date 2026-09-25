# tests/clients/test_deepinfra_client.py

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from choicebench.clients.deepinfra_client import DeepInfraClient
from choicebench.clients.types import (
    BATCH_COMPLETED,
    BATCH_FAILED,
    BATCH_IN_PROGRESS,
    ModelRequest,
    ProviderCallError,
    ProviderConfigurationError,
    ProviderResponseError,
)


@pytest.fixture
def deepinfra_client() -> DeepInfraClient:
    return DeepInfraClient(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", api_key="test-key")


@pytest.fixture
def model_request() -> ModelRequest:
    return ModelRequest(
        provider="deepinfra", model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        payload="test prompt", temperature=0.0, max_tokens=128,
    )


def _make_response(*, text="hello", finish_reason="stop", include_usage=True) -> SimpleNamespace:
    choices = [SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)]
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15) if include_usage else None
    return SimpleNamespace(choices=choices, usage=usage)


@pytest.fixture
def mock_create(deepinfra_client: DeepInfraClient) -> AsyncMock:
    mock = AsyncMock()
    deepinfra_client.client.chat.completions.create = mock
    return mock


class TestDeepInfraClientGenerateProviderResponse:
    pytestmark = pytest.mark.asyncio

    async def test_returns_model_response_on_success(self, deepinfra_client, model_request, mock_create):
        mock_create.return_value = _make_response(text="C")
        response = await deepinfra_client._generate_provider_response(model_request)
        assert response.raw_text == "C"
        assert response.provider == "deepinfra"

    async def test_uses_the_deepinfra_base_url(self, deepinfra_client):
        assert str(deepinfra_client.client.base_url) == "https://api.deepinfra.com/v1/openai/"

    async def test_raises_provider_response_error_when_text_is_empty(self, deepinfra_client, model_request, mock_create):
        mock_create.return_value = _make_response(text="")
        with pytest.raises(ProviderResponseError):
            await deepinfra_client._generate_provider_response(model_request)

    async def test_raises_provider_configuration_error_for_404(self, deepinfra_client, model_request, mock_create):
        request = httpx.Request("POST", "https://api.deepinfra.com/v1/openai/chat/completions")
        http_response = httpx.Response(404, request=request)
        mock_create.side_effect = openai.NotFoundError("not found", response=http_response, body={"message": "not found"})
        with pytest.raises(ProviderConfigurationError):
            await deepinfra_client._generate_provider_response(model_request)

    async def test_raises_provider_call_error_for_500(self, deepinfra_client, model_request, mock_create):
        request = httpx.Request("POST", "https://api.deepinfra.com/v1/openai/chat/completions")
        http_response = httpx.Response(500, request=request)
        mock_create.side_effect = openai.InternalServerError("server error", response=http_response, body={"message": "server error"})
        with pytest.raises(ProviderCallError):
            await deepinfra_client._generate_provider_response(model_request)


def _chat_completion_body(text: str, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> dict:
    return {
        "id": "chatcmpl-abc", "object": "chat.completion", "created": 123, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def _output_line(custom_id: str, text: str) -> str:
    return json.dumps({
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": _chat_completion_body(text)},
        "error": None,
    })


def _error_line(custom_id: str, message: str) -> str:
    return json.dumps({
        "custom_id": custom_id, "response": None, "error": {"code": "invalid_request", "message": message},
    })


@pytest.fixture
def batch_requests() -> list[ModelRequest]:
    return [
        ModelRequest(provider="deepinfra", model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                     payload=f"question {i}", temperature=0.0, max_tokens=64)
        for i in range(3)
    ]


class TestDeepInfraClientSubmitBatch:
    pytestmark = pytest.mark.asyncio

    async def test_uploads_jsonl_and_creates_the_batch_against_chat_completions(self, deepinfra_client, batch_requests):
        deepinfra_client.client.files.create = AsyncMock(return_value=SimpleNamespace(id="file_abc"))
        create_batch = AsyncMock(return_value=SimpleNamespace(id="batch_abc"))
        deepinfra_client.client.batches.create = create_batch

        batch_id = await deepinfra_client.submit_batch(batch_requests)

        assert batch_id == "batch_abc"
        upload_kwargs = deepinfra_client.client.files.create.call_args.kwargs
        assert upload_kwargs["purpose"] == "batch"
        _, jsonl_bytes = upload_kwargs["file"]
        lines = jsonl_bytes.decode("utf-8").strip().splitlines()
        parsed = [json.loads(line) for line in lines]
        assert [p["custom_id"] for p in parsed] == ["0", "1", "2"]
        assert all(p["url"] == "/v1/chat/completions" for p in parsed)
        assert parsed[0]["body"]["messages"] == [{"role": "user", "content": "question 0"}]
        create_batch.assert_awaited_once_with(
            input_file_id="file_abc", endpoint="/v1/chat/completions", completion_window="24h",
        )


class TestDeepInfraClientPollBatch:
    pytestmark = pytest.mark.asyncio

    @pytest.mark.parametrize("raw_status,expected", [
        ("completed", BATCH_COMPLETED),
        ("in_progress", BATCH_IN_PROGRESS),
        ("failed", BATCH_FAILED),
        ("expired", BATCH_FAILED),
        ("cancelled", BATCH_FAILED),
    ])
    async def test_maps_deepinfra_status_to_normalized_status(self, deepinfra_client, raw_status, expected):
        deepinfra_client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(status=raw_status))
        assert await deepinfra_client.poll_batch("batch_abc") == expected


class TestDeepInfraClientFetchBatchResults:
    pytestmark = pytest.mark.asyncio

    async def test_parses_successful_lines_from_the_output_file(self, deepinfra_client, batch_requests):
        jsonl = "\n".join([_output_line("0", "A"), _output_line("1", "B"), _output_line("2", "C")])
        deepinfra_client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id=None,
        ))
        deepinfra_client.client.files.content = AsyncMock(return_value=SimpleNamespace(text=jsonl))

        results = await deepinfra_client.fetch_batch_results("batch_abc", batch_requests)

        assert [r.raw_text for r in results] == ["A", "B", "C"]
        assert all(r.is_success() for r in results)

    async def test_reassembles_by_custom_id_not_line_order(self, deepinfra_client, batch_requests):
        jsonl = "\n".join([_output_line("2", "C"), _output_line("0", "A"), _output_line("1", "B")])
        deepinfra_client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id=None,
        ))
        deepinfra_client.client.files.content = AsyncMock(return_value=SimpleNamespace(text=jsonl))

        results = await deepinfra_client.fetch_batch_results("batch_abc", batch_requests)

        assert [r.raw_text for r in results] == ["A", "B", "C"]

    async def test_a_failed_line_within_the_output_file_becomes_a_failure_response(self, deepinfra_client, batch_requests):
        """DeepInfra's own docs show failed lines mixed into the SAME
        output file (null response + populated error), unlike OpenAI's
        separate error_file_id -- both shapes must be handled."""
        jsonl = "\n".join([_output_line("0", "A"), _error_line("1", "quota exceeded"), _output_line("2", "C")])
        deepinfra_client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id=None,
        ))
        deepinfra_client.client.files.content = AsyncMock(return_value=SimpleNamespace(text=jsonl))

        results = await deepinfra_client.fetch_batch_results("batch_abc", batch_requests)

        assert results[0].is_success() and results[0].raw_text == "A"
        assert not results[1].is_success()
        assert "quota exceeded" in results[1].error.message
        assert results[2].is_success() and results[2].raw_text == "C"

    async def test_a_missing_custom_id_becomes_a_failure_response_not_a_crash(self, deepinfra_client, batch_requests):
        jsonl = "\n".join([_output_line("0", "A"), _output_line("2", "C")])
        deepinfra_client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id=None,
        ))
        deepinfra_client.client.files.content = AsyncMock(return_value=SimpleNamespace(text=jsonl))

        results = await deepinfra_client.fetch_batch_results("batch_abc", batch_requests)

        assert results[0].is_success() and results[2].is_success()
        assert not results[1].is_success()
