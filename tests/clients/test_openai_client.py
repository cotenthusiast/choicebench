# tests/clients/test_openai_client.py

import json

import httpx
import pytest
import openai
from types import SimpleNamespace
from unittest.mock import AsyncMock

from choicebench.clients.openai_client import OpenAIClient
from choicebench.clients.types import (
    BATCH_COMPLETED,
    BATCH_FAILED,
    BATCH_IN_PROGRESS,
    ModelRequest,
    ProviderResponseError,
    ProviderCallError,
    ProviderConfigurationError,
)


@pytest.fixture
def openai_client() -> OpenAIClient:
    return OpenAIClient(model_name="gpt-4o-mini", api_key="test-key")


@pytest.fixture
def model_request() -> ModelRequest:
    return ModelRequest(
        provider="openai", model_name="gpt-4o-mini",
        payload="test prompt", temperature=0.2, max_tokens=128,
    )


@pytest.fixture
def make_openai_response():
    def _make_response(
        *, text="hello", prompt_tokens=10, completion_tokens=5,
        total_tokens=15, include_usage=True,
    ):
        usage = None
        if include_usage:
            usage = SimpleNamespace(input_tokens=prompt_tokens, output_tokens=completion_tokens, total_tokens=total_tokens)
        return SimpleNamespace(output_text=text, usage=usage)
    return _make_response


@pytest.fixture
def mock_create(openai_client: OpenAIClient) -> AsyncMock:
    mock = AsyncMock()
    openai_client.client.responses.create = mock
    return mock


class TestOpenAIClientGenerateProviderResponse:
    """Tests for OpenAIClient._generate_provider_response."""

    pytestmark = pytest.mark.asyncio

    async def test_returns_model_response_on_success(self, openai_client, model_request, make_openai_response, mock_create) -> None:
        mock_create.return_value = make_openai_response()
        response = await openai_client._generate_provider_response(model_request)
        assert response.raw_text == "hello"
        assert response.provider == "openai"

    async def test_raises_provider_response_error_when_text_is_none(self, openai_client, model_request, make_openai_response, mock_create) -> None:
        mock_create.return_value = make_openai_response(text=None)
        with pytest.raises(ProviderResponseError):
            await openai_client._generate_provider_response(model_request)

    async def test_raises_provider_response_error_when_text_is_whitespace(self, openai_client, model_request, make_openai_response, mock_create) -> None:
        mock_create.return_value = make_openai_response(text="   ")
        with pytest.raises(ProviderResponseError):
            await openai_client._generate_provider_response(model_request)

    async def test_allows_missing_usage(self, openai_client, model_request, make_openai_response, mock_create) -> None:
        mock_create.return_value = make_openai_response(include_usage=False)
        response = await openai_client._generate_provider_response(model_request)
        assert response.usage is None

    async def test_raises_provider_configuration_error_for_404(self, openai_client, model_request, mock_create) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        http_response = httpx.Response(404, request=request)
        mock_create.side_effect = openai.NotFoundError("not found", response=http_response, body={"message": "not found"})
        with pytest.raises(ProviderConfigurationError):
            await openai_client._generate_provider_response(model_request)

    async def test_raises_provider_call_error_for_500(self, openai_client, model_request, mock_create) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        http_response = httpx.Response(500, request=request)
        mock_create.side_effect = openai.InternalServerError("server error", response=http_response, body={"message": "server error"})
        with pytest.raises(ProviderCallError):
            await openai_client._generate_provider_response(model_request)


def _responses_api_body(text: str, model: str = "gpt-4.1-mini") -> dict:
    """A minimal raw JSON body shaped like the OpenAI Responses API --
    NOT the SDK's computed .output_text convenience property, the real
    output/content array structure batch results come back as."""
    return {
        "id": "resp_abc", "object": "response", "created_at": 1234567890,
        "status": "completed", "model": model,
        "output": [{
            "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _output_line(custom_id: str, text: str) -> str:
    return json.dumps({
        "id": f"batch_req_{custom_id}", "custom_id": custom_id,
        "response": {"status_code": 200, "request_id": "req", "body": _responses_api_body(text)},
        "error": None,
    })


def _error_line(custom_id: str, message: str) -> str:
    return json.dumps({
        "id": f"batch_req_{custom_id}", "custom_id": custom_id,
        "response": None, "error": {"code": "invalid_request", "message": message},
    })


@pytest.fixture
def batch_requests() -> list[ModelRequest]:
    return [
        ModelRequest(provider="openai", model_name="gpt-4.1-mini", payload=f"question {i}",
                     temperature=0.0, max_tokens=64)
        for i in range(3)
    ]


class TestOpenAIClientSubmitBatch:
    pytestmark = pytest.mark.asyncio

    async def test_uploads_a_jsonl_file_with_one_line_per_request(self, openai_client, batch_requests):
        openai_client.client.files.create = AsyncMock(return_value=SimpleNamespace(id="file_abc"))
        openai_client.client.batches.create = AsyncMock(return_value=SimpleNamespace(id="batch_abc"))

        batch_id = await openai_client.submit_batch(batch_requests)

        assert batch_id == "batch_abc"
        upload_kwargs = openai_client.client.files.create.call_args.kwargs
        assert upload_kwargs["purpose"] == "batch"
        _, jsonl_bytes = upload_kwargs["file"]
        lines = jsonl_bytes.decode("utf-8").strip().splitlines()
        assert len(lines) == 3
        parsed = [json.loads(line) for line in lines]
        assert [p["custom_id"] for p in parsed] == ["0", "1", "2"]
        assert all(p["method"] == "POST" and p["url"] == "/v1/responses" for p in parsed)
        assert parsed[0]["body"]["input"] == "question 0"
        assert parsed[0]["body"]["model"] == "gpt-4.1-mini"

    async def test_submits_the_uploaded_file_to_the_responses_endpoint(self, openai_client, batch_requests):
        openai_client.client.files.create = AsyncMock(return_value=SimpleNamespace(id="file_abc"))
        create_batch = AsyncMock(return_value=SimpleNamespace(id="batch_abc"))
        openai_client.client.batches.create = create_batch

        await openai_client.submit_batch(batch_requests)

        create_batch.assert_awaited_once_with(
            input_file_id="file_abc", endpoint="/v1/responses", completion_window="24h",
        )


class TestOpenAIClientPollBatch:
    pytestmark = pytest.mark.asyncio

    @pytest.mark.parametrize("raw_status,expected", [
        ("completed", BATCH_COMPLETED),
        ("validating", BATCH_IN_PROGRESS),
        ("in_progress", BATCH_IN_PROGRESS),
        ("finalizing", BATCH_IN_PROGRESS),
        ("cancelling", BATCH_IN_PROGRESS),
        ("failed", BATCH_FAILED),
        ("expired", BATCH_FAILED),
        ("cancelled", BATCH_FAILED),
    ])
    async def test_maps_openai_status_to_normalized_status(self, openai_client, raw_status, expected):
        openai_client.client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(status=raw_status)
        )
        assert await openai_client.poll_batch("batch_abc") == expected


class TestOpenAIClientFetchBatchResults:
    pytestmark = pytest.mark.asyncio

    async def test_parses_successful_lines_from_the_output_file(self, openai_client, batch_requests):
        jsonl = "\n".join([
            _output_line("0", "A"), _output_line("1", "B"), _output_line("2", "C"),
        ])
        openai_client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id=None,
        ))
        openai_client.client.files.content = AsyncMock(return_value=SimpleNamespace(text=jsonl))

        results = await openai_client.fetch_batch_results("batch_abc", batch_requests)

        assert [r.raw_text for r in results] == ["A", "B", "C"]
        assert all(r.is_success() for r in results)

    async def test_reassembles_by_custom_id_not_line_order(self, openai_client, batch_requests):
        # Deliberately out of order -- matches real provider behavior.
        jsonl = "\n".join([_output_line("2", "C"), _output_line("0", "A"), _output_line("1", "B")])
        openai_client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id=None,
        ))
        openai_client.client.files.content = AsyncMock(return_value=SimpleNamespace(text=jsonl))

        results = await openai_client.fetch_batch_results("batch_abc", batch_requests)

        assert [r.raw_text for r in results] == ["A", "B", "C"]

    async def test_a_line_in_the_error_file_becomes_a_failure_response(self, openai_client, batch_requests):
        output_jsonl = "\n".join([_output_line("0", "A"), _output_line("2", "C")])
        error_jsonl = _error_line("1", "content policy violation")
        openai_client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id="err_1",
        ))

        async def _fake_content(file_id):
            return SimpleNamespace(text=output_jsonl if file_id == "out_1" else error_jsonl)

        openai_client.client.files.content = AsyncMock(side_effect=_fake_content)

        results = await openai_client.fetch_batch_results("batch_abc", batch_requests)

        assert results[0].is_success() and results[0].raw_text == "A"
        assert not results[1].is_success()
        assert "content policy violation" in results[1].error.message
        assert results[2].is_success() and results[2].raw_text == "C"

    async def test_a_missing_custom_id_becomes_a_failure_response_not_a_crash(self, openai_client, batch_requests):
        jsonl = "\n".join([_output_line("0", "A"), _output_line("2", "C")])  # "1" missing entirely
        openai_client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id=None,
        ))
        openai_client.client.files.content = AsyncMock(return_value=SimpleNamespace(text=jsonl))

        results = await openai_client.fetch_batch_results("batch_abc", batch_requests)

        assert results[0].is_success() and results[2].is_success()
        assert not results[1].is_success()
