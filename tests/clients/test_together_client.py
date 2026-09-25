# tests/clients/test_together_client.py

import asyncio
import json
from pathlib import Path

import httpx
import openai
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock


from choicebench.clients.together_client import TogetherAIClient
from choicebench.clients.types import (
    BATCH_COMPLETED,
    BATCH_FAILED,
    BATCH_IN_PROGRESS,
    ModelRequest,
    ProviderCallError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)

_TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"


@pytest.fixture
def together_client() -> TogetherAIClient:
    return TogetherAIClient(model_name="Qwen/Qwen2.5-7B-Instruct", api_key="test-key")


@pytest.fixture
def model_request() -> ModelRequest:
    return ModelRequest(
        provider="together",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        payload="test prompt",
        temperature=0.0,
        max_tokens=128,
    )



def _make_together_response(
    *,
    text: str | None = "hello",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    include_choices: bool = True,
    include_usage: bool = True,
) -> SimpleNamespace:
    choices = None
    if include_choices:
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ]
    usage = None
    if include_usage:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    return SimpleNamespace(choices=choices, usage=usage)


@pytest.fixture
def mock_create(together_client: TogetherAIClient) -> AsyncMock:
    mock = AsyncMock()
    together_client.client.chat.completions.create = mock
    return mock


class TestTogetherAIClientGenerateProviderResponse:
    """Tests for TogetherAIClient._generate_provider_response."""

    # ── happy path ──────────────────────────────────────────────────────────

    def test_returns_model_response_on_success(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            mock_create.return_value = _make_together_response()
            return await together_client._generate_provider_response(model_request)

        response = asyncio.run(_inner())
        assert response.raw_text == "hello"
        assert response.provider == "together"
        assert response.finish_reason == "stop"

    def test_usage_fields_populated(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            mock_create.return_value = _make_together_response(
                prompt_tokens=20, completion_tokens=7, total_tokens=27
            )
            return await together_client._generate_provider_response(model_request)

        response = asyncio.run(_inner())
        assert response.usage is not None
        assert response.usage.prompt_tokens == 20
        assert response.usage.completion_tokens == 7
        assert response.usage.total_tokens == 27

    def test_missing_usage_gives_none(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            mock_create.return_value = _make_together_response(include_usage=False)
            return await together_client._generate_provider_response(model_request)

        response = asyncio.run(_inner())
        assert response.usage is None

    # ── error / empty responses ─────────────────────────────────────────────

    def test_raises_when_text_is_none(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            mock_create.return_value = _make_together_response(text=None)
            await together_client._generate_provider_response(model_request)

        with pytest.raises(ProviderResponseError):
            asyncio.run(_inner())

    def test_raises_when_text_is_whitespace(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            mock_create.return_value = _make_together_response(text="   ")
            await together_client._generate_provider_response(model_request)

        with pytest.raises(ProviderResponseError):
            asyncio.run(_inner())

    def test_raises_when_choices_missing(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            mock_create.return_value = _make_together_response(include_choices=False)
            await together_client._generate_provider_response(model_request)

        with pytest.raises(ProviderResponseError):
            asyncio.run(_inner())

    # ── provider error mapping ──────────────────────────────────────────────

    def test_raises_provider_rate_limit_error_for_429(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            request = httpx.Request("POST", _TOGETHER_CHAT_URL)
            http_response = httpx.Response(429, request=request)
            mock_create.side_effect = openai.RateLimitError(
                "rate limit", response=http_response, body={"message": "rate limit"}
            )
            await together_client._generate_provider_response(model_request)

        with pytest.raises(ProviderRateLimitError):
            asyncio.run(_inner())

    def test_raises_provider_configuration_error_for_401(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            request = httpx.Request("POST", _TOGETHER_CHAT_URL)
            http_response = httpx.Response(401, request=request)
            mock_create.side_effect = openai.AuthenticationError(
                "unauthorized", response=http_response, body={"message": "unauthorized"}
            )
            await together_client._generate_provider_response(model_request)

        with pytest.raises(ProviderConfigurationError):
            asyncio.run(_inner())

    def test_raises_provider_configuration_error_for_404(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            request = httpx.Request("POST", _TOGETHER_CHAT_URL)
            http_response = httpx.Response(404, request=request)
            mock_create.side_effect = openai.NotFoundError(
                "not found", response=http_response, body={"message": "not found"}
            )
            await together_client._generate_provider_response(model_request)

        with pytest.raises(ProviderConfigurationError):
            asyncio.run(_inner())

    def test_raises_provider_call_error_for_500(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            request = httpx.Request("POST", _TOGETHER_CHAT_URL)
            http_response = httpx.Response(500, request=request)
            mock_create.side_effect = openai.InternalServerError(
                "server error", response=http_response, body={"message": "server error"}
            )
            await together_client._generate_provider_response(model_request)

        with pytest.raises(ProviderCallError):
            asyncio.run(_inner())

    def test_raises_provider_timeout_error(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            mock_create.side_effect = openai.APITimeoutError("timeout")
            await together_client._generate_provider_response(model_request)

        with pytest.raises(ProviderTimeoutError):
            asyncio.run(_inner())

    def test_raises_provider_call_error_on_connection_error(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            mock_create.side_effect = openai.APIConnectionError(
                message="connection failed",
                request=httpx.Request("POST", _TOGETHER_CHAT_URL),
            )
            await together_client._generate_provider_response(model_request)

        with pytest.raises(ProviderCallError):
            asyncio.run(_inner())


def _chat_completion_body(text: str, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo") -> dict:
    """Raw JSON body shaped like an OpenAI-compatible Chat Completions
    response -- what a Together batch output line's response.body contains."""
    return {
        "id": "chatcmpl-abc", "object": "chat.completion", "created": 123, "model": model,
        "choices": [{
            "index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop",
        }],
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
        "custom_id": custom_id, "response": None,
        "error": {"message": message, "type": "invalid_request_error"},
    })


@pytest.fixture
def batch_requests() -> list[ModelRequest]:
    return [
        ModelRequest(provider="together", model_name="Qwen/Qwen2.5-7B-Instruct", payload=f"question {i}",
                     temperature=0.0, max_tokens=64)
        for i in range(3)
    ]


class TestTogetherClientSubmitBatch:
    pytestmark = pytest.mark.asyncio

    async def test_uploads_a_jsonl_file_and_creates_the_batch_job(self, together_client, batch_requests):
        upload = AsyncMock(return_value=SimpleNamespace(id="file_abc"))
        create = AsyncMock(return_value=SimpleNamespace(job=SimpleNamespace(id="batch_abc")))
        together_client._batch_client.files.upload = upload
        together_client._batch_client.batches.create = create

        batch_id = await together_client.submit_batch(batch_requests)

        assert batch_id == "batch_abc"
        assert upload.call_args.kwargs["purpose"] == "batch-api"
        create.assert_awaited_once()
        assert create.call_args.kwargs["input_file_id"] == "file_abc"
        assert create.call_args.kwargs["endpoint"] == "/v1/chat/completions"

    async def test_jsonl_file_has_one_line_per_request_with_stringified_index_custom_id(
        self, together_client, batch_requests,
    ):
        captured = {}

        async def _fake_upload(file, **kwargs):
            # submit_batch() deletes the temp file in its own finally block
            # once upload "returns" -- read it here, while it still exists,
            # not after submit_batch() has already returned.
            captured["content"] = Path(file).read_text()
            return SimpleNamespace(id="file_abc")

        together_client._batch_client.files.upload = AsyncMock(side_effect=_fake_upload)
        together_client._batch_client.batches.create = AsyncMock(
            return_value=SimpleNamespace(job=SimpleNamespace(id="batch_abc"))
        )

        await together_client.submit_batch(batch_requests)

        lines = captured["content"].strip().splitlines()
        assert len(lines) == 3
        parsed = [json.loads(line) for line in lines]
        assert [p["custom_id"] for p in parsed] == ["0", "1", "2"]
        assert all(p["url"] == "/v1/chat/completions" for p in parsed)
        assert parsed[0]["body"]["messages"] == [{"role": "user", "content": "question 0"}]


class TestTogetherClientPollBatch:
    pytestmark = pytest.mark.asyncio

    @pytest.mark.parametrize("raw_status,expected", [
        ("COMPLETED", BATCH_COMPLETED),
        ("VALIDATING", BATCH_IN_PROGRESS),
        ("IN_PROGRESS", BATCH_IN_PROGRESS),
        ("FAILED", BATCH_FAILED),
        ("EXPIRED", BATCH_FAILED),
        ("CANCELLED", BATCH_FAILED),
    ])
    async def test_maps_together_status_to_normalized_status(self, together_client, raw_status, expected):
        together_client._batch_client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(status=raw_status)
        )
        assert await together_client.poll_batch("batch_abc") == expected


class TestTogetherClientFetchBatchResults:
    pytestmark = pytest.mark.asyncio

    async def test_parses_successful_lines_from_the_output_file(self, together_client, batch_requests):
        jsonl = "\n".join([_output_line("0", "A"), _output_line("1", "B"), _output_line("2", "C")])
        together_client._batch_client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id=None,
        ))
        together_client._batch_client.files.content = AsyncMock(return_value=SimpleNamespace(text=jsonl))

        results = await together_client.fetch_batch_results("batch_abc", batch_requests)

        assert [r.raw_text for r in results] == ["A", "B", "C"]
        assert all(r.is_success() for r in results)

    async def test_reassembles_by_custom_id_not_line_order(self, together_client, batch_requests):
        jsonl = "\n".join([_output_line("2", "C"), _output_line("0", "A"), _output_line("1", "B")])
        together_client._batch_client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id=None,
        ))
        together_client._batch_client.files.content = AsyncMock(return_value=SimpleNamespace(text=jsonl))

        results = await together_client.fetch_batch_results("batch_abc", batch_requests)

        assert [r.raw_text for r in results] == ["A", "B", "C"]

    async def test_a_line_in_the_error_file_becomes_a_failure_response(self, together_client, batch_requests):
        output_jsonl = "\n".join([_output_line("0", "A"), _output_line("2", "C")])
        error_jsonl = _error_line("1", "rate limited")
        together_client._batch_client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id="err_1",
        ))

        async def _fake_content(file_id):
            return SimpleNamespace(text=output_jsonl if file_id == "out_1" else error_jsonl)

        together_client._batch_client.files.content = AsyncMock(side_effect=_fake_content)

        results = await together_client.fetch_batch_results("batch_abc", batch_requests)

        assert results[0].is_success() and results[0].raw_text == "A"
        assert not results[1].is_success()
        assert "rate limited" in results[1].error.message
        assert results[2].is_success() and results[2].raw_text == "C"

    async def test_a_missing_custom_id_becomes_a_failure_response_not_a_crash(self, together_client, batch_requests):
        jsonl = "\n".join([_output_line("0", "A"), _output_line("2", "C")])
        together_client._batch_client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
            output_file_id="out_1", error_file_id=None,
        ))
        together_client._batch_client.files.content = AsyncMock(return_value=SimpleNamespace(text=jsonl))

        results = await together_client.fetch_batch_results("batch_abc", batch_requests)

        assert results[0].is_success() and results[2].is_success()
        assert not results[1].is_success()
