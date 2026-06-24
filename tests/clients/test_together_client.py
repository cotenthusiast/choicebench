# tests/clients/test_together_client.py

import asyncio
import httpx
import openai
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcq_eval.clients.together_client import TogetherAIClient
from mcq_eval.clients.types import (
    ModelRequest,
    RequestMetadata,
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
def request_metadata() -> RequestMetadata:
    return RequestMetadata(
        question_id="q1",
        split_name="robustness",
        method_name="baseline",
        subject="anatomy",
        run_id="run_001",
        prompt_version="v1",
        perturbation_name=None,
        sample_index=0,
    )


@pytest.fixture
def model_request(request_metadata: RequestMetadata) -> ModelRequest:
    return ModelRequest(
        provider="together",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        payload="test prompt",
        temperature=0.0,
        max_tokens=128,
        metadata=request_metadata,
    )


@pytest.fixture
def model_request_with_logprobs(request_metadata: RequestMetadata) -> ModelRequest:
    return ModelRequest(
        provider="together",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        payload="test prompt",
        temperature=0.0,
        max_tokens=128,
        metadata=request_metadata,
        request_logprobs=True,
    )


def _make_top_logprob_entry(token: str, logprob: float) -> SimpleNamespace:
    return SimpleNamespace(token=token, logprob=logprob)


def _make_token_logprob_entry(
    token: str, logprob: float, top_logprobs: list
) -> SimpleNamespace:
    return SimpleNamespace(token=token, logprob=logprob, top_logprobs=top_logprobs)


def _make_together_response_nonstandard_logprobs(
    *,
    text: str = "A",
    tokens: list,
    token_logprobs_list: list,
    top_logprobs: list,
) -> SimpleNamespace:
    """Together AI non-standard parallel-array logprob format."""
    choice_logprobs = SimpleNamespace(
        content=[],  # empty → client falls through to parallel-array path
        tokens=tokens,
        token_logprobs=token_logprobs_list,
        top_logprobs=top_logprobs,
    )
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(content=text),
            finish_reason="stop",
            logprobs=choice_logprobs,
        )
    ]
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=1, total_tokens=11)
    return SimpleNamespace(choices=choices, usage=usage)


def _make_together_response(
    *,
    text: str | None = "hello",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    include_choices: bool = True,
    include_usage: bool = True,
    logprobs_content: list | None = None,
) -> SimpleNamespace:
    choices = None
    if include_choices:
        choice_logprobs = None
        if logprobs_content is not None:
            choice_logprobs = SimpleNamespace(content=logprobs_content)
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
                logprobs=choice_logprobs,
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

    # ── logprobs ────────────────────────────────────────────────────────────

    def test_logprobs_none_when_not_requested(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            mock_create.return_value = _make_together_response()
            return await together_client._generate_provider_response(model_request)

        response = asyncio.run(_inner())
        assert response.logprobs is None

    def test_logprobs_populated_when_requested(
        self, together_client, model_request_with_logprobs, mock_create
    ) -> None:
        async def _inner():
            top_lps = [
                _make_top_logprob_entry("A", -0.1),
                _make_top_logprob_entry("B", -1.5),
            ]
            token_lps = [_make_token_logprob_entry("A", -0.1, top_lps)]
            mock_create.return_value = _make_together_response(logprobs_content=token_lps)
            return await together_client._generate_provider_response(model_request_with_logprobs)

        response = asyncio.run(_inner())
        assert response.logprobs is not None
        assert len(response.logprobs) == 1
        assert response.logprobs[0]["token"] == "A"
        assert response.logprobs[0]["logprob"] == pytest.approx(-0.1)
        assert len(response.logprobs[0]["top_logprobs"]) == 2

    def test_logprobs_top_logprobs_entries_have_token_and_logprob(
        self, together_client, model_request_with_logprobs, mock_create
    ) -> None:
        async def _inner():
            top_lps = [_make_top_logprob_entry("C", -0.5)]
            token_lps = [_make_token_logprob_entry("C", -0.5, top_lps)]
            mock_create.return_value = _make_together_response(logprobs_content=token_lps)
            return await together_client._generate_provider_response(model_request_with_logprobs)

        response = asyncio.run(_inner())
        entry = response.logprobs[0]["top_logprobs"][0]
        assert "token" in entry
        assert "logprob" in entry

    def test_logprobs_empty_list_when_content_empty(
        self, together_client, model_request_with_logprobs, mock_create
    ) -> None:
        async def _inner():
            mock_create.return_value = _make_together_response(logprobs_content=[])
            return await together_client._generate_provider_response(model_request_with_logprobs)

        response = asyncio.run(_inner())
        assert response.logprobs == []

    def test_logprobs_param_not_sent_when_not_requested(
        self, together_client, model_request, mock_create
    ) -> None:
        async def _inner():
            mock_create.return_value = _make_together_response()
            await together_client._generate_provider_response(model_request)

        asyncio.run(_inner())
        call_kwargs = mock_create.call_args.kwargs
        assert "logprobs" not in call_kwargs
        assert "top_logprobs" not in call_kwargs

    def test_logprobs_params_sent_when_requested(
        self, together_client, model_request_with_logprobs, mock_create
    ) -> None:
        async def _inner():
            mock_create.return_value = _make_together_response(logprobs_content=[])
            await together_client._generate_provider_response(model_request_with_logprobs)

        asyncio.run(_inner())
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs.get("logprobs") is True
        assert call_kwargs.get("top_logprobs") == 20
        # Assistant prefill must be appended so the first token is a letter.
        messages = call_kwargs.get("messages", [])
        assert len(messages) == 2
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "The answer is "

    def test_logprobs_populated_from_nonstandard_parallel_array_format(
        self, together_client, model_request_with_logprobs, mock_create
    ) -> None:
        """Together AI non-standard format: parallel arrays with top_logprobs as list of dicts."""
        async def _inner():
            mock_create.return_value = _make_together_response_nonstandard_logprobs(
                text="A",
                tokens=["A"],
                token_logprobs_list=[-0.1],
                top_logprobs=[{"A": -0.1, "B": -1.5, "C": -2.0, "D": -2.5}],
            )
            return await together_client._generate_provider_response(model_request_with_logprobs)

        response = asyncio.run(_inner())
        assert response.logprobs is not None
        assert len(response.logprobs) == 1
        assert response.logprobs[0]["token"] == "A"
        assert response.logprobs[0]["logprob"] == pytest.approx(-0.1)
        top = response.logprobs[0]["top_logprobs"]
        assert len(top) == 4
        assert {e["token"] for e in top} == {"A", "B", "C", "D"}

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
