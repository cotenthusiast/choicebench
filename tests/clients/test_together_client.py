# tests/clients/test_together_client.py

import asyncio
import httpx
import openai
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock


from choicebench.clients.together_client import TogetherAIClient
from choicebench.clients.types import (
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
