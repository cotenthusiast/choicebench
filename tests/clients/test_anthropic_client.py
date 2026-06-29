# tests/clients/test_anthropic_client.py

import httpx
import pytest
import anthropic
from types import SimpleNamespace
from unittest.mock import AsyncMock

from choicebench.clients.anthropic_client import AnthropicClient
from choicebench.clients.types import (
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
