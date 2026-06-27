# tests/clients/test_openai_client.py

import httpx
import pytest
import openai
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcq_eval.clients.openai_client import OpenAIClient
from mcq_eval.clients.types import (
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
