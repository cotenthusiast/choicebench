# tests/clients/test_groq_client.py

import httpx
import pytest
import groq
from types import SimpleNamespace
from unittest.mock import AsyncMock

from choicebench.clients.groq_client import GroqClient
from choicebench.clients.types import (
    ModelRequest,
    ProviderResponseError,
    ProviderCallError,
    ProviderConfigurationError,
)


@pytest.fixture
def groq_client() -> GroqClient:
    return GroqClient(model_name="llama-3.3-70b-versatile", api_key="test-key")


@pytest.fixture
def model_request() -> ModelRequest:
    return ModelRequest(
        provider="groq", model_name="llama-3.3-70b-versatile",
        payload="test prompt", temperature=0.2, max_tokens=128,
    )


@pytest.fixture
def make_groq_response():
    def _make_response(
        *, text="hello", finish_reason="stop",
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
        include_choices=True, include_usage=True,
    ):
        choices = None
        if include_choices:
            choices = [SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)]
        usage = None
        if include_usage:
            usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
        return SimpleNamespace(choices=choices, usage=usage)
    return _make_response


@pytest.fixture
def mock_create(groq_client: GroqClient) -> AsyncMock:
    mock = AsyncMock()
    groq_client.client.chat.completions.create = mock
    return mock


class TestGroqClientGenerateProviderResponse:
    """Tests for GroqClient._generate_provider_response."""

    pytestmark = pytest.mark.asyncio

    async def test_returns_model_response_on_success(self, groq_client, model_request, make_groq_response, mock_create) -> None:
        mock_create.return_value = make_groq_response()
        response = await groq_client._generate_provider_response(model_request)
        assert response.raw_text == "hello"
        assert response.provider == "groq"
        assert response.finish_reason == "stop"

    async def test_raises_provider_response_error_when_text_is_none(self, groq_client, model_request, make_groq_response, mock_create) -> None:
        mock_create.return_value = make_groq_response(text=None)
        with pytest.raises(ProviderResponseError):
            await groq_client._generate_provider_response(model_request)

    async def test_raises_provider_response_error_when_text_is_whitespace(self, groq_client, model_request, make_groq_response, mock_create) -> None:
        mock_create.return_value = make_groq_response(text="   ")
        with pytest.raises(ProviderResponseError):
            await groq_client._generate_provider_response(model_request)

    async def test_allows_missing_usage(self, groq_client, model_request, make_groq_response, mock_create) -> None:
        mock_create.return_value = make_groq_response(include_usage=False)
        response = await groq_client._generate_provider_response(model_request)
        assert response.usage is None

    async def test_raises_provider_response_error_when_choices_are_missing(self, groq_client, model_request, make_groq_response, mock_create) -> None:
        mock_create.return_value = make_groq_response(include_choices=False)
        with pytest.raises(ProviderResponseError):
            await groq_client._generate_provider_response(model_request)

    async def test_raises_provider_configuration_error_for_404(self, groq_client, model_request, mock_create) -> None:
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        http_response = httpx.Response(404, request=request)
        mock_create.side_effect = groq.NotFoundError("not found", response=http_response, body={"error": {"message": "not found"}})
        with pytest.raises(ProviderConfigurationError):
            await groq_client._generate_provider_response(model_request)

    async def test_raises_provider_call_error_for_500(self, groq_client, model_request, mock_create) -> None:
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        http_response = httpx.Response(500, request=request)
        mock_create.side_effect = groq.InternalServerError("server error", response=http_response, body={"error": {"message": "server error"}})
        with pytest.raises(ProviderCallError):
            await groq_client._generate_provider_response(model_request)
