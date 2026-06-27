# tests/clients/test_gemini_client.py

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from google.genai import errors

from mcq_eval.clients.gemini_client import GeminiClient
from mcq_eval.clients.types import (
    ModelRequest,
    ProviderResponseError,
    ProviderCallError,
    ProviderConfigurationError,
)


@pytest.fixture
def gemini_client() -> GeminiClient:
    return GeminiClient(model_name="gemini-2.0-flash")


@pytest.fixture
def model_request() -> ModelRequest:
    return ModelRequest(
        provider="gemini", model_name="gemini-2.0-flash",
        payload="test prompt", temperature=0.2, max_tokens=128,
    )


@pytest.fixture
def make_gemini_response():
    def _make_response(
        *, text="hello", finish_reason="STOP",
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
        include_candidates=True, include_usage=True,
    ):
        candidates = None
        if include_candidates:
            candidates = [SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))]
        usage_metadata = None
        if include_usage:
            usage_metadata = SimpleNamespace(
                prompt_token_count=prompt_tokens,
                candidates_token_count=completion_tokens,
                total_token_count=total_tokens,
            )
        return SimpleNamespace(text=text, candidates=candidates, usage_metadata=usage_metadata)
    return _make_response


@pytest.fixture
def mock_generate_content(gemini_client: GeminiClient) -> AsyncMock:
    mock = AsyncMock()
    gemini_client.client.aio.models.generate_content = mock
    return mock


class TestGeminiClientGenerateProviderResponse:
    """Tests for GeminiClient._generate_provider_response."""

    pytestmark = pytest.mark.asyncio

    async def test_returns_model_response_on_success(
        self, gemini_client, model_request, make_gemini_response, mock_generate_content,
    ) -> None:
        mock_generate_content.return_value = make_gemini_response()
        response = await gemini_client._generate_provider_response(model_request)
        assert response.raw_text == "hello"
        assert response.provider == "gemini"

    async def test_raises_provider_response_error_when_text_is_none(
        self, gemini_client, model_request, make_gemini_response, mock_generate_content,
    ) -> None:
        fake = make_gemini_response()
        fake.text = None
        mock_generate_content.return_value = fake
        with pytest.raises(ProviderResponseError):
            await gemini_client._generate_provider_response(model_request)

    async def test_raises_provider_response_error_when_text_is_whitespace(
        self, gemini_client, model_request, make_gemini_response, mock_generate_content,
    ) -> None:
        fake = make_gemini_response()
        fake.text = " "
        mock_generate_content.return_value = fake
        with pytest.raises(ProviderResponseError):
            await gemini_client._generate_provider_response(model_request)

    async def test_allows_missing_candidates(
        self, gemini_client, model_request, make_gemini_response, mock_generate_content,
    ) -> None:
        fake = make_gemini_response()
        fake.candidates = None
        mock_generate_content.return_value = fake
        response = await gemini_client._generate_provider_response(model_request)
        assert response.raw_text == "hello"
        assert response.finish_reason is None

    async def test_allows_missing_usage_metadata(
        self, gemini_client, model_request, make_gemini_response, mock_generate_content,
    ) -> None:
        fake = make_gemini_response()
        fake.usage_metadata = None
        mock_generate_content.return_value = fake
        response = await gemini_client._generate_provider_response(model_request)
        assert response.usage is None
        assert response.finish_reason == "STOP"

    async def test_falls_back_to_total_minus_prompt_for_completion_tokens(
        self, gemini_client, model_request, make_gemini_response, mock_generate_content,
    ) -> None:
        fake = make_gemini_response()
        fake.usage_metadata.candidates_token_count = None
        fake.usage_metadata.response_token_count = None
        fake.usage_metadata.prompt_token_count = 10
        fake.usage_metadata.total_token_count = 15
        mock_generate_content.return_value = fake
        response = await gemini_client._generate_provider_response(model_request)
        assert response.usage.completion_tokens == 5

    async def test_raises_provider_configuration_error_for_404(
        self, gemini_client, model_request, mock_generate_content,
    ) -> None:
        mock_generate_content.side_effect = errors.ClientError(404, {"message": "not found"}, None)
        with pytest.raises(ProviderConfigurationError):
            await gemini_client._generate_provider_response(model_request)

    async def test_raises_provider_call_error_for_500(
        self, gemini_client, model_request, mock_generate_content,
    ) -> None:
        mock_generate_content.side_effect = errors.ServerError(500, {"message": "server error"}, None)
        with pytest.raises(ProviderCallError):
            await gemini_client._generate_provider_response(model_request)
