# tests/clients/test_vllm_client.py

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from choicebench.clients.vllm_client import VLLMClient, _DEFAULT_BASE_URL, _DEFAULT_API_KEY
from choicebench.clients.types import (
    ModelRequest,
    ProviderResponseError,
    ProviderCallError,
    ProviderConfigurationError,
)


# ---------------------------------------------------------------------------
# Helpers for building fake chat completion responses
# ---------------------------------------------------------------------------

def _make_chat_response(
    *,
    text: str = "B",
    prompt_tokens: int = 10,
    completion_tokens: int = 1,
    total_tokens: int = 11,
    include_usage: bool = True,
    finish_reason: str = "stop",
):
    usage = None
    if include_usage:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text),
        finish_reason=finish_reason,
        logprobs=None,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vllm_client():
    return VLLMClient(model_name="meta-llama/Llama-3.1-8B-Instruct")


@pytest.fixture
def vllm_client_custom_url():
    return VLLMClient(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://gpu-server:9000/v1",
    )


@pytest.fixture
def model_request():
    return ModelRequest(
        provider="vllm",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        payload="Which planet is closest to the Sun?\nA. Earth\nB. Mercury\nC. Venus\nD. Mars",
        temperature=0.0,
        max_tokens=16,
    )


@pytest.fixture
def mock_chat_create(vllm_client):
    mock = AsyncMock()
    vllm_client.client.chat.completions.create = mock
    return mock


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestVLLMClientConstruction:

    def test_default_base_url(self, vllm_client):
        assert vllm_client._base_url == _DEFAULT_BASE_URL

    def test_custom_base_url(self, vllm_client_custom_url):
        assert vllm_client_custom_url._base_url == "http://gpu-server:9000/v1"

    def test_api_key_defaults_to_vllm_string(self, monkeypatch):
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        client = VLLMClient(model_name="test-model")
        assert client._api_key == _DEFAULT_API_KEY
        assert client._api_key == "vllm"

    def test_api_key_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "my-secret-key")
        client = VLLMClient(model_name="test-model")
        assert client._api_key == "my-secret-key"

    def test_provider_is_vllm(self, vllm_client):
        assert vllm_client.provider == "vllm"

    def test_model_name_stored(self, vllm_client):
        assert vllm_client.model_name == "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# Generation tests
# ---------------------------------------------------------------------------

class TestVLLMClientGenerate:
    pytestmark = pytest.mark.asyncio

    async def test_returns_correct_text(self, vllm_client, model_request, mock_chat_create):
        mock_chat_create.return_value = _make_chat_response(text="B")
        response = await vllm_client._generate_provider_response(model_request)
        assert response.raw_text == "B"
        assert response.provider == "vllm"

    async def test_raises_provider_response_error_on_empty_text(
        self, vllm_client, model_request, mock_chat_create
    ):
        mock_chat_create.return_value = _make_chat_response(text="")
        with pytest.raises(ProviderResponseError):
            await vllm_client._generate_provider_response(model_request)

    async def test_raises_provider_response_error_on_whitespace_text(
        self, vllm_client, model_request, mock_chat_create
    ):
        mock_chat_create.return_value = _make_chat_response(text="   ")
        with pytest.raises(ProviderResponseError):
            await vllm_client._generate_provider_response(model_request)

    async def test_allows_missing_usage(
        self, vllm_client, model_request, mock_chat_create
    ):
        mock_chat_create.return_value = _make_chat_response(include_usage=False)
        response = await vllm_client._generate_provider_response(model_request)
        assert response.usage is None

    async def test_usage_populated_when_present(
        self, vllm_client, model_request, mock_chat_create
    ):
        mock_chat_create.return_value = _make_chat_response(
            prompt_tokens=20, completion_tokens=3, total_tokens=23
        )
        response = await vllm_client._generate_provider_response(model_request)
        assert response.usage is not None
        assert response.usage.prompt_tokens == 20
        assert response.usage.completion_tokens == 3

    async def test_raises_provider_call_error_for_500(
        self, vllm_client, model_request, mock_chat_create
    ):
        request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
        http_response = httpx.Response(500, request=request)
        mock_chat_create.side_effect = openai.InternalServerError(
            "server error", response=http_response, body={"message": "server error"}
        )
        with pytest.raises(ProviderCallError):
            await vllm_client._generate_provider_response(model_request)

    async def test_raises_provider_configuration_error_for_404(
        self, vllm_client, model_request, mock_chat_create
    ):
        request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
        http_response = httpx.Response(404, request=request)
        mock_chat_create.side_effect = openai.NotFoundError(
            "not found", response=http_response, body={"message": "not found"}
        )
        with pytest.raises(ProviderConfigurationError):
            await vllm_client._generate_provider_response(model_request)


# ---------------------------------------------------------------------------
# VLLMClient mirrors TogetherAIClient: generate-only, no logprob capability.
# ---------------------------------------------------------------------------

class TestVLLMClientHasNoScoreOptions:

    def test_no_score_options_async_method(self, vllm_client):
        assert not hasattr(vllm_client, "score_options_async")

    def test_backend_supports_logprobs_false_for_vllm(self, tmp_path):
        from choicebench.backends.api_backend import APIBackend

        client = VLLMClient(model_name="meta-llama/Llama-3.1-8B-Instruct")
        backend = APIBackend(
            provider=client.provider,
            model_name=client.model_name,
            client=client,
            cache_dir=tmp_path / "cache",
            temperature=0.0,
            max_tokens=512,
            seed=42,
        )
        assert backend.supports_logprobs is False

    def test_backend_score_options_raises_not_implemented_for_vllm(self, tmp_path):
        from choicebench.backends.api_backend import APIBackend

        client = VLLMClient(model_name="meta-llama/Llama-3.1-8B-Instruct")
        backend = APIBackend(
            provider=client.provider,
            model_name=client.model_name,
            client=client,
            cache_dir=tmp_path / "cache",
            temperature=0.0,
            max_tokens=512,
            seed=42,
        )
        with pytest.raises(NotImplementedError):
            backend.score_options("prompt", ["A", "B"])
