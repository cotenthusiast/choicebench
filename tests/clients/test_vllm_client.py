# tests/clients/test_vllm_client.py

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
from choicebench.backends.api_backend import APIBackend
from choicebench.clients.openai_client import OpenAIClient


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


def _make_logprob_response(logprob_entries: list[tuple[str, float]]):
    """Build a fake chat completion response with top_logprobs data."""
    top_lp = [
        SimpleNamespace(token=token, logprob=lp) for token, lp in logprob_entries
    ]
    content_lp = [SimpleNamespace(top_logprobs=top_lp)]
    choice = SimpleNamespace(
        message=SimpleNamespace(content="A"),
        finish_reason="stop",
        logprobs=SimpleNamespace(content=content_lp),
    )
    return SimpleNamespace(choices=[choice], usage=None)


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
# score_options_async tests
# ---------------------------------------------------------------------------

class TestVLLMClientScoreOptionsAsync:
    pytestmark = pytest.mark.asyncio

    async def test_returns_dict_of_option_logprobs(self, vllm_client):
        fake_response = _make_logprob_response([
            ("A", -0.1),
            ("B", -1.5),
            ("C", -2.0),
            ("D", -3.5),
        ])
        with patch.object(
            vllm_client, "score_options_async",
            new=AsyncMock(return_value={"A": -0.1, "B": -1.5, "C": -2.0, "D": -3.5}),
        ):
            result = await vllm_client.score_options_async(
                "Which is the capital of France?", ["A", "B", "C", "D"]
            )
        assert result == {"A": -0.1, "B": -1.5, "C": -2.0, "D": -3.5}

    async def test_floor_value_for_option_not_in_top_logprobs(self):
        # Build a client, patch the inner AsyncOpenAI call
        client = VLLMClient(model_name="test-model")
        fake_response = _make_logprob_response([("A", -0.5), ("B", -1.0)])

        with patch("choicebench.clients.vllm_client.AsyncOpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=fake_response)
            mock_instance.aclose = AsyncMock()
            mock_cls.return_value = mock_instance

            result = await client.score_options_async("prompt", ["A", "B", "C", "D"])

        assert result["A"] == -0.5
        assert result["B"] == -1.0
        assert result["C"] == -100.0  # floor: not in top logprobs
        assert result["D"] == -100.0

    async def test_strips_space_prefixed_tokens(self):
        # vLLM sometimes returns " A" (with leading space) for the option token.
        client = VLLMClient(model_name="test-model")
        fake_response = _make_logprob_response([(" A", -0.2), (" B", -1.8)])

        with patch("choicebench.clients.vllm_client.AsyncOpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=fake_response)
            mock_instance.aclose = AsyncMock()
            mock_cls.return_value = mock_instance

            result = await client.score_options_async("prompt", ["A", "B"])

        assert result["A"] == -0.2
        assert result["B"] == -1.8

    async def test_raises_on_empty_logprob_content(self):
        client = VLLMClient(model_name="test-model")
        # Response with logprobs but empty content list
        choice = SimpleNamespace(
            message=SimpleNamespace(content="A"),
            finish_reason="stop",
            logprobs=SimpleNamespace(content=[]),
        )
        fake_response = SimpleNamespace(choices=[choice], usage=None)

        with patch("choicebench.clients.vllm_client.AsyncOpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=fake_response)
            mock_instance.aclose = AsyncMock()
            mock_cls.return_value = mock_instance

            with pytest.raises(ProviderResponseError):
                await client.score_options_async("prompt", ["A", "B"])


# ---------------------------------------------------------------------------
# APIBackend.supports_score_options tests
# ---------------------------------------------------------------------------

class TestAPIBackendSupportsScoreOptions:

    def _make_api_backend(self, client, tmp_path: Path) -> APIBackend:
        return APIBackend(
            provider=client.provider,
            model_name=client.model_name,
            client=client,
            cache_dir=tmp_path / "cache",
            temperature=0.0,
            max_tokens=512,
            seed=42,
        )

    def test_supports_score_options_true_for_vllm(self, tmp_path):
        client = VLLMClient(model_name="meta-llama/Llama-3.1-8B-Instruct")
        backend = self._make_api_backend(client, tmp_path)
        assert backend.supports_score_options() is True

    def test_supports_score_options_false_for_openai(self, tmp_path):
        client = OpenAIClient(model_name="gpt-4o-mini", api_key="test-key")
        backend = self._make_api_backend(client, tmp_path)
        assert backend.supports_score_options() is False

    def test_supports_logprobs_true_for_vllm(self, tmp_path):
        client = VLLMClient(model_name="meta-llama/Llama-3.1-8B-Instruct")
        backend = self._make_api_backend(client, tmp_path)
        assert backend.supports_logprobs is True

    def test_supports_logprobs_false_for_openai(self, tmp_path):
        client = OpenAIClient(model_name="gpt-4o-mini", api_key="test-key")
        backend = self._make_api_backend(client, tmp_path)
        assert backend.supports_logprobs is False

    def test_score_options_raises_for_non_vllm(self, tmp_path):
        client = OpenAIClient(model_name="gpt-4o-mini", api_key="test-key")
        backend = self._make_api_backend(client, tmp_path)
        with pytest.raises(NotImplementedError):
            backend.score_options("prompt", ["A", "B"])

    def test_score_options_calls_score_options_async_for_vllm(self, tmp_path):
        client = VLLMClient(model_name="meta-llama/Llama-3.1-8B-Instruct")
        backend = self._make_api_backend(client, tmp_path)

        async def _fake_async(prompt, options):
            return {"A": -0.5, "B": -1.0, "C": -2.0, "D": -3.0}

        with patch.object(client, "score_options_async", side_effect=_fake_async):
            result = backend.score_options("some prompt", ["A", "B", "C", "D"])

        assert result == [-0.5, -1.0, -2.0, -3.0]
