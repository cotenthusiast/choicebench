# tests/clients/test_openrouter_client.py

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from choicebench.clients.openrouter_client import OpenRouterClient
from choicebench.clients.types import (
    ModelRequest,
    ProviderResponseError,
)


@pytest.fixture
def openrouter_client() -> OpenRouterClient:
    return OpenRouterClient(model_name="meta-llama/llama-3.1-8b-instruct", api_key="test-key")


@pytest.fixture
def pinned_client() -> OpenRouterClient:
    return OpenRouterClient(
        model_name="meta-llama/llama-3.1-8b-instruct",
        api_key="test-key",
        upstream_provider="deepinfra",
        allow_fallbacks=False,
    )


@pytest.fixture
def model_request() -> ModelRequest:
    return ModelRequest(
        provider="openrouter",
        model_name="meta-llama/llama-3.1-8b-instruct",
        payload="test prompt",
        temperature=0.0,
        max_tokens=128,
    )


def _make_response(
    *,
    text: str | None = "hello",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    include_usage: bool = True,
    provider: str | None = None,
) -> SimpleNamespace:
    choices = [SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)]
    usage = None
    if include_usage:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens,
        )
    return SimpleNamespace(choices=choices, usage=usage, provider=provider)


@pytest.fixture
def mock_create(openrouter_client: OpenRouterClient) -> AsyncMock:
    mock = AsyncMock()
    openrouter_client.client.chat.completions.create = mock
    return mock


@pytest.fixture
def pinned_mock_create(pinned_client: OpenRouterClient) -> AsyncMock:
    mock = AsyncMock()
    pinned_client.client.chat.completions.create = mock
    return mock


class TestOpenRouterClientGenerateProviderResponse:
    def test_returns_model_response_on_success(self, openrouter_client, model_request, mock_create):
        async def _inner():
            mock_create.return_value = _make_response()
            return await openrouter_client._generate_provider_response(model_request)

        response = asyncio.run(_inner())
        assert response.raw_text == "hello"
        assert response.provider == "openrouter"
        assert response.finish_reason == "stop"

    def test_usage_fields_populated(self, openrouter_client, model_request, mock_create):
        async def _inner():
            mock_create.return_value = _make_response(prompt_tokens=20, completion_tokens=7, total_tokens=27)
            return await openrouter_client._generate_provider_response(model_request)

        response = asyncio.run(_inner())
        assert response.usage.prompt_tokens == 20
        assert response.usage.completion_tokens == 7
        assert response.usage.total_tokens == 27

    def test_raises_when_text_is_empty(self, openrouter_client, model_request, mock_create):
        async def _inner():
            mock_create.return_value = _make_response(text="")
            await openrouter_client._generate_provider_response(model_request)

        with pytest.raises(ProviderResponseError):
            asyncio.run(_inner())

    # ── provider-pinning behavior (the reason this client exists) ──────────

    def test_no_pinning_omits_extra_body_entirely(self, openrouter_client, model_request, mock_create):
        """A caller that never sets upstream_provider gets OpenRouter's default
        automatic routing -- no extra_body sent at all, not an empty one."""
        async def _inner():
            mock_create.return_value = _make_response()
            await openrouter_client._generate_provider_response(model_request)

        asyncio.run(_inner())
        call_kwargs = mock_create.call_args.kwargs
        assert "extra_body" not in call_kwargs

    def test_pinning_sends_provider_order_and_disables_fallback(
        self, pinned_client, model_request, pinned_mock_create
    ):
        async def _inner():
            pinned_mock_create.return_value = _make_response()
            await pinned_client._generate_provider_response(model_request)

        asyncio.run(_inner())
        call_kwargs = pinned_mock_create.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "provider": {"order": ["deepinfra"], "allow_fallbacks": False}
        }

    def test_pinning_uses_sync_routing_parameter_shape(self, pinned_client):
        """Regression guard: OpenRouter's sync pinning parameter is
        provider.order/allow_fallbacks -- must not accidentally use the
        batch-only provider.only shape."""
        extra_body = pinned_client._provider_routing_extra_body()
        assert "only" not in extra_body["provider"]
        assert extra_body["provider"]["order"] == ["deepinfra"]

    def test_default_allow_fallbacks_is_true_when_pinned_without_override(self):
        client = OpenRouterClient(
            model_name="m", api_key="k", upstream_provider="deepinfra",
        )
        assert client._provider_routing_extra_body()["provider"]["allow_fallbacks"] is True

    # ── actual upstream provider provenance ─────────────────────────────

    def test_captures_actual_upstream_provider_when_present(
        self, openrouter_client, model_request, mock_create
    ):
        async def _inner():
            mock_create.return_value = _make_response(provider="Phala")
            return await openrouter_client._generate_provider_response(model_request)

        response = asyncio.run(_inner())
        assert response.actual_upstream_provider == "Phala"

    def test_actual_upstream_provider_is_none_when_absent(
        self, openrouter_client, model_request, mock_create
    ):
        async def _inner():
            mock_create.return_value = _make_response(provider=None)
            return await openrouter_client._generate_provider_response(model_request)

        response = asyncio.run(_inner())
        assert response.actual_upstream_provider is None

    def test_pinned_matching_actual_provider_is_accepted(
        self, pinned_client, model_request, pinned_mock_create
    ):
        async def _inner():
            pinned_mock_create.return_value = _make_response(provider="deepinfra")
            return await pinned_client._generate_provider_response(model_request)

        response = asyncio.run(_inner())
        assert response.actual_upstream_provider == "deepinfra"

    def test_pinned_mismatched_actual_provider_fails_loudly(
        self, pinned_client, model_request, pinned_mock_create
    ):
        """allow_fallbacks=False + pinned upstream_provider="deepinfra": if
        OpenRouter's response reports a DIFFERENT actual provider, this must
        raise rather than silently return a success response under the
        wrong deployment's identity."""
        async def _inner():
            pinned_mock_create.return_value = _make_response(provider="together")
            await pinned_client._generate_provider_response(model_request)

        with pytest.raises(ProviderResponseError):
            asyncio.run(_inner())

    def test_mismatched_actual_provider_allowed_when_fallbacks_enabled(
        self, model_request, mock_create
    ):
        """upstream_provider pinned but allow_fallbacks=True: a mismatch is
        the expected/requested behavior (OpenRouter is explicitly allowed to
        fall back), so it must NOT raise -- only the fallbacks=False +
        mismatch combination is a provenance violation."""
        client = OpenRouterClient(
            model_name="meta-llama/llama-3.1-8b-instruct",
            api_key="test-key",
            upstream_provider="deepinfra",
            allow_fallbacks=True,
        )
        client.client.chat.completions.create = mock_create

        async def _inner():
            mock_create.return_value = _make_response(provider="together")
            return await client._generate_provider_response(model_request)

        response = asyncio.run(_inner())
        assert response.actual_upstream_provider == "together"
