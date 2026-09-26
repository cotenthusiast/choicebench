# tests/infra/test_cache.py

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from choicebench.infra.cache import _cache_key, ResponseCache, CachingClientWrapper
from choicebench.clients.types import (
    ModelRequest,
    ModelResponse,
    UsageInfo,
    SUCCESS_STATUS,
    FAILURE_STATUS,
    ErrorInfo,
)


@pytest.fixture
def base_request() -> ModelRequest:
    return ModelRequest(
        provider="openai",
        model_name="gpt-4.1-mini",
        payload="What is the capital of France?",
        temperature=0.0,
        max_tokens=128,
    )


@pytest.fixture
def success_response() -> ModelResponse:
    return ModelResponse(
        provider="openai",
        model_name="gpt-4.1-mini",
        status=SUCCESS_STATUS,
        latency_seconds=0.25,
        raw_text="Paris",
        finish_reason="stop",
        usage=UsageInfo(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        error=None,
        timestamp_utc=None,
    )


@pytest.fixture
def failure_response() -> ModelResponse:
    return ModelResponse(
        provider="openai",
        model_name="gpt-4.1-mini",
        status=FAILURE_STATUS,
        latency_seconds=0.1,
        raw_text=None,
        finish_reason=None,
        usage=None,
        error=ErrorInfo("ProviderTimeoutError", "timed out", True, "provider_call"),
        timestamp_utc=None,
    )


# ---------------------------------------------------------------------------
# _cache_key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_same_request_produces_same_key(self, base_request):
        assert _cache_key(base_request) == _cache_key(base_request)

    def test_different_payload_produces_different_key(self, base_request):
        other = ModelRequest(
            provider="openai",
            model_name="gpt-4.1-mini",
            payload="A completely different question?",
            temperature=0.0,
            max_tokens=128,
        )
        assert _cache_key(base_request) != _cache_key(other)

    def test_different_model_produces_different_key(self, base_request):
        other = ModelRequest(
            provider="groq",
            model_name="llama-3.1-8b-instant",
            payload=base_request.payload,
            temperature=0.0,
            max_tokens=128,
        )
        assert _cache_key(base_request) != _cache_key(other)

    def test_different_temperature_produces_different_key(self, base_request):
        other = ModelRequest(
            provider="openai",
            model_name="gpt-4.1-mini",
            payload=base_request.payload,
            temperature=0.7,
            max_tokens=128,
        )
        assert _cache_key(base_request) != _cache_key(other)

    def test_different_max_tokens_produces_different_key(self, base_request):
        other = ModelRequest(
            provider="openai",
            model_name="gpt-4.1-mini",
            payload=base_request.payload,
            temperature=0.0,
            max_tokens=512,
        )
        assert _cache_key(base_request) != _cache_key(other)

    def test_key_is_64_char_hex_string(self, base_request):
        key = _cache_key(base_request)
        assert isinstance(key, str)
        assert len(key) == 64
        int(key, 16)  # raises ValueError if not valid hex

    def test_omitted_extra_identity_matches_the_pre_existing_key_formula(self, base_request):
        """Backward compatibility: a request with no extra_identity (every
        non-OpenRouter client, and an OpenRouter client with no upstream
        pinning) must produce the EXACT SAME key as before extra_identity
        existed -- historical cache entries for these must stay reachable."""
        assert _cache_key(base_request, extra_identity=None) == _cache_key(base_request)
        assert _cache_key(base_request, extra_identity={}) == _cache_key(base_request)

    def test_different_extra_identity_produces_different_key(self, base_request):
        """Two OpenRouter configs for the identical nominal provider/model
        but different upstream_provider pinning must never collide -- they
        can resolve to genuinely different deployments (e.g. different
        quantization)."""
        key_a = _cache_key(base_request, extra_identity={"upstream_provider": "deepinfra", "allow_fallbacks": False})
        key_b = _cache_key(base_request, extra_identity={"upstream_provider": "together", "allow_fallbacks": False})
        assert key_a != key_b
        assert key_a != _cache_key(base_request)  # also differs from "no pinning at all"


# ---------------------------------------------------------------------------
# ResponseCache
# ---------------------------------------------------------------------------


class TestResponseCache:
    def test_foreign_namespace_entry_is_not_reused(self, tmp_path):
        path = tmp_path / "shared"
        first = ResponseCache(path, namespace="model-one")
        first.put("ab-key", {"raw_text": "A"})
        assert ResponseCache(path, namespace="model-two").get("ab-key") is None

    def test_get_returns_none_on_miss(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        assert cache.get("nonexistent_key_" + "a" * 48) is None

    def test_put_then_get_round_trips_payload(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        key = "ab" + "c" * 62
        payload = {"raw_text": "hello", "finish_reason": "stop", "usage": None}
        cache.put(key, payload)
        result = cache.get(key)
        assert result is not None
        assert result["raw_text"] == "hello"
        assert result["finish_reason"] == "stop"

    def test_put_stores_file_in_two_level_directory(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        key = "ab" + "c" * 62
        cache.put(key, {"raw_text": "test", "finish_reason": None, "usage": None})
        expected_path = tmp_path / "cache" / "ab" / f"{key}.json"
        assert expected_path.exists()

    def test_get_returns_none_for_corrupted_json(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        key = "xy" + "z" * 62
        path = tmp_path / "cache" / "xy" / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this is not valid json }{")
        assert cache.get(key) is None

    def test_second_put_overwrites_first(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        key = "aa" + "b" * 62
        cache.put(key, {"raw_text": "first", "finish_reason": None, "usage": None})
        cache.put(key, {"raw_text": "second", "finish_reason": None, "usage": None})
        assert cache.get(key)["raw_text"] == "second"

    def test_cache_directory_created_on_init(self, tmp_path):
        cache_dir = tmp_path / "new_cache_dir"
        assert not cache_dir.exists()
        ResponseCache(cache_dir)
        assert cache_dir.exists()

    def test_put_stores_usage_dict(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        key = "cd" + "e" * 62
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        cache.put(key, {"raw_text": "hi", "finish_reason": "stop", "usage": usage})
        result = cache.get(key)
        assert result["usage"] == usage

    def test_no_tmp_file_left_after_put(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        key = "ff" + "0" * 62
        cache.put(key, {"raw_text": "x", "finish_reason": None, "usage": None})
        tmp_file = tmp_path / "cache" / "ff" / f"{key}.tmp"
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# CachingClientWrapper
# ---------------------------------------------------------------------------


class TestCachingClientWrapper:
    def test_logprobs_round_trip_through_cache(self, tmp_path):
        lp = [{"token": "A", "logprob": -0.12, "top_logprobs": []}]

        async def _inner():
            req = ModelRequest(
                provider="together",
                model_name="Qwen/Qwen2.5-7B-Instruct",
                payload="Pick A B C or D.",
                temperature=0.0,
                max_tokens=8,
            )
            resp_ok = ModelResponse(
                provider=req.provider,
                model_name=req.model_name,
                status=SUCCESS_STATUS,
                latency_seconds=0.1,
                raw_text="A",
                finish_reason="stop",
                usage=UsageInfo(1, 1, 2),
                error=None,
                timestamp_utc=None,
                logprobs=lp,
            )
            mock_client = AsyncMock()
            mock_client.provider = req.provider
            mock_client.model_name = req.model_name
            mock_client.generate = AsyncMock(return_value=resp_ok)

            wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
            r1 = await wrapper.generate(req)
            r2 = await wrapper.generate(req)
            assert r1.raw_text == r2.raw_text == "A"
            assert r1.logprobs == r2.logprobs == lp
            assert mock_client.generate.await_count == 1

        asyncio.run(_inner())


    def test_cache_miss_calls_underlying_client(
        self, tmp_path, base_request, success_response
    ):
        async def _inner():
            mock_client = AsyncMock()
            mock_client.provider = "openai"
            mock_client.model_name = "gpt-4.1-mini"
            mock_client.generate = AsyncMock(return_value=success_response)

            wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
            response = await wrapper.generate(base_request)

            mock_client.generate.assert_awaited_once_with(base_request)
            assert response.raw_text == "Paris"

        asyncio.run(_inner())

    def test_second_call_is_served_from_cache(
        self, tmp_path, base_request, success_response
    ):
        async def _inner():
            mock_client = AsyncMock()
            mock_client.provider = "openai"
            mock_client.model_name = "gpt-4.1-mini"
            mock_client.generate = AsyncMock(return_value=success_response)

            wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
            await wrapper.generate(base_request)
            await wrapper.generate(base_request)

            assert mock_client.generate.await_count == 1

        asyncio.run(_inner())

    def test_cached_response_has_same_raw_text(
        self, tmp_path, base_request, success_response
    ):
        async def _inner():
            mock_client = AsyncMock()
            mock_client.provider = "openai"
            mock_client.model_name = "gpt-4.1-mini"
            mock_client.generate = AsyncMock(return_value=success_response)

            wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
            r1 = await wrapper.generate(base_request)
            r2 = await wrapper.generate(base_request)

            assert r1.raw_text == r2.raw_text == "Paris"

        asyncio.run(_inner())

    def test_failed_response_is_not_cached(
        self, tmp_path, base_request, failure_response
    ):
        async def _inner():
            mock_client = AsyncMock()
            mock_client.provider = "openai"
            mock_client.model_name = "gpt-4.1-mini"
            mock_client.generate = AsyncMock(return_value=failure_response)

            wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
            await wrapper.generate(base_request)
            await wrapper.generate(base_request)

            assert mock_client.generate.await_count == 2

        asyncio.run(_inner())

    def test_cached_response_status_is_success(
        self, tmp_path, base_request, success_response
    ):
        async def _inner():
            mock_client = AsyncMock()
            mock_client.provider = "openai"
            mock_client.model_name = "gpt-4.1-mini"
            mock_client.generate = AsyncMock(return_value=success_response)

            wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
            await wrapper.generate(base_request)
            cached = await wrapper.generate(base_request)

            assert cached.status == SUCCESS_STATUS

        asyncio.run(_inner())

    def test_usage_round_trips_through_cache(
        self, tmp_path, base_request, success_response
    ):
        async def _inner():
            mock_client = AsyncMock()
            mock_client.provider = "openai"
            mock_client.model_name = "gpt-4.1-mini"
            mock_client.generate = AsyncMock(return_value=success_response)

            wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
            await wrapper.generate(base_request)
            cached = await wrapper.generate(base_request)

            assert cached.usage is not None
            assert cached.usage.prompt_tokens == 10
            assert cached.usage.completion_tokens == 2
            assert cached.usage.total_tokens == 12

        asyncio.run(_inner())

    def test_exposes_provider_from_underlying_client(self, tmp_path):
        mock_client = MagicMock()
        mock_client.provider = "groq"
        mock_client.model_name = "llama-3.1-8b-instant"

        wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
        assert wrapper.provider == "groq"

    def test_exposes_model_name_from_underlying_client(self, tmp_path):
        mock_client = MagicMock()
        mock_client.provider = "groq"
        mock_client.model_name = "llama-3.1-8b-instant"

        wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
        assert wrapper.model_name == "llama-3.1-8b-instant"

    def test_generate_batch_calls_generate_for_each_request(
        self, tmp_path, base_request, success_response
    ):
        async def _inner():
            mock_client = AsyncMock()
            mock_client.provider = "openai"
            mock_client.model_name = "gpt-4.1-mini"
            mock_client.generate = AsyncMock(return_value=success_response)

            other_request = ModelRequest(
                provider="openai",
                model_name="gpt-4.1-mini",
                payload="A different question entirely?",
                temperature=0.0,
                max_tokens=128,
            )

            wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
            results = await wrapper.generate_batch([base_request, other_request])

            assert len(results) == 2
            assert mock_client.generate.await_count == 2

        asyncio.run(_inner())

    def test_generate_batch_respects_cache_for_duplicates(
        self, tmp_path, base_request, success_response
    ):
        async def _inner():
            mock_client = AsyncMock()
            mock_client.provider = "openai"
            mock_client.model_name = "gpt-4.1-mini"
            mock_client.generate = AsyncMock(return_value=success_response)

            wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
            results = await wrapper.generate_batch([base_request, base_request])

            assert len(results) == 2
            # First call misses cache, second hits it
            assert mock_client.generate.await_count == 1

        asyncio.run(_inner())


class TestCachingClientWrapperActualUpstreamProviderProvenance:
    """actual_upstream_provider is a response-level provenance field (the
    upstream OpenRouter actually routed to), distinct from the client-level
    upstream_provider/allow_fallbacks pinning already covered by
    TestCachingClientWrapperUpstreamProviderIdentity below."""

    def test_actual_upstream_provider_round_trips_through_cache(
        self, tmp_path, base_request
    ):
        async def _inner():
            resp = ModelResponse(
                provider="openrouter",
                model_name="qwen/qwen-2.5-7b-instruct",
                status=SUCCESS_STATUS,
                latency_seconds=0.1,
                raw_text="A",
                finish_reason="stop",
                usage=None,
                error=None,
                timestamp_utc=None,
                actual_upstream_provider="Phala",
            )
            req = ModelRequest(
                provider="openrouter", model_name="qwen/qwen-2.5-7b-instruct",
                payload="Which letter?", temperature=0.0, max_tokens=8,
            )
            mock_client = AsyncMock()
            mock_client.provider = req.provider
            mock_client.model_name = req.model_name
            mock_client.generate = AsyncMock(return_value=resp)

            wrapper = CachingClientWrapper(mock_client, ResponseCache(tmp_path / "cache"))
            await wrapper.generate(req)
            cached = await wrapper.generate(req)

            assert cached.actual_upstream_provider == "Phala"
            assert mock_client.generate.await_count == 1

        asyncio.run(_inner())

    def test_cached_payload_missing_actual_upstream_provider_reads_as_none(
        self, tmp_path, base_request
    ):
        """Backward compatibility: an old cache entry written before this
        field existed has no "actual_upstream_provider" key at all -- it
        must remain readable, deserializing to None rather than raising."""
        cache = ResponseCache(tmp_path / "cache")
        key = _cache_key(base_request)
        old_payload = {"raw_text": "Paris", "finish_reason": "stop", "usage": None}
        cache.put(key, old_payload)

        from choicebench.infra.cache import CachingClientWrapper as _CCW
        restored = _CCW._build_cached_response(base_request, cache.get(key))
        assert restored.actual_upstream_provider is None

    def test_build_cached_response_restores_actual_upstream_provider(
        self, base_request
    ):
        from choicebench.infra.cache import CachingClientWrapper as _CCW
        payload = {
            "raw_text": "Paris", "finish_reason": "stop", "usage": None,
            "actual_upstream_provider": "Phala",
        }
        restored = _CCW._build_cached_response(base_request, payload)
        assert restored.actual_upstream_provider == "Phala"


class TestCachingClientWrapperUpstreamProviderIdentity:
    """The audit's "cache identity may omit runtime-affecting properties"
    concern, specifically for OpenRouter's upstream_provider/
    allow_fallbacks pinning: two configs for the identical nominal
    provider/model but pinned to different upstream deployments (e.g.
    Llama served via deepinfra vs. via together) must never share a cache
    entry."""

    @staticmethod
    def _pinned_client(upstream_provider, allow_fallbacks, generate_return):
        client = AsyncMock()
        client.provider = "openrouter"
        client.model_name = "meta-llama/llama-3.1-8b-instruct"
        client._upstream_provider = upstream_provider
        client._allow_fallbacks = allow_fallbacks
        client.generate = AsyncMock(return_value=generate_return)
        return client

    def test_different_upstream_pinning_never_shares_a_cache_entry(self, tmp_path, success_response):
        async def _inner():
            req = ModelRequest(
                provider="openrouter", model_name="meta-llama/llama-3.1-8b-instruct",
                payload="Which protocol secures web browsing?", temperature=0.0, max_tokens=128,
            )
            cache_dir = tmp_path / "cache"

            deepinfra_client = self._pinned_client("deepinfra", False, success_response)
            deepinfra_wrapper = CachingClientWrapper(deepinfra_client, ResponseCache(cache_dir))
            await deepinfra_wrapper.generate(req)

            together_client = self._pinned_client("together", False, success_response)
            together_wrapper = CachingClientWrapper(together_client, ResponseCache(cache_dir))
            await together_wrapper.generate(req)

            # Sharing one cache_dir (same as a "shared" cache_scope, or a
            # collision in model_identity, would produce) must NOT make
            # the together-pinned wrapper see the deepinfra-pinned
            # response as a cache hit -- each must genuinely call through.
            assert deepinfra_client.generate.await_count == 1
            assert together_client.generate.await_count == 1

        asyncio.run(_inner())

    def test_unpinned_openrouter_client_cache_key_is_unaffected(self, tmp_path, base_request, success_response):
        """Backward compatibility: an OpenRouter client with no pinning at
        all (upstream_provider=None, the pre-existing default) must
        produce the exact same key as before this existed."""
        async def _inner():
            client = self._pinned_client(None, True, success_response)
            wrapper = CachingClientWrapper(client, ResponseCache(tmp_path / "cache"))
            await wrapper.generate(base_request)
            await wrapper.generate(base_request)
            # Second call is a cache hit -- only one real call made.
            assert client.generate.await_count == 1

        asyncio.run(_inner())
