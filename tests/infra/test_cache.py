# tests/infra/test_cache.py

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from mcq_eval.infra.cache import _cache_key, ResponseCache, CachingClientWrapper
from mcq_eval.clients.types import (
    ModelRequest,
    ModelResponse,
    RequestMetadata,
    UsageInfo,
    SUCCESS_STATUS,
    FAILURE_STATUS,
    ErrorInfo,
)


@pytest.fixture
def cache_metadata() -> RequestMetadata:
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
def base_request(cache_metadata: RequestMetadata) -> ModelRequest:
    return ModelRequest(
        provider="openai",
        model_name="gpt-4.1-mini",
        payload="What is the capital of France?",
        temperature=0.0,
        max_tokens=128,
        metadata=cache_metadata,
    )


@pytest.fixture
def success_response(cache_metadata: RequestMetadata) -> ModelResponse:
    return ModelResponse(
        provider="openai",
        model_name="gpt-4.1-mini",
        status=SUCCESS_STATUS,
        latency_seconds=0.25,
        metadata=cache_metadata,
        raw_text="Paris",
        finish_reason="stop",
        usage=UsageInfo(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        error=None,
        timestamp_utc=None,
    )


@pytest.fixture
def failure_response(cache_metadata: RequestMetadata) -> ModelResponse:
    return ModelResponse(
        provider="openai",
        model_name="gpt-4.1-mini",
        status=FAILURE_STATUS,
        latency_seconds=0.1,
        metadata=cache_metadata,
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

    def test_different_payload_produces_different_key(self, base_request, cache_metadata):
        other = ModelRequest(
            provider="openai",
            model_name="gpt-4.1-mini",
            payload="A completely different question?",
            temperature=0.0,
            max_tokens=128,
            metadata=cache_metadata,
        )
        assert _cache_key(base_request) != _cache_key(other)

    def test_different_model_produces_different_key(self, base_request, cache_metadata):
        other = ModelRequest(
            provider="groq",
            model_name="llama-3.1-8b-instant",
            payload=base_request.payload,
            temperature=0.0,
            max_tokens=128,
            metadata=cache_metadata,
        )
        assert _cache_key(base_request) != _cache_key(other)

    def test_different_temperature_produces_different_key(self, base_request, cache_metadata):
        other = ModelRequest(
            provider="openai",
            model_name="gpt-4.1-mini",
            payload=base_request.payload,
            temperature=0.7,
            max_tokens=128,
            metadata=cache_metadata,
        )
        assert _cache_key(base_request) != _cache_key(other)

    def test_different_max_tokens_produces_different_key(self, base_request, cache_metadata):
        other = ModelRequest(
            provider="openai",
            model_name="gpt-4.1-mini",
            payload=base_request.payload,
            temperature=0.0,
            max_tokens=512,
            metadata=cache_metadata,
        )
        assert _cache_key(base_request) != _cache_key(other)

    def test_metadata_does_not_affect_key(self, base_request):
        """Trace metadata fields (question_id, run_id, etc.) must not change the cache key."""
        other_meta = RequestMetadata(
            question_id="completely_different_qid",
            split_name="review",
            method_name="two_prompt",
            subject="philosophy",
            run_id="run_999",
            prompt_version="v2",
            perturbation_name=None,
            sample_index=5,
        )
        other = ModelRequest(
            provider="openai",
            model_name="gpt-4.1-mini",
            payload=base_request.payload,
            temperature=0.0,
            max_tokens=128,
            metadata=other_meta,
        )
        assert _cache_key(base_request) == _cache_key(other)

    def test_key_is_64_char_hex_string(self, base_request):
        key = _cache_key(base_request)
        assert isinstance(key, str)
        assert len(key) == 64
        int(key, 16)  # raises ValueError if not valid hex

    def test_request_logprobs_toggle_changes_key(self, base_request):
        """Logprob requests must not share cache entries with non-logprob."""
        with_lp = ModelRequest(
            provider=base_request.provider,
            model_name=base_request.model_name,
            payload=base_request.payload,
            temperature=base_request.temperature,
            max_tokens=base_request.max_tokens,
            metadata=base_request.metadata,
            seed=base_request.seed,
            request_logprobs=True,
        )
        assert _cache_key(base_request) != _cache_key(with_lp)


# ---------------------------------------------------------------------------
# ResponseCache
# ---------------------------------------------------------------------------


class TestResponseCache:
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
            md = RequestMetadata(
                question_id="q_lp",
                split_name="robustness",
                method_name="pride",
                subject="computer_security",
                run_id="run_lp",
                prompt_version="v1",
                perturbation_name=None,
                sample_index=0,
            )
            req = ModelRequest(
                provider="together",
                model_name="Qwen/Qwen2.5-7B-Instruct",
                payload="Pick A B C or D.",
                temperature=0.0,
                max_tokens=8,
                metadata=md,
                request_logprobs=True,
            )
            resp_ok = ModelResponse(
                provider=req.provider,
                model_name=req.model_name,
                status=SUCCESS_STATUS,
                latency_seconds=0.1,
                metadata=md,
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
        self, tmp_path, base_request, cache_metadata, success_response
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
                metadata=cache_metadata,
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
