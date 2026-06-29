# tests/runners/test_async_runner.py
#
# Tests for the two-level async concurrency architecture:
#   Level 1 — within-model: all questions in a batch fire concurrently
#   Level 2 — across-models: API models run concurrently in run_models_concurrently()
#
# Uses asyncio.gather and lightweight mocks rather than real API calls.

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
from pathlib import Path
from typing import Any
import types

import pandas as pd
import pytest

from choicebench.backends.api_backend import APIBackend
from choicebench.backends.dummy_backend import DummyBackend
from choicebench.backends.hf_backend import HuggingFaceBackend
from choicebench.clients.types import (
    ModelRequest,
    ModelResponse,
    SUCCESS_STATUS,
    FAILURE_STATUS,
    ErrorInfo,
)
from choicebench.methods.library.direct_mcq import DirectMCQRunner

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "prompts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_run_experiment():
    spec = importlib.util.spec_from_file_location(
        "run_experiment_async_test", _REPO_ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_success_response(provider="openai", model_name="gpt-4.1-mini", raw_text="C") -> ModelResponse:
    return ModelResponse(
        provider=provider,
        model_name=model_name,
        status=SUCCESS_STATUS,
        latency_seconds=0.01,
        raw_text=raw_text,
        finish_reason="stop",
        usage=None,
        error=None,
        timestamp_utc=None,
    )


def _make_failure_response(provider="openai", model_name="gpt-4.1-mini") -> ModelResponse:
    return ModelResponse(
        provider=provider,
        model_name=model_name,
        status=FAILURE_STATUS,
        latency_seconds=0.01,
        raw_text=None,
        finish_reason=None,
        usage=None,
        error=ErrorInfo("TimeoutError", "timed out", True, "provider_call"),
        timestamp_utc=None,
    )


class _AsyncMockBackend(APIBackend):
    """APIBackend subclass that replaces network calls with a canned list of responses.

    generate_batch() returns the queued responses in order (one per prompt)
    and records all prompt strings it received.
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        # Bypass APIBackend.__init__ entirely — we don't need a real client.
        self._provider = "openai"
        self._model_name = "mock-model"
        self._temperature = 0.0
        self._max_tokens = 16
        self._seed = 42
        self._concurrency_limit = len(responses) or 10
        self._semaphore = None
        self._queued = list(responses)
        self.prompts_received: list[str] = []
        self.call_count = 0

    async def generate_batch(self, prompts: list[str]) -> list[ModelResponse]:
        self.prompts_received.extend(prompts)
        self.call_count += 1
        if len(prompts) > len(self._queued):
            raise RuntimeError(
                f"_AsyncMockBackend: requested {len(prompts)} responses but "
                f"only {len(self._queued)} queued."
            )
        out = self._queued[: len(prompts)]
        self._queued = self._queued[len(prompts):]
        return out


def _make_direct_mcq_runner(backend, method_name="direct_mcq"):
    return DirectMCQRunner(
        backend=backend,
        method_name=method_name,
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="async_test_run",
    )


def _question_df(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "question_id": f"q{i:04d}",
            "subject": "computer_security",
            "question_text": f"Question {i}: Which protocol is used for secure web browsing?",
            "choice_a": "FTP",
            "choice_b": "HTTP",
            "choice_c": "HTTPS",
            "choice_d": "SMTP",
            "correct_option": "C",
            "correct_answer_text": "HTTPS",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Level 1 — within-model concurrency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_many_async_calls_generate_batch_once():
    """run_many_async() must call generate_batch() exactly once with all prompts."""
    n = 5
    responses = [_make_success_response(raw_text="C") for _ in range(n)]
    backend = _AsyncMockBackend(responses)
    runner = _make_direct_mcq_runner(backend)

    results = await runner.run_many_async(_question_df(n))

    assert backend.call_count == 1
    assert len(backend.prompts_received) == n
    assert len(results) == n


@pytest.mark.asyncio
async def test_run_many_async_returns_correct_result_count():
    """result list length must equal question count."""
    n = 4
    responses = [_make_success_response(raw_text="C") for _ in range(n)]
    backend = _AsyncMockBackend(responses)
    runner = _make_direct_mcq_runner(backend)

    results = await runner.run_many_async(_question_df(n))

    assert len(results) == n


@pytest.mark.asyncio
async def test_run_many_async_correct_answers_score_true():
    """All questions answered 'C' (the correct option) must be scored correct."""
    n = 3
    responses = [_make_success_response(raw_text="C") for _ in range(n)]
    backend = _AsyncMockBackend(responses)
    runner = _make_direct_mcq_runner(backend)

    results = await runner.run_many_async(_question_df(n))

    for r in results:
        assert r["is_correct"] is True
        assert r["parsed_choice"] == "C"


@pytest.mark.asyncio
async def test_run_many_async_failure_response_produces_error_row():
    """A failed response must produce a result row with error_type set and is_correct None."""
    responses = [_make_failure_response()]
    backend = _AsyncMockBackend(responses)
    runner = _make_direct_mcq_runner(backend)

    results = await runner.run_many_async(_question_df(1))

    assert len(results) == 1
    r = results[0]
    assert r["is_correct"] is None
    assert r["parsed_choice"] is None
    assert r["error_type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_run_many_async_mixed_success_and_failure():
    """Mixed success/failure batch: each row gets the right outcome independently."""
    responses = [
        _make_success_response(raw_text="C"),
        _make_failure_response(),
        _make_success_response(raw_text="A"),  # wrong answer
    ]
    backend = _AsyncMockBackend(responses)
    runner = _make_direct_mcq_runner(backend)

    results = await runner.run_many_async(_question_df(3))

    assert results[0]["is_correct"] is True
    assert results[1]["is_correct"] is None
    assert results[1]["error_type"] == "TimeoutError"
    assert results[2]["is_correct"] is False


# ---------------------------------------------------------------------------
# Semaphore: concurrency_limit enforcement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semaphore_limits_concurrent_requests():
    """concurrency_limit=1 must serialize requests (only 1 in-flight at a time)."""
    in_flight: list[int] = []
    max_concurrent = 0

    async def fake_generate(request: ModelRequest) -> ModelResponse:
        nonlocal max_concurrent
        in_flight.append(1)
        max_concurrent = max(max_concurrent, len(in_flight))
        await asyncio.sleep(0)  # yield to allow other coroutines to try
        in_flight.pop()
        return ModelResponse(
            provider=request.provider,
            model_name=request.model_name,
            status=SUCCESS_STATUS,
            latency_seconds=0.0,
            raw_text="C",
            finish_reason="stop",
            usage=None,
            error=None,
            timestamp_utc=None,
        )

    # Build a minimal APIBackend with a mock client that exposes fake_generate.
    class _MinimalClient:
        provider = "openai"
        model_name = "mock"
        async def generate(self, request):
            return await fake_generate(request)
        async def generate_batch(self, requests):
            return list(await asyncio.gather(*[self.generate(r) for r in requests]))

    from choicebench.infra.cache import CachingClientWrapper, ResponseCache
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as td:
        cache = ResponseCache(pathlib.Path(td))

        class _APIBackendDirect(APIBackend):
            def __init__(self, limit):
                self._provider = "openai"
                self._model_name = "mock"
                self._temperature = 0.0
                self._max_tokens = 16
                self._seed = 42
                self._concurrency_limit = limit
                self._semaphore = None
                inner = _MinimalClient()
                self._client = CachingClientWrapper(inner, cache)

        backend = _APIBackendDirect(limit=1)
        # Fire 4 requests with concurrency_limit=1
        responses = await backend.generate_batch(["p1", "p2", "p3", "p4"])

    assert len(responses) == 4
    assert all(r.is_success() for r in responses)
    assert max_concurrent == 1, f"Expected max 1 concurrent request, got {max_concurrent}"


@pytest.mark.asyncio
async def test_semaphore_allows_multiple_concurrent_with_higher_limit():
    """concurrency_limit=4 allows multiple requests in-flight simultaneously."""
    in_flight: list[int] = []
    max_concurrent = 0

    async def fake_generate(request: ModelRequest) -> ModelResponse:
        nonlocal max_concurrent
        in_flight.append(1)
        max_concurrent = max(max_concurrent, len(in_flight))
        await asyncio.sleep(0.01)  # small delay to allow overlap
        in_flight.pop()
        return ModelResponse(
            provider=request.provider,
            model_name=request.model_name,
            status=SUCCESS_STATUS,
            latency_seconds=0.0,
            raw_text="C",
            finish_reason="stop",
            usage=None,
            error=None,
            timestamp_utc=None,
        )

    class _MinimalClient:
        provider = "openai"
        model_name = "mock"
        async def generate(self, request):
            return await fake_generate(request)
        async def generate_batch(self, requests):
            return list(await asyncio.gather(*[self.generate(r) for r in requests]))

    from choicebench.infra.cache import CachingClientWrapper, ResponseCache
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as td:
        cache = ResponseCache(pathlib.Path(td))

        class _APIBackendDirect(APIBackend):
            def __init__(self, limit):
                self._provider = "openai"
                self._model_name = "mock"
                self._temperature = 0.0
                self._max_tokens = 16
                self._seed = 42
                self._concurrency_limit = limit
                self._semaphore = None
                inner = _MinimalClient()
                self._client = CachingClientWrapper(inner, cache)

        backend = _APIBackendDirect(limit=4)
        responses = await backend.generate_batch(["p1", "p2", "p3", "p4"])

    assert len(responses) == 4
    # With limit=4 and sleep(0.01), all 4 should be in-flight simultaneously
    assert max_concurrent > 1, f"Expected >1 concurrent requests, got {max_concurrent}"


# ---------------------------------------------------------------------------
# Level 2 — model-level concurrency (run_models_concurrently)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_models_concurrently_api_models_run_in_parallel(tmp_path):
    """Two API models must be launched concurrently, not sequentially."""
    run_exp = _load_run_experiment()

    call_order: list[str] = []
    started: dict[str, asyncio.Event] = {}

    async def fake_run_model(method, model_config, *args, **kwargs):
        name = model_config.model_name_or_path
        started[name] = asyncio.Event()
        call_order.append(f"start:{name}")
        started[name].set()
        await asyncio.sleep(0.02)
        call_order.append(f"end:{name}")

    import choicebench.config.schema as schema_mod
    model_a = schema_mod.ModelConfig(backend="api", model_name_or_path="model_a", provider="openai")
    model_b = schema_mod.ModelConfig(backend="api", model_name_or_path="model_b", provider="openai")

    from choicebench.config.schema import BenchmarkConfig, ExperimentConfig, MethodConfig, RunConfig
    method = MethodConfig(name="direct_mcq")
    bench = BenchmarkConfig(name="toy")
    config = ExperimentConfig(
        name="test",
        models=[model_a, model_b],
        benchmarks=[bench],
        methods=[method],
        metrics=["accuracy"],
        run=RunConfig(),
    )

    import unittest.mock as mock
    with mock.patch.object(run_exp, "_run_model", side_effect=fake_run_model):
        await run_exp.run_models_concurrently(
            method=method,
            benchmark_cfg=bench,
            model_configs=[model_a, model_b],
            preflight_questions=None,
            questions=pd.DataFrame(),
            config=config,
            run_id="rid",
            output_dir=tmp_path,
            checkpoint_dir=tmp_path / "checkpoints",
        )

    # Both models should have started before either finished
    assert call_order.index("start:model_a") < call_order.index("end:model_b")
    assert call_order.index("start:model_b") < call_order.index("end:model_a")


@pytest.mark.asyncio
async def test_run_models_concurrently_sync_models_run_sequentially(tmp_path):
    """Non-API (HF/Dummy) models must run sequentially, not concurrently."""
    run_exp = _load_run_experiment()

    call_order: list[str] = []

    async def fake_run_model(method, model_config, *args, **kwargs):
        name = model_config.model_name_or_path
        call_order.append(f"start:{name}")
        await asyncio.sleep(0.02)
        call_order.append(f"end:{name}")

    import choicebench.config.schema as schema_mod
    model_a = schema_mod.ModelConfig(backend="dummy", model_name_or_path="dummy_a")
    model_b = schema_mod.ModelConfig(backend="dummy", model_name_or_path="dummy_b")

    from choicebench.config.schema import BenchmarkConfig, ExperimentConfig, MethodConfig, RunConfig
    method = MethodConfig(name="direct_mcq")
    bench = BenchmarkConfig(name="toy")
    config = ExperimentConfig(
        name="test",
        models=[model_a, model_b],
        benchmarks=[bench],
        methods=[method],
        metrics=["accuracy"],
        run=RunConfig(),
    )

    import unittest.mock as mock
    with mock.patch.object(run_exp, "_run_model", side_effect=fake_run_model):
        await run_exp.run_models_concurrently(
            method=method,
            benchmark_cfg=bench,
            model_configs=[model_a, model_b],
            preflight_questions=None,
            questions=pd.DataFrame(),
            config=config,
            run_id="rid",
            output_dir=tmp_path,
            checkpoint_dir=tmp_path / "checkpoints",
        )

    # Sequential: model_a must finish before model_b starts
    assert call_order == ["start:dummy_a", "end:dummy_a", "start:dummy_b", "end:dummy_b"]


# ---------------------------------------------------------------------------
# HuggingFace / Dummy backend: verify sync path not broken
# ---------------------------------------------------------------------------

def test_dummy_backend_is_not_async_capable():
    """DummyBackend.is_async_capable() must return False."""
    assert DummyBackend().is_async_capable() is False


def test_api_backend_is_async_capable(tmp_path):
    """APIBackend.is_async_capable() must return True."""
    from choicebench.infra.cache import ResponseCache

    class _FakeClient:
        provider = "openai"
        model_name = "mock"

    from choicebench.infra.cache import CachingClientWrapper
    cache = ResponseCache(tmp_path)
    client = _FakeClient()
    backend = APIBackend.__new__(APIBackend)
    backend._provider = "openai"
    backend._model_name = "mock"
    backend._temperature = 0.0
    backend._max_tokens = 16
    backend._seed = 42
    backend._concurrency_limit = 5
    backend._semaphore = None
    backend._client = CachingClientWrapper(client, cache)

    assert backend.is_async_capable() is True


def test_api_backend_generate_raises_runtime_error(tmp_path):
    """APIBackend.generate() must raise RuntimeError (not asyncio.run())."""
    from choicebench.infra.cache import CachingClientWrapper, ResponseCache

    class _FakeClient:
        provider = "openai"
        model_name = "mock"

    cache = ResponseCache(tmp_path)
    backend = APIBackend.__new__(APIBackend)
    backend._provider = "openai"
    backend._model_name = "mock"
    backend._temperature = 0.0
    backend._max_tokens = 16
    backend._seed = 42
    backend._concurrency_limit = 5
    backend._semaphore = None
    backend._client = CachingClientWrapper(_FakeClient(), cache)

    with pytest.raises(RuntimeError, match="generate_batch"):
        backend.generate("test prompt")


# ---------------------------------------------------------------------------
# Checkpoint safety: concurrent model runs write to separate paths
# ---------------------------------------------------------------------------

def test_checkpoint_paths_are_unique_per_model(tmp_path):
    """Two CheckpointManagers for different models must not share a path."""
    from choicebench.infra.checkpoint import CheckpointManager

    mgr_a = CheckpointManager(tmp_path, "rid", "direct_mcq", "openai/gpt-4.1", "toy")
    mgr_b = CheckpointManager(tmp_path, "rid", "direct_mcq", "google/gemini", "toy")

    assert mgr_a._path != mgr_b._path


def test_output_csv_paths_are_unique_per_model(tmp_path):
    """write_run_results for two different models must produce different file paths."""
    from choicebench.io.writers import write_run_results

    path_a = write_run_results(
        results=[{"question_id": "q1", "is_correct": True}],
        output_dir=tmp_path,
        run_id="rid",
        method_name="direct_mcq",
        model_name="openai/gpt-4.1",
        benchmark="toy",
    )
    path_b = write_run_results(
        results=[{"question_id": "q1", "is_correct": True}],
        output_dir=tmp_path,
        run_id="rid",
        method_name="direct_mcq",
        model_name="google/gemini",
        benchmark="toy",
    )

    assert path_a != path_b
    assert path_a.exists()
    assert path_b.exists()
