# tests/runners/test_async_runner.py
#
# Tests for the two-level async concurrency architecture:
#   Level 1 — within-model: all questions in a batch fire concurrently
#   Level 2 — across-models: API models run concurrently in run_models_concurrently()
#
# Uses asyncio.gather and lightweight mocks rather than real API calls.

from __future__ import annotations

import asyncio
import importlib
import json
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
from choicebench.methods.direct_mcq import DirectMCQRunner

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "prompts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_run_experiment():
    import choicebench.cli.run_experiment as module
    return importlib.reload(module)


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


_QUESTION_CHOICES_JSON = json.dumps(
    [{"text": t, "source_index": i} for i, t in enumerate(["FTP", "HTTP", "HTTPS", "SMTP"])]
)


def _question_df(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "question_id": f"q{i:04d}",
            "subject": "computer_security",
            "question_text": f"Question {i}: Which protocol is used for secure web browsing?",
            "choices_json": _QUESTION_CHOICES_JSON,
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
    # Concurrency gating now lives on the client (mirroring BaseClient's own
    # semaphore) since APIBackend's redundant outer semaphore was removed —
    # see step 26.1.
    class _MinimalClient:
        provider = "openai"
        model_name = "mock"
        def __init__(self, limit):
            self._semaphore = asyncio.Semaphore(limit)
        async def generate(self, request):
            async with self._semaphore:
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
                inner = _MinimalClient(limit)
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
        def __init__(self, limit):
            self._semaphore = asyncio.Semaphore(limit)
        async def generate(self, request):
            async with self._semaphore:
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
                inner = _MinimalClient(limit)
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
            conditions=[{"condition_id": "cond_model_a"}, {"condition_id": "cond_model_b"}],
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
            conditions=[{"condition_id": "cond_dummy_a"}, {"condition_id": "cond_dummy_b"}],
        )

    # Sequential: model_a must finish before model_b starts
    assert call_order == ["start:dummy_a", "end:dummy_a", "start:dummy_b", "end:dummy_b"]


@pytest.mark.asyncio
async def test_run_models_concurrently_isolates_failures(tmp_path):
    """One API model raising must not prevent its sibling from completing.

    Regression test: run_models_concurrently used to call asyncio.gather()
    with the default return_exceptions=False, so one model's exception
    aborted the whole gather() and could destroy the sibling's still-running
    task before it finished.
    """
    run_exp = _load_run_experiment()

    call_order: list[str] = []

    async def fake_run_model(method, model_config, *args, **kwargs):
        name = model_config.model_name_or_path
        call_order.append(f"start:{name}")
        if name == "model_broken":
            await asyncio.sleep(0.01)
            raise RuntimeError("boom: simulated failure in external method/metric")
        await asyncio.sleep(0.03)
        call_order.append(f"end:{name}")

    import choicebench.config.schema as schema_mod
    model_ok = schema_mod.ModelConfig(backend="api", model_name_or_path="model_ok", provider="openai")
    model_broken = schema_mod.ModelConfig(backend="api", model_name_or_path="model_broken", provider="openai")

    from choicebench.config.schema import BenchmarkConfig, ExperimentConfig, MethodConfig, RunConfig
    method = MethodConfig(name="direct_mcq")
    bench = BenchmarkConfig(name="toy")
    config = ExperimentConfig(
        name="test",
        models=[model_ok, model_broken],
        benchmarks=[bench],
        methods=[method],
        metrics=["accuracy"],
        run=RunConfig(),
    )

    import unittest.mock as mock
    with mock.patch.object(run_exp, "_run_model", side_effect=fake_run_model):
        failures, gated = await run_exp.run_models_concurrently(
            method=method,
            benchmark_cfg=bench,
            model_configs=[model_ok, model_broken],
            preflight_questions=None,
            questions=pd.DataFrame(),
            config=config,
            run_id="rid",
            output_dir=tmp_path,
            checkpoint_dir=tmp_path / "checkpoints",
            conditions=[{"condition_id": "cond_ok"}, {"condition_id": "cond_broken"}],
        )

    # The healthy sibling must have completed despite the other raising.
    assert "end:model_ok" in call_order

    # Exactly one failure reported, naming the broken model.
    assert len(failures) == 1
    label, exc = failures[0]
    assert "model_broken" in label
    assert isinstance(exc, RuntimeError)
    assert gated == []


@pytest.mark.asyncio
async def test_run_models_concurrently_routes_modal_k_gate_to_gated_not_failures(tmp_path):
    """A ModalKGateError (PriDe skipped by design) must not count as a failure.

    Regression test: this used to be caught by the same broad except Exception
    as real bugs, landing in `failures` and triggering sys.exit(1) in main() —
    which also aborted the sbatch script (set -e) before evaluate_run.py ran,
    even though the exclusion was intentional (documented modal-k gate).
    """
    run_exp = _load_run_experiment()

    from choicebench.pride_gate import ModalKGateError, ModalKGateReport

    report = ModalKGateReport(
        benchmark="mmlu_pro", modal_k=10, threshold=0.95,
        n_total=10, n_evaluated=7, n_excluded=3, proportion=0.7,
        reason="skipped by design: only 7/10 question(s) have modal k=10",
    )

    async def fake_run_model(method, model_config, *args, **kwargs):
        raise ModalKGateError("gate failed", report=report)

    import unittest.mock as mock
    import choicebench.config.schema as schema_mod
    from choicebench.config.schema import BenchmarkConfig, ExperimentConfig, MethodConfig, RunConfig

    model = schema_mod.ModelConfig(backend="dummy", model_name_or_path="dummy_a")
    method = MethodConfig(name="pride")
    bench = BenchmarkConfig(name="mmlu_pro")
    config = ExperimentConfig(
        name="test", models=[model], benchmarks=[bench], methods=[method],
        metrics=["accuracy"], run=RunConfig(),
    )

    with mock.patch.object(run_exp, "_run_model", side_effect=fake_run_model):
        failures, gated = await run_exp.run_models_concurrently(
            method=method,
            benchmark_cfg=bench,
            model_configs=[model],
            preflight_questions=None,
            questions=pd.DataFrame(),
            config=config,
            run_id="rid",
            output_dir=tmp_path,
            checkpoint_dir=tmp_path / "checkpoints",
            conditions=[{"condition_id": "cond_dummy_a"}],
        )

    assert failures == []
    assert len(gated) == 1
    label, gate_report = gated[0]
    assert "dummy_a" in label
    assert gate_report is report


# ---------------------------------------------------------------------------
# Level 2.5 — backend_cache reuse across (benchmark, method) combinations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_model_reuses_cached_backend_across_calls(tmp_path):
    """A model already built for one (benchmark, method) must not be rebuilt.

    Regression test: build_backend() used to be called fresh on every
    (benchmark, method) pair, reloading full model weights from disk each
    time instead of loading once and reusing across the whole run.
    """
    run_exp = _load_run_experiment()

    build_calls: list[str] = []

    def fake_build_backend(model_config, run_id, run_seed, default_concurrency_limit=10, model_identity=None):
        build_calls.append(model_config.model_name_or_path)
        return object()

    async def fake_run_method(**kwargs):
        return None

    import unittest.mock as mock
    import choicebench.config.schema as schema_mod
    from choicebench.config.schema import BenchmarkConfig, ExperimentConfig, MethodConfig, RunConfig

    model = schema_mod.ModelConfig(backend="dummy", model_name_or_path="dummy_a")
    method_a = MethodConfig(name="direct_mcq")
    method_b = MethodConfig(name="cyclic_logprob")
    bench = BenchmarkConfig(name="toy")
    config = ExperimentConfig(
        name="test",
        models=[model],
        benchmarks=[bench],
        methods=[method_a, method_b],
        metrics=["accuracy"],
        run=RunConfig(),
    )

    condition_a = {
        "condition_id": "cond_a", "experiment_id": "exp_test", "selection_id": "sel_a",
        "artifact_id": "ds_test", "model_id": "model_test", "method_id": "method_a",
        "prompt_id": "prompt_test", "split": "test", "identity": {"preflight": None},
    }
    condition_b = {
        "condition_id": "cond_b", "experiment_id": "exp_test", "selection_id": "sel_b",
        "artifact_id": "ds_test", "model_id": "model_test", "method_id": "method_b",
        "prompt_id": "prompt_test", "split": "test", "identity": {"preflight": None},
    }

    backend_cache: dict = {}
    with mock.patch.object(run_exp, "build_backend", side_effect=fake_build_backend), \
         mock.patch.object(run_exp, "instantiate_runner", return_value=mock.MagicMock()), \
         mock.patch.object(run_exp, "run_method", side_effect=fake_run_method):
        await run_exp._run_model(
            method_a, model, bench, pd.DataFrame({"question_id": []}), None,
            config, "rid", tmp_path, tmp_path / "checkpoints", backend_cache, condition_a,
        )
        await run_exp._run_model(
            method_b, model, bench, pd.DataFrame({"question_id": []}), None,
            config, "rid", tmp_path, tmp_path / "checkpoints", backend_cache, condition_b,
        )

    # Built once for the first (benchmark, method) call; reused on the second.
    assert build_calls == ["dummy_a"]


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
    backend._client = CachingClientWrapper(_FakeClient(), cache)

    with pytest.raises(RuntimeError, match="generate_batch"):
        backend.generate("test prompt")


# ---------------------------------------------------------------------------
# Checkpoint safety: concurrent model runs write to separate paths
# ---------------------------------------------------------------------------

def test_checkpoint_paths_are_unique_per_model(tmp_path):
    """Two CheckpointManagers for different conditions must not share a path."""
    from choicebench.infra.checkpoint import CheckpointManager

    mgr_a = CheckpointManager(tmp_path, condition_id="cond_a")
    mgr_b = CheckpointManager(tmp_path, condition_id="cond_b")

    assert mgr_a._path != mgr_b._path


def test_output_csv_paths_are_unique_per_model(tmp_path):
    """write_run_results for two different conditions must produce different file paths."""
    from choicebench.io.writers import write_run_results

    identity_a = {
        "experiment_id": "exp", "condition_id": "cond_a", "dataset_artifact_id": "ds",
        "dataset_selection_id": "sel", "model_id": "openai/gpt-4.1", "method_id": "method",
        "prompt_id": "prompt", "benchmark_split": "test",
    }
    identity_b = {**identity_a, "condition_id": "cond_b", "model_id": "google/gemini"}

    path_a = write_run_results(
        results=[{"question_id": "q1", "is_correct": True, **identity_a}],
        output_dir=tmp_path,
        condition_id="cond_a",
        benchmark="toy",
    )
    path_b = write_run_results(
        results=[{"question_id": "q1", "is_correct": True, **identity_b}],
        output_dir=tmp_path,
        condition_id="cond_b",
        benchmark="toy",
    )

    assert path_a != path_b
    assert path_a.exists()
    assert path_b.exists()
