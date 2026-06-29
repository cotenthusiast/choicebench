# tests/runners/test_pride_async.py
#
# Tests for PriDeRunner.run_many_async(), _score_question_async(), and
# _cyclic_rollout_prob_matrix_async().
#
# Uses a lightweight _MockVLLMBackend whose _raw_client has an async
# score_options_async() method, so no real vLLM server is needed.

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from choicebench.backends.base import BaseBackend
from choicebench.methods.base import ExperimentRunner
from choicebench.methods.library.pride import PriDeRunner
from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockVLLMRawClient:
    """Simulates VLLMClient.score_options_async() with a queue of fixed responses.

    Each queued entry is either a dict[str, float] (returned) or an Exception
    instance (raised). Once the queue is exhausted, all subsequent calls raise.
    """

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.calls_received: list[tuple[str, list[str]]] = []

    async def score_options_async(
        self, prompt: str, options: list[str]
    ) -> dict[str, float]:
        self.calls_received.append((prompt, options))
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if isinstance(resp, Exception):
                raise resp
            return resp
        raise RuntimeError("_MockVLLMRawClient: no more queued responses")


class _MockVLLMBackend(BaseBackend):
    """Mock backend shaped like APIBackend(VLLMClient).

    Has _raw_client with an async score_options_async() method and exposes
    supports_score_options() / supports_logprobs for PriDeRunner construction.
    """

    def __init__(
        self,
        score_async_responses: list | None = None,
        supports_score_options_val: bool = True,
    ) -> None:
        self._raw_client = _MockVLLMRawClient(score_async_responses or [])
        self._supports_score_options_val = supports_score_options_val

    @property
    def model_name(self) -> str:
        return "mock-llama"

    @property
    def provider(self) -> str:
        return "vllm"

    @property
    def supports_logprobs(self) -> bool:
        return True

    def supports_score_options(self) -> bool:
        return self._supports_score_options_val

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("_MockVLLMBackend does not support generate()")

    def score_options(self, prompt: str, options: list[str], **kwargs) -> list[float]:
        raise NotImplementedError("_MockVLLMBackend: use score_options_async()")


def _make_vllm_runner(
    backend: _MockVLLMBackend,
    tmp_path: Path,
    *,
    calibration_n: int = 0,
    calibration_questions: list | None = None,
    run_id: str = "async_test_run",
) -> PriDeRunner:
    return PriDeRunner(
        backend=backend,
        method_name="pride",
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id=run_id,
        calibration_n=calibration_n,
        calibration_seed=0,
        calibration_benchmark="mmlu",
        calibration_runs_dir=tmp_path,
        calibration_questions=calibration_questions or [],
    )


_UNIFORM_RESP: dict[str, float] = {"A": -0.1, "B": -2.0, "C": -3.0, "D": -4.0}


# ---------------------------------------------------------------------------
# Tests: run_many_async() routing
# ---------------------------------------------------------------------------

class TestPriDeRunManyAsyncRouting:
    pytestmark = pytest.mark.asyncio

    async def test_routes_to_score_options_async_when_supported(
        self, runner_question_row, tmp_path
    ):
        """When supports_score_options() is True, run_many_async() calls score_options_async()."""
        backend = _MockVLLMBackend(score_async_responses=[_UNIFORM_RESP])
        runner = _make_vllm_runner(backend, tmp_path)
        df = pd.DataFrame([runner_question_row])

        results = await runner.run_many_async(df)

        assert len(results) == 1
        assert backend._raw_client._call_count == 1
        assert results[0]["pride_inference_mode"] == "eq8_transfer"

    async def test_falls_back_to_super_when_not_supported(
        self, runner_question_row, tmp_path
    ):
        """When supports_score_options() is False, run_many_async() delegates to super()."""
        # MockBackend inherits BaseBackend.supports_score_options() → False.
        backend = MockBackend(supports_logprobs=True)
        runner = PriDeRunner(
            backend=backend,
            method_name="pride",
            split_name="test",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="fallback_test",
            calibration_n=0,
            calibration_seed=0,
            calibration_runs_dir=tmp_path,
            calibration_questions=[],
        )
        df = pd.DataFrame([runner_question_row])
        sentinel = [{"question_id": "from_super"}]

        with patch.object(
            ExperimentRunner,
            "run_many_async",
            new=AsyncMock(return_value=sentinel),
        ):
            result = await runner.run_many_async(df)

        assert result is sentinel

    async def test_multiple_questions_all_scored(
        self, runner_question_row, tmp_path
    ):
        """run_many_async() returns one result per question."""
        n = 3
        backend = _MockVLLMBackend(score_async_responses=[_UNIFORM_RESP] * n)
        runner = _make_vllm_runner(backend, tmp_path)
        df = pd.DataFrame([runner_question_row] * n)

        results = await runner.run_many_async(df)

        assert len(results) == n
        assert backend._raw_client._call_count == n


# ---------------------------------------------------------------------------
# Tests: output row structure
# ---------------------------------------------------------------------------

class TestPriDeAsyncRowStructure:
    pytestmark = pytest.mark.asyncio

    async def test_async_row_has_same_keys_as_sync(
        self, runner_question_row, tmp_path
    ):
        """_score_question_async() must produce exactly the same keys as run_one()."""
        async_backend = _MockVLLMBackend(score_async_responses=[_UNIFORM_RESP])
        async_runner = _make_vllm_runner(async_backend, tmp_path, run_id="async_key_test")
        df = pd.DataFrame([runner_question_row])
        async_rows = await async_runner.run_many_async(df)

        # Sync reference: MockBackend.score_options() path, same calibration settings.
        sync_backend = MockBackend(
            score_responses=[[-0.1, -2.0, -3.0, -4.0]],
            supports_logprobs=True,
        )
        sync_runner = PriDeRunner(
            backend=sync_backend,
            method_name="pride",
            split_name="test",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="sync_key_test",
            calibration_n=0,
            calibration_seed=0,
            calibration_runs_dir=tmp_path,
            calibration_questions=[],
        )
        sync_runner._ensure_calibration()
        sync_row = sync_runner.run_one(runner_question_row, 0)

        assert set(async_rows[0].keys()) == set(sync_row.keys())

    async def test_async_row_has_pride_extra_fields(
        self, runner_question_row, tmp_path
    ):
        """All PriDe-specific output fields must be present in the async result."""
        backend = _MockVLLMBackend(score_async_responses=[_UNIFORM_RESP])
        runner = _make_vllm_runner(backend, tmp_path)
        df = pd.DataFrame([runner_question_row])

        results = await runner.run_many_async(df)
        row = results[0]

        assert row["pride_inference_mode"] == "eq8_transfer"
        assert row["pride_adjusted_choice"] is not None
        assert row["peprior_json"] is not None
        assert row["option_logprob_json"] is not None

    async def test_async_row_scores_correct_answer(
        self, runner_question_row, tmp_path
    ):
        """With uniform prior and C having the highest logprob → is_correct=True."""
        # correct_option is "C"; give C the highest logprob
        resp = {"A": -3.0, "B": -2.5, "C": -0.1, "D": -4.0}
        backend = _MockVLLMBackend(score_async_responses=[resp])
        runner = _make_vllm_runner(backend, tmp_path)
        df = pd.DataFrame([runner_question_row])

        results = await runner.run_many_async(df)

        assert results[0]["is_correct"] is True
        assert results[0]["pride_adjusted_choice"] == "C"

    async def test_async_row_error_when_score_options_async_raises(
        self, runner_question_row, tmp_path
    ):
        """When score_options_async() raises, the error is recorded and is_correct is None."""
        backend = _MockVLLMBackend(
            score_async_responses=[RuntimeError("connection timeout")]
        )
        runner = _make_vllm_runner(backend, tmp_path)
        df = pd.DataFrame([runner_question_row])

        results = await runner.run_many_async(df)
        row = results[0]

        assert row["pride_adjusted_choice"] is None
        assert row["option_logprob_json"] is None
        assert row["is_correct"] is None
        assert "connection timeout" in (row.get("error_message") or "")


# ---------------------------------------------------------------------------
# Tests: _cyclic_rollout_prob_matrix_async()
# ---------------------------------------------------------------------------

class TestPriDeCyclicRolloutAsync:
    pytestmark = pytest.mark.asyncio

    async def test_returns_correct_shape(self, runner_question_row, tmp_path):
        """_cyclic_rollout_prob_matrix_async() must return shape (n_permutations, n_letters)."""
        n_perm, n_letters = 4, 4
        resp = {"A": -0.1, "B": -1.0, "C": -2.0, "D": -3.0}
        backend = _MockVLLMBackend(score_async_responses=[resp] * n_perm)
        runner = _make_vllm_runner(backend, tmp_path)

        mat = await runner._cyclic_rollout_prob_matrix_async(runner_question_row)

        assert mat.shape == (n_perm, n_letters)
        assert mat.dtype == np.float64

    async def test_fallback_to_uniform_when_one_permutation_fails(
        self, runner_question_row, tmp_path
    ):
        """When one permutation's call fails, it gets a uniform row; shape is preserved."""
        good = {"A": -0.1, "B": -1.0, "C": -2.0, "D": -3.0}
        # Third call raises; the other three succeed.
        responses = [good, good, RuntimeError("network error"), good]
        backend = _MockVLLMBackend(score_async_responses=responses)
        runner = _make_vllm_runner(backend, tmp_path)

        mat = await runner._cyclic_rollout_prob_matrix_async(runner_question_row)

        assert mat.shape == (4, 4)
        # All rows must be valid probability distributions (sum to 1).
        np.testing.assert_allclose(mat.sum(axis=1), np.ones(4), atol=1e-6)

    async def test_raises_when_all_permutations_fail(
        self, runner_question_row, tmp_path
    ):
        """RuntimeError when every permutation's score_options_async() call fails."""
        responses = [RuntimeError("server down")] * 4
        backend = _MockVLLMBackend(score_async_responses=responses)
        runner = _make_vllm_runner(backend, tmp_path)

        with pytest.raises(RuntimeError, match="score_options_async"):
            await runner._cyclic_rollout_prob_matrix_async(runner_question_row)
