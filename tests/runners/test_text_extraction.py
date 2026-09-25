# tests/runners/test_text_extraction.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from choicebench.backends.api_backend import APIBackend
from choicebench.clients.types import ModelResponse, SUCCESS_STATUS
from choicebench.methods.library.text_extraction import TextExtractionRunner
from choicebench.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE

from tests.runners.conftest import MockBackend


class _AsyncMockBackend(APIBackend):
    """Minimal APIBackend stand-in: generate_batch() returns queued responses
    in order and records the prompts it received -- same double every other
    runner's batched-path tests use (test_independent_hypothesis.py etc.)."""

    def __init__(self, texts: list[str]) -> None:
        self._provider = "openai"
        self._model_name = "mock-model"
        self._temperature = 0.0
        self._max_tokens = 512
        self._seed = 42
        self._concurrency_limit = len(texts) or 10
        self._queued = list(texts)
        self.prompts_received: list[str] = []

    async def generate_batch(self, prompts: list[str]) -> list[ModelResponse]:
        self.prompts_received.extend(prompts)
        out = [
            ModelResponse(
                provider=self._provider, model_name=self._model_name,
                status=SUCCESS_STATUS, latency_seconds=0.0, raw_text=text,
                finish_reason="stop", usage=None, error=None, timestamp_utc=None,
            )
            for text in self._queued[: len(prompts)]
        ]
        self._queued = self._queued[len(prompts):]
        return out

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _no_match_embed_fn(texts: list[str]) -> np.ndarray:
    """Stub embedder: every text gets an orthogonal one-hot vector, so
    cosine similarity between any two distinct texts is always 0 --
    guarantees the cosine stage stays below threshold without needing the
    real sentence-transformers model in these fast, offline unit tests."""
    n = len(texts)
    return np.eye(n, dtype=np.float64)


def _make_runner(backend, embed_fn=_no_match_embed_fn):
    return TextExtractionRunner(
        backend=backend,
        method_name="text_extraction",
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
        seed=42,
        benchmark_name="mmlu",
        embed_fn=embed_fn,
    )


class TestTextExtractionRunOne:
    def test_one_call_per_question(self, runner_question_row):
        backend = MockBackend(responses=["HTTPS"])
        _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert len(backend.requests_received) == 1

    def test_options_are_visible_in_the_prompt(self, runner_question_row):
        """Distinguishes this from two_stage's hidden-options Stage 1."""
        backend = MockBackend(responses=["HTTPS"])
        _make_runner(backend).run_one(runner_question_row, sample_index=0)
        prompt = backend.requests_received[0]
        assert "A. FTP" in prompt
        assert "B. HTTP" in prompt
        assert "C. HTTPS" in prompt
        assert "D. SMTP" in prompt

    def test_exact_text_match_scores_correct(self, runner_question_row):
        backend = MockBackend(responses=["HTTPS"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_incorrect_text_match(self, runner_question_row):
        backend = MockBackend(responses=["FTP"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_unmatched_free_text_is_unscorable(self, runner_question_row):
        backend = MockBackend(responses=["completely unrelated gibberish response"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    def test_free_text_response_is_persisted(self, runner_question_row):
        backend = MockBackend(responses=["HTTPS"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["raw_text"] == "HTTPS"

    def test_backend_failure_returns_unscorable(self, runner_question_row):
        from choicebench.clients.types import ProviderTimeoutError
        backend = MockBackend(responses=[ProviderTimeoutError("timed out")])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["parsed_choice"] is None
        assert result["transport_status"] == "failure"

    def test_defaults_to_the_real_embedder_lazily_when_none_supplied(self, runner_question_row):
        """embed_fn=None (the constructor default) must defer to
        default_embed_fn -- checked without actually invoking it, so this
        stays a fast, offline assertion about the wiring, not a real
        model-load test."""
        from choicebench.scoring.text_matcher import default_embed_fn
        runner = TextExtractionRunner(
            backend=MockBackend(responses=["HTTPS"]),
            method_name="text_extraction", split_name="test",
            prompt_version="v1", prompts_dir=_PROMPTS_DIR,
            run_id="test_run_001", seed=42, benchmark_name="mmlu",
        )
        assert runner._effective_embed_fn() is default_embed_fn


class TestRunRotations:
    """text_extraction's options are VISIBLE in the prompt (unlike
    two_stage's hidden stage 1), so rotating the options is a genuine
    order-sensitivity probe -- unlike semantic_matching_v1's derivation,
    this requires a full new call per rotation, not a reused stage-1
    answer."""

    def test_makes_exactly_n_calls_one_per_rotation(self, runner_question_row):
        backend = MockBackend(responses=["HTTPS"] * 4)
        result = _make_runner(backend).run_rotations(runner_question_row, sample_index=0)
        assert len(backend.requests_received) == 4
        per_rotation = json.loads(result["per_rotation_choices_json"])
        assert len(per_rotation) == 4

    def test_options_are_actually_rotated_across_calls(self, runner_question_row):
        """Distinguishes this from four identical repeated calls: each
        prompt must show a genuinely different option ordering."""
        backend = MockBackend(responses=["HTTPS"] * 4)
        _make_runner(backend).run_rotations(runner_question_row, sample_index=0)
        assert len(set(backend.requests_received)) == 4

    def test_unanimous_correct_match_across_rotations(self, runner_question_row):
        backend = MockBackend(responses=["HTTPS"] * 4)
        result = _make_runner(backend).run_rotations(runner_question_row, sample_index=0)
        per_rotation = json.loads(result["per_rotation_choices_json"])
        assert per_rotation == ["C", "C", "C", "C"]
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_dissenting_rotation_is_visible_in_the_trace(self, runner_question_row):
        backend = MockBackend(responses=["HTTPS", "HTTPS", "HTTPS", "FTP"])
        result = _make_runner(backend).run_rotations(runner_question_row, sample_index=0)
        per_rotation = json.loads(result["per_rotation_choices_json"])
        assert per_rotation == ["C", "C", "C", "A"]
        assert result["parsed_choice"] == "C"  # majority vote still wins

    def test_all_rotations_fail_yields_no_parsed_choice(self, runner_question_row):
        from choicebench.clients.types import ProviderTimeoutError
        backend = MockBackend(responses=[ProviderTimeoutError("timed out") for _ in range(4)])
        result = _make_runner(backend).run_rotations(runner_question_row, sample_index=0)
        assert result["parsed_choice"] is None
        per_rotation = json.loads(result["per_rotation_choices_json"])
        assert per_rotation == [None, None, None, None]

    def test_result_row_has_metadata(self, runner_question_row):
        backend = MockBackend(responses=["HTTPS"] * 4)
        result = _make_runner(backend).run_rotations(runner_question_row, sample_index=0)
        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "text_extraction"
        assert result["split_name"] == "test"

    def test_persists_raw_text_per_rotation_for_downstream_reuse(self, runner_question_row):
        """visible_llm_matcher's own rotation rerun needs each rotation's
        actual extracted TEXT (not just the collapsed canonical letter) to
        feed its own Stage-2 LLM match under that same rotation."""
        backend = MockBackend(responses=["HTTPS", "FTP", "HTTPS", "HTTPS"])
        result = _make_runner(backend).run_rotations(runner_question_row, sample_index=0)
        per_rotation_text = json.loads(result["per_rotation_raw_text_json"])
        assert per_rotation_text == ["HTTPS", "FTP", "HTTPS", "HTTPS"]

    def test_raw_text_is_null_for_a_failed_rotation(self, runner_question_row):
        from choicebench.clients.types import ProviderTimeoutError
        backend = MockBackend(
            responses=[ProviderTimeoutError("timed out"), "FTP", "HTTPS", "HTTPS"]
        )
        result = _make_runner(backend).run_rotations(runner_question_row, sample_index=0)
        per_rotation_text = json.loads(result["per_rotation_raw_text_json"])
        assert per_rotation_text == [None, "FTP", "HTTPS", "HTTPS"]


class TestRunRotationsManyAsync:
    """Batched sibling of run_rotations(): flattens ALL (question, rotation)
    prompt pairs across every given question into ONE generate_batch() call.

    Required for genuine Batch API benefit -- run_rotations's own sequential
    per-rotation _call_backend_generate() calls would, against an
    async-capable backend, silently submit one single-request "batch" per
    call instead of one real batch job across everything, defeating the
    whole point of marking text_extraction rotations batch-safe."""

    pytestmark = pytest.mark.asyncio

    async def test_makes_exactly_one_generate_batch_call_for_all_questions(self, runner_question_row):
        # 2 questions x 4 rotations = 8 prompts, all in one generate_batch().
        backend = _AsyncMockBackend(["HTTPS"] * 8)
        df = pd.DataFrame([runner_question_row, runner_question_row])

        results = await _make_runner(backend).run_rotations_many_async(df)

        assert len(backend.prompts_received) == 8
        assert len(results) == 2

    async def test_per_question_results_are_independent(self, runner_question_row):
        # Question 1: unanimous HTTPS (correct). Question 2: unanimous FTP (incorrect).
        backend = _AsyncMockBackend(["HTTPS"] * 4 + ["FTP"] * 4)
        df = pd.DataFrame([runner_question_row, runner_question_row])

        results = await _make_runner(backend).run_rotations_many_async(df)

        assert results[0]["parsed_choice"] == "C"
        assert results[0]["is_correct"] is True
        assert results[1]["parsed_choice"] == "A"
        assert results[1]["is_correct"] is False

    async def test_per_rotation_fields_are_persisted_per_question(self, runner_question_row):
        backend = _AsyncMockBackend(["HTTPS"] * 4 + ["FTP"] * 4)
        df = pd.DataFrame([runner_question_row, runner_question_row])

        results = await _make_runner(backend).run_rotations_many_async(df)

        assert json.loads(results[0]["per_rotation_choices_json"]) == ["C", "C", "C", "C"]
        assert json.loads(results[1]["per_rotation_choices_json"]) == ["A", "A", "A", "A"]
        assert json.loads(results[0]["per_rotation_raw_text_json"]) == ["HTTPS"] * 4
        assert json.loads(results[1]["per_rotation_raw_text_json"]) == ["FTP"] * 4

    async def test_result_row_has_metadata(self, runner_question_row):
        backend = _AsyncMockBackend(["HTTPS"] * 4)
        df = pd.DataFrame([runner_question_row])

        results = await _make_runner(backend).run_rotations_many_async(df)

        assert results[0]["run_id"] == "test_run_001"
        assert results[0]["method_name"] == "text_extraction"
        assert results[0]["split_name"] == "test"
