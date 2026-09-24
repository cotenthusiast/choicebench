# tests/runners/test_independent_hypothesis.py

from pathlib import Path

import pandas as pd
import pytest

from choicebench.backends.api_backend import APIBackend
from choicebench.clients.types import ModelResponse, SUCCESS_STATUS
from choicebench.methods.library.independent_hypothesis import IndependentHypothesisRunner
from choicebench.parsing.types import PARSE_OK
from choicebench.scoring.types import SCORE_CORRECT, SCORE_INCORRECT
from choicebench.scoring.tiebreak import resolve_tie

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


class _AsyncMockBackend(APIBackend):
    """Minimal APIBackend stand-in: generate_batch() returns queued responses
    in order and records the prompts it received."""

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


def _make_runner(backend, *, seed=42, benchmark_name="mmlu"):
    return IndependentHypothesisRunner(
        backend=backend,
        method_name="independent_hypothesis",
        split_name="test",
        prompt_version="v1_ihs",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
        seed=seed,
        benchmark_name=benchmark_name,
    )


def _score_response(score: float, analysis: str = "brief analysis") -> str:
    return f"{analysis} <score>{score}</score>"


class TestIndependentHypothesisRunOne:
    def test_four_options_makes_exactly_four_calls(self, runner_question_row):
        backend = MockBackend(responses=[
            _score_response(10), _score_response(20), _score_response(90), _score_response(5),
        ])
        _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert len(backend.requests_received) == 4

    def test_three_real_options_makes_exactly_three_calls(self, runner_question_row_missing_trailing_option):
        backend = MockBackend(responses=[
            _score_response(10), _score_response(20), _score_response(90),
        ])
        _make_runner(backend).run_one(runner_question_row_missing_trailing_option, sample_index=0)
        assert len(backend.requests_received) == 3

    def test_no_option_ever_appears_alongside_another_in_a_prompt(self, runner_question_row):
        """Core structural property of IHS: each prompt names exactly one
        option's text, never a list."""
        backend = MockBackend(responses=[
            _score_response(10), _score_response(20), _score_response(90), _score_response(5),
        ])
        _make_runner(backend).run_one(runner_question_row, sample_index=0)
        # "HTTP" is a substring of "HTTPS", so a naive substring check on
        # this fixture's options would false-positive -- assert against the
        # exact "Hypothesis: The correct answer is X." line instead.
        expected_lines = [
            "Hypothesis: The correct answer is FTP.",
            "Hypothesis: The correct answer is HTTP.",
            "Hypothesis: The correct answer is HTTPS.",
            "Hypothesis: The correct answer is SMTP.",
        ]
        for prompt in backend.requests_received:
            present = [line for line in expected_lines if line in prompt]
            assert len(present) == 1, f"prompt should name exactly one option, saw {present}: {prompt!r}"

    def test_argmax_picks_highest_scoring_option(self, runner_question_row):
        backend = MockBackend(responses=[
            _score_response(10), _score_response(20), _score_response(90), _score_response(5),
        ])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["parsed_choice"] == "C"  # 90 is highest
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_incorrect_argmax(self, runner_question_row):
        backend = MockBackend(responses=[
            _score_response(95), _score_response(20), _score_response(10), _score_response(5),
        ])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_last_occurrence_score_wins(self, runner_question_row):
        """Regex extraction prefers the final restated score over an earlier draft."""
        backend = MockBackend(responses=[
            "draft <score>10</score> revised <score>99</score>",
            _score_response(20), _score_response(30), _score_response(5),
        ])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["option_a_score"] == 99.0
        assert result["parsed_choice"] == "A"  # 99 > 30

    def test_parse_failure_scores_zero_but_still_participates(self, runner_question_row):
        """No <score> tag at all -> 0.0, parse_ok False, but the option is
        NOT excluded from the argmax (a 0 is a real candidate, not missing)."""
        backend = MockBackend(responses=[
            "no score tag here at all", _score_response(20), _score_response(-5), _score_response(-10),
        ])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["option_a_score"] == 0.0
        assert result["option_a_score_parse_ok"] is False
        assert result["parsed_choice"] == "B"  # 20 is the max including A's forced 0.0

    def test_backend_call_failure_scores_zero(self, runner_question_row):
        from choicebench.clients.types import ProviderTimeoutError
        backend = MockBackend(responses=[
            ProviderTimeoutError("timed out"), _score_response(20), _score_response(30), _score_response(5),
        ])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["option_a_score"] == 0.0
        assert result["option_a_score_parse_ok"] is False
        assert result["parsed_choice"] == "C"

    def test_per_option_fields_are_persisted(self, runner_question_row):
        backend = MockBackend(responses=[
            _score_response(10), _score_response(20), _score_response(90), _score_response(5),
        ])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        for letter, score in [("a", 10.0), ("b", 20.0), ("c", 90.0), ("d", 5.0)]:
            assert result[f"option_{letter}_score"] == score
            assert result[f"option_{letter}_score_parse_ok"] is True
            assert result[f"option_{letter}_raw_text"] is not None

    def test_exact_tie_routes_through_shared_tiebreak_utility(self, runner_question_row):
        backend = MockBackend(responses=[
            _score_response(50), _score_response(90), _score_response(90), _score_response(5),
        ])
        result = _make_runner(backend, seed=42).run_one(runner_question_row, sample_index=0)

        # B (source_index 1) and C (source_index 2) are tied at 90.
        expected_id = resolve_tie(
            seed=42, benchmark_id="mmlu", question_id=runner_question_row["question_id"],
            method_name="independent_hypothesis", tied_canonical_ids=[1, 2],
        )
        expected_letter = "B" if expected_id == 1 else "C"
        assert result["parsed_choice"] == expected_letter


class TestIndependentHypothesisRunManyAsync:
    @pytest.mark.asyncio
    async def test_batched_path_matches_sync_path_call_count(self, runner_question_row):
        backend = _AsyncMockBackend([
            _score_response(10), _score_response(20), _score_response(90), _score_response(5),
        ])
        df = pd.DataFrame([runner_question_row])
        results = await _make_runner(backend).run_many_async(df)
        assert len(results) == 1
        assert results[0]["parsed_choice"] == "C"
        assert len(backend.prompts_received) == 4

    @pytest.mark.asyncio
    async def test_batched_path_handles_multiple_questions(self, runner_question_row):
        # 2 questions x 4 options = 8 calls total, all in one generate_batch().
        backend = _AsyncMockBackend([
            _score_response(10), _score_response(20), _score_response(90), _score_response(5),
            _score_response(80), _score_response(10), _score_response(5), _score_response(5),
        ])
        df = pd.DataFrame([dict(runner_question_row), dict(runner_question_row)])
        results = await _make_runner(backend).run_many_async(df)
        assert len(results) == 2
        assert len(backend.prompts_received) == 8
        assert results[0]["parsed_choice"] == "C"
        assert results[1]["parsed_choice"] == "A"
