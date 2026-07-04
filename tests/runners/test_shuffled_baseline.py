# tests/runners/test_shuffled_baseline.py

from __future__ import annotations

import random
from pathlib import Path

from choicebench.clients.types import ProviderTimeoutError
from choicebench.methods.library.shuffled_baseline import ShuffledBaselineRunner
from choicebench.scoring.types import SCORE_CORRECT, SCORE_UNSCORABLE
from choicebench.parsing.types import PARSE_OK

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend, seed=0, method_name="shuffled_baseline"):
    return ShuffledBaselineRunner(
        backend=backend,
        method_name=method_name,
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run",
        seed=seed,
    )


class TestShuffledBaselineRunnerClassAttributes:
    def test_requires_score_options_is_false(self):
        assert ShuffledBaselineRunner.requires_score_options is False

    def test_applies_modal_k_gate_is_false(self):
        assert ShuffledBaselineRunner.applies_modal_k_gate is False


class TestShuffleOptions:
    def test_shuffle_is_a_bijection_over_letters_and_text(self):
        canon = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        rng = random.Random(0)
        shuffled, label_map = ShuffledBaselineRunner._shuffle_options(canon, rng)

        assert set(shuffled.keys()) == set(canon.keys())
        assert sorted(shuffled.values()) == sorted(canon.values())
        assert set(label_map.keys()) == set(canon.keys())
        assert set(label_map.values()) == set(canon.keys())
        for shuffled_label, canonical_label in label_map.items():
            assert shuffled[shuffled_label] == canon[canonical_label]

    def test_shuffle_is_deterministic_for_same_rng_seed(self):
        canon = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        shuffled_1, map_1 = ShuffledBaselineRunner._shuffle_options(canon, random.Random(7))
        shuffled_2, map_2 = ShuffledBaselineRunner._shuffle_options(canon, random.Random(7))
        assert shuffled_1 == shuffled_2
        assert map_1 == map_2


class TestShuffledBaselineRunnerRunOne:
    def test_run_one_remaps_shuffled_letter_to_canonical(self, runner_question_row):
        """The model answers with a shuffled-prompt letter; run_one must score
        against the canonical letter it maps back to, not the raw letter."""
        canon = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        seed = 0
        rng = random.Random(f"{seed}:{runner_question_row['question_id']}")
        shuffled, label_map = ShuffledBaselineRunner._shuffle_options(canon, rng)
        shuffled_letter_for_correct = next(
            letter for letter, text in shuffled.items() if text == "HTTPS"
        )

        backend = MockBackend(responses=[shuffled_letter_for_correct])
        result = _make_runner(backend, seed=seed).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT
        assert result["parse_status"] == PARSE_OK

    def test_run_one_matches_answer_text_regardless_of_shuffle(self, runner_question_row):
        """Text-match fallback works against the shuffled options too."""
        backend = MockBackend(responses=["The answer is HTTPS."])
        result = _make_runner(backend, seed=1).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_failed_response(self, runner_question_row):
        backend = MockBackend(responses=[ProviderTimeoutError("Request timed out.")])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["error_type"] == "ProviderTimeoutError"

    def test_unparseable_response(self, runner_question_row):
        backend = MockBackend(responses=["I'm not sure about this question"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    def test_prompt_contains_all_option_texts(self, runner_question_row):
        backend = MockBackend(responses=["The answer is HTTPS."])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert "FTP" in result["prompt"]
        assert "HTTP" in result["prompt"]
        assert "HTTPS" in result["prompt"]
        assert "SMTP" in result["prompt"]

    def test_same_question_same_seed_produces_same_prompt(self, runner_question_row):
        """The shuffle is seeded per (run seed, question_id) — repeat calls
        for the same question in the same run must render an identical prompt."""
        backend = MockBackend(responses=["The answer is HTTPS.", "The answer is HTTPS."])
        runner = _make_runner(backend, seed=3)
        result_1 = runner.run_one(runner_question_row, sample_index=0)
        result_2 = runner.run_one(runner_question_row, sample_index=1)

        assert result_1["prompt"] == result_2["prompt"]
