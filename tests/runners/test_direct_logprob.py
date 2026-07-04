# tests/runners/test_direct_logprob.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from choicebench.methods.library.direct_logprob import DirectLogprobRunner
from choicebench.methods.library.pride_math import logprob_map_to_label_distribution

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend, **kw) -> DirectLogprobRunner:
    return DirectLogprobRunner(
        backend=backend,
        method_name="direct_logprob",
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run",
        **kw,
    )


class TestDirectLogprobRunnerClassAttributes:
    def test_requires_score_options_is_true(self):
        assert DirectLogprobRunner.requires_score_options is True

    def test_applies_modal_k_gate_is_false(self):
        assert DirectLogprobRunner.applies_modal_k_gate is False


class TestDirectLogprobRunnerConstructionGuard:
    def test_raises_when_backend_does_not_support_logprobs(self):
        backend = MockBackend(supports_logprobs=False)
        with pytest.raises(ValueError, match="score_options"):
            _make_runner(backend)


class TestDirectLogprobRunnerRunOne:
    def test_argmax_picks_correct_letter(self, runner_question_row):
        # correct_option is "C". Give C the highest logprob.
        backend = MockBackend(
            score_responses=[[-2.0, -3.0, -0.1, -4.0]],
            supports_logprobs=True,
        )
        row = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert row["parsed_choice"] == "C"
        assert row["is_correct"] is True
        assert row["direct_logprob_inference_mode"] == "argmax"

    def test_argmax_picks_wrong_letter_when_scored_higher(self, runner_question_row):
        backend = MockBackend(
            score_responses=[[-0.1, -3.0, -4.0, -5.0]],  # A wins, gold is C
            supports_logprobs=True,
        )
        row = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert row["parsed_choice"] == "A"
        assert row["is_correct"] is False

    def test_missing_letter_handling(self, runner_question_row):
        """score_options returning fewer scores than letters leaves the
        trailing letter(s) out of lp_map (zip truncation); those letters must
        still get a floored, near-zero probability rather than crashing."""
        # 4 letters (A-D) but only 3 scores -> D is missing from lp_map.
        backend = MockBackend(
            score_responses=[[-0.1, -5.0, -5.0]],
            supports_logprobs=True,
        )
        row = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert row["parsed_choice"] == "A"
        mat = json.loads(row["option_distributions_json"])
        dist = mat[0]
        assert len(dist) == 4
        assert abs(sum(dist) - 1.0) < 1e-6
        assert dist[3] < dist[0]  # D (missing/floored) loses to explicit A

    def test_scoring_failure_leaves_no_answer(self, runner_question_row):
        backend = MockBackend(
            score_responses=[RuntimeError("server down")],
            supports_logprobs=True,
        )
        row = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert row["parsed_choice"] is None
        assert row["is_correct"] is None
        assert row["answer_status"] == "failure"
        assert row["option_distributions_json"] is None

    def test_makes_exactly_one_score_options_call(self, runner_question_row):
        backend = MockBackend(
            score_responses=[[-0.1, -2.0, -3.0, -4.0]],
            supports_logprobs=True,
        )
        _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert backend._score_call_count == 1

    def test_never_calls_generate(self, runner_question_row):
        backend = MockBackend(
            score_responses=[[-0.1, -2.0, -3.0, -4.0]],
            supports_logprobs=True,
        )
        _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert len(backend.requests_received) == 0

    def test_base_schema_fields_present(self, runner_question_row):
        backend = MockBackend(
            score_responses=[[-0.1, -2.0, -3.0, -4.0]],
            supports_logprobs=True,
        )
        row = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        for field in ("run_id", "question_id", "method_name", "split_name",
                      "provider", "model_name", "prompt"):
            assert field in row, f"missing field: {field}"
        assert row["method_name"] == "direct_logprob"


class TestOptionDistributionsRoundTrip:
    def test_distribution_round_trips_through_csv_exactly(self, runner_question_row, tmp_path):
        scores = [-0.1, -2.3456789, -3.0, -4.9999999]
        backend = MockBackend(score_responses=[scores], supports_logprobs=True)
        row = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        expected = logprob_map_to_label_distribution(
            dict(zip(["A", "B", "C", "D"], scores)), letters=["A", "B", "C", "D"]
        ).tolist()

        csv_path = tmp_path / "result.csv"
        pd.DataFrame([row]).to_csv(csv_path, index=False)
        reloaded = pd.read_csv(csv_path)

        round_tripped = json.loads(reloaded.loc[0, "option_distributions_json"])[0]
        assert round_tripped == expected
