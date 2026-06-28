# tests/runners/test_pride.py

from pathlib import Path

import pytest

from choicebench.backends.dummy_backend import DummyBackend
from choicebench.methods.library.pride import PriDeRunner

from tests.runners.conftest import MockBackend


REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


class TestPriDeRunnerIntegration:
    def test_raises_without_logprob_support(self, tmp_path: Path):
        """Constructing PriDeRunner on a backend without score_options() support
        should raise — this replaces the old client.provider != "together" check
        now that the capability check is backend.supports_logprobs."""
        backend = MockBackend(provider="openai", model_name="gpt-5-mini", supports_logprobs=False)
        with pytest.raises(ValueError):
            PriDeRunner(
                backend=backend,
                method_name="pride",
                split_name="robustness",
                prompt_version="v1",
                prompts_dir=_PROMPTS_DIR,
                run_id="r1",
                calibration_n=0,
                calibration_seed=0,
                calibration_benchmark="mmlu",
                calibration_runs_dir=tmp_path,
                calibration_questions=[],
            )

    def test_missing_trailing_option_fails_clearly_in_v01(
            self, runner_question_row, tmp_path: Path,
    ):
        row = dict(runner_question_row, choice_d="", correct_option="C")
        runner = PriDeRunner(
            backend=DummyBackend(),
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="pride_missing_option",
            calibration_n=0,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[],
        )

        with pytest.raises(ValueError, match="PriDe requires four valid A-D options"):
            runner.run_one(row, sample_index=0)

    def test_no_calibration_questions_uses_uniform_prior(
            self, runner_question_row, tmp_path: Path,
    ):
        """Empty calibration pool → uniform prior (all letters = 0.25)."""
        import json
        # 1 score_options call for the single eval question; no calibration calls.
        backend = MockBackend(
            score_responses=[[-1.0, -2.0, -3.0, -4.0]],
            supports_logprobs=True,
        )
        runner = PriDeRunner(
            backend=backend,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="pride_no_cal",
            calibration_n=50,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[],
        )

        rows = runner.run_many([runner_question_row])
        assert len(rows) == 1
        assert rows[0]["pride_inference_mode"] == "eq8_transfer"
        prior = json.loads(rows[0]["peprior_json"])
        for v in prior.values():
            assert abs(v - 0.25) < 1e-6, f"Prior should be uniform; got {prior}"

    def test_calibration_questions_produce_learned_prior(
            self, runner_question_row, tmp_path: Path,
    ):
        """K=1 calibration question with working score_options → prior deviates from uniform.

        4 calibration score_options calls (one per cyclic permutation) then
        1 eval call. PriDe uses score_options only — generate() is never called.
        """
        import json
        # Strongly prefer position A on every permutation → positional bias toward A.
        biased = [-0.1, -5.0, -5.0, -5.0]
        backend = MockBackend(
            score_responses=[biased, biased, biased, biased,  # 4 calibration permutations
                             [-1.0, -2.0, -3.0, -4.0]],      # 1 eval question
            supports_logprobs=True,
        )
        cal_row = {**runner_question_row, "question_id": "cal_001"}
        runner = PriDeRunner(
            backend=backend,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="pride_learned",
            calibration_n=1,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[cal_row],
        )

        rows = runner.run_many([runner_question_row])
        assert len(rows) == 1
        assert rows[0]["pride_inference_mode"] == "eq8_transfer"
        prior = json.loads(rows[0]["peprior_json"])
        # A prior estimated from strongly position-biased logprobs must deviate from uniform.
        assert any(abs(v - 0.25) > 0.01 for v in prior.values()), (
            f"Prior should deviate from uniform when calibration data is provided; got {prior}"
        )
        # generate() is never called by PriDe.
        assert len(backend.requests_received) == 0

    def test_calibration_fails_clearly_when_score_options_always_raises(
            self, runner_question_row, tmp_path: Path,
    ):
        """Calibration questions + score_options always raises → RuntimeError, not silent uniform.

        If calibration_questions are provided but every score_options call fails
        (e.g. backend claims supports_logprobs=True but raises NotImplementedError),
        PriDe must raise rather than silently fall back to a uniform prior that is
        indistinguishable from the no-calibration case.
        """
        # No score_responses queued → every score_options call raises NotImplementedError.
        backend = MockBackend(supports_logprobs=True)
        cal_row = {**runner_question_row, "question_id": "cal_001"}
        runner = PriDeRunner(
            backend=backend,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="pride_broken_cal",
            calibration_n=1,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[cal_row],
        )

        with pytest.raises(RuntimeError, match="score_options"):
            runner.run_many([runner_question_row])
