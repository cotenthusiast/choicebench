# tests/runners/test_pride.py

from pathlib import Path

import pandas as pd
import pytest

from choicebench.backends.dummy_backend import DummyBackend
from choicebench.methods.library.pride import PriDeRunner
from choicebench.metrics.accuracy import Accuracy
from choicebench.metrics.mad import MAD

from tests.runners.conftest import MockBackend


REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_question_row(qid: str, correct_option: str) -> dict:
    """A normalized 4-option question row with the given gold letter."""
    return {
        "question_id": qid,
        "subject": "demo",
        "question_text": f"Question {qid}?",
        "choice_a": "alpha",
        "choice_b": "bravo",
        "choice_c": "charlie",
        "choice_d": "delta",
        "correct_option": correct_option,
        "correct_answer_text": {"A": "alpha", "B": "bravo", "C": "charlie", "D": "delta"}[correct_option],
    }


class TestPriDeMetricsIntegration:
    def test_pride_is_scored_by_builtin_metrics(self, tmp_path: Path):
        """Regression for MF-A / FSF-1: PriDe must populate parsed_choice so the
        built-in metrics score it. Accuracy must match the row-level is_correct
        rate (not 0.0) and MAD must not be NaN."""
        rows = [
            _make_question_row("q0", "A"),
            _make_question_row("q1", "B"),
            _make_question_row("q2", "A"),
            _make_question_row("q3", "C"),
        ]
        runner = PriDeRunner(
            backend=DummyBackend(),
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="pride_metrics",
            calibration_n=0,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[],
        )

        result_rows = [runner.run_one(r, i) for i, r in enumerate(rows)]
        df = pd.DataFrame(result_rows)

        # Every row must carry parsed_choice == its debiased answer.
        assert df["parsed_choice"].notna().all()

        # PF-3: success rows must carry a non-None model_status even though
        # PriDe never calls generate().
        assert df["model_status"].notna().all()
        assert (df["model_status"] == "success").all()

        actual_correct_rate = df["is_correct"].mean()
        acc = Accuracy().compute(df)
        assert acc["accuracy"] == pytest.approx(actual_correct_rate)
        # DummyBackend always scores A highest → 2 of 4 gold answers are A.
        assert acc["accuracy"] == pytest.approx(0.5)

        mad = MAD().compute(df)
        assert mad["mad"] == mad["mad"]  # not NaN


class TestPriDePartialFailureFlag:
    @pytest.mark.parametrize("n_failed", [0, 2, 3])
    def test_eval_rows_report_calibration_permutation_failures(
        self, runner_question_row, tmp_path: Path, n_failed: int
    ):
        """PF-4: n_permutations_failed/_total reflect uniform-fallback permutations
        during calibration (PriDe's per-row score is a single call). Tested for
        0 (none), 2 (some), and 3 (all-but-one of 4) failures."""
        good = [-0.1, -2.0, -3.0, -4.0]
        # 4 calibration permutation calls (some fail → uniform), then 1 eval call.
        cal_calls = [good] * (4 - n_failed) + [RuntimeError("down")] * n_failed
        backend = MockBackend(
            score_responses=cal_calls + [good],  # + eval call
            supports_logprobs=True,
        )
        cal_row = {**runner_question_row, "question_id": "cal_001"}
        runner = PriDeRunner(
            backend=backend,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id=f"pride_pf4_{n_failed}",
            calibration_n=1,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[cal_row],
        )

        rows = runner.run_many(pd.DataFrame([runner_question_row]))
        assert rows[0]["n_permutations_total"] == 4
        assert rows[0]["n_permutations_failed"] == n_failed


class TestPriDeConstruction:
    def test_accepts_and_forwards_model_label(self, tmp_path: Path):
        """Regression: instantiate_runner always passes model_label=, so PriDe's
        __init__ must accept and forward it (otherwise pride can never be built
        through the orchestrator)."""
        runner = PriDeRunner(
            backend=DummyBackend(),
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="pride_label",
            model_label="org/my-model",
            calibration_n=0,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[],
        )
        assert runner.model_label == "org/my-model"


class TestPriDeSidecarKey:
    def test_sidecar_records_generation_settings(self, runner_question_row, tmp_path: Path):
        """PF-9: the sidecar payload must record temperature/max_tokens/prompt_version
        so a prior fit under different settings is not silently reused."""
        import json

        biased = [-0.1, -5.0, -5.0, -5.0]
        backend = MockBackend(
            score_responses=[biased] * 4 + [[-1.0, -2.0, -3.0, -4.0]],
            supports_logprobs=True,
        )
        cal_row = {**runner_question_row, "question_id": "cal_001"}
        runner = PriDeRunner(
            backend=backend,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="pride_sidecar",
            temperature=0.7,
            max_tokens=128,
            calibration_n=1,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[cal_row],
        )
        runner.run_many(pd.DataFrame([runner_question_row]))

        sidecar = json.loads(runner._sidecar_path().read_text())
        assert sidecar["temperature"] == 0.7
        assert sidecar["max_tokens"] == 128
        assert sidecar["prompt_version"] == "v1"


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

        with pytest.raises(ValueError, match="PriDe requires exactly 4 valid options"):
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

        rows = runner.run_many(pd.DataFrame([runner_question_row]))
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

        rows = runner.run_many(pd.DataFrame([runner_question_row]))
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
            runner.run_many(pd.DataFrame([runner_question_row]))
