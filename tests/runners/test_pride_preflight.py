# tests/runners/test_pride_preflight.py
#
# Tests for PriDe's preflight_questions / calibration_questions precedence rules.

from pathlib import Path

import pytest

from choicebench.backends.dummy_backend import DummyBackend
from choicebench.methods.library.pride import PriDeRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(tmp_path, **kwargs) -> PriDeRunner:
    return PriDeRunner(
        backend=DummyBackend(),
        method_name="pride",
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="preflight_test",
        calibration_runs_dir=tmp_path,
        **kwargs,
    )


_CAL_ROW = {
    "question_id": "cal_q1",
    "subject": "general",
    "question_text": "Which is correct?",
    "choice_a": "Alpha",
    "choice_b": "Beta",
    "choice_c": "Gamma",
    "choice_d": "Delta",
    "correct_option": "A",
    "correct_answer_text": "Alpha",
}


class TestPriDePreflightPrecedence:
    def test_neither_provided_uses_uniform_prior(self, tmp_path):
        """Neither calibration_questions nor preflight_questions → uniform prior."""
        runner = _make_runner(
            tmp_path,
            calibration_questions=None,
            preflight_questions=None,
            calibration_n=0,
        )
        runner._ensure_calibration()
        probs = runner._calibration_state.peprior_probs
        # Uniform prior: all four letters have equal probability.
        values = list(probs.values())
        assert all(abs(v - values[0]) < 1e-9 for v in values)

    def test_calibration_questions_only_is_used(self, tmp_path):
        """calibration_questions provided, no preflight → uses calibration_questions."""
        runner = _make_runner(
            tmp_path,
            calibration_questions=[_CAL_ROW],
            preflight_questions=None,
            calibration_n=1,
        )
        # Should not raise; calibration questions are set.
        assert runner._calibration_questions == [_CAL_ROW]

    def test_preflight_questions_only_used_as_calibration(self, tmp_path):
        """preflight_questions provided, no calibration_questions → preflight is used."""
        runner = _make_runner(
            tmp_path,
            calibration_questions=None,
            preflight_questions=[_CAL_ROW],
            calibration_n=1,
        )
        # The constructor should have promoted preflight to calibration.
        assert runner._calibration_questions == [_CAL_ROW]

    def test_calibration_takes_precedence_over_preflight(self, tmp_path):
        """Both provided → calibration_questions wins; preflight_questions is ignored."""
        cal_row = dict(_CAL_ROW, question_id="explicit_cal")
        pre_row = dict(_CAL_ROW, question_id="preflight_row")
        runner = _make_runner(
            tmp_path,
            calibration_questions=[cal_row],
            preflight_questions=[pre_row],
            calibration_n=1,
        )
        # calibration_questions must win.
        assert runner._calibration_questions == [cal_row]
        assert all(r["question_id"] != "preflight_row" for r in runner._calibration_questions)
