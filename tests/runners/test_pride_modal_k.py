# tests/runners/test_pride_modal_k.py
#
# PriDe generalized to a modal option count != 4 (the modal-k gate guarantees a
# homogeneous option count, so PriDe calibrates and evaluates over that k).

import json
from pathlib import Path

import pytest

from choicebench.backends.dummy_backend import DummyBackend
from choicebench.benchmarks.base import make_normalized_row
from choicebench.methods.library.pride import PriDeRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "prompts"


def _row(n: int, correct_index: int, tag: str) -> dict:
    return make_normalized_row("cat", f"q_{tag}", [f"o{i}" for i in range(n)], correct_index)


def _make_runner(tmp_path, modal_k, calibration_questions, gate_summary=None):
    return PriDeRunner(
        backend=DummyBackend(),
        method_name="pride",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="r_modal_k",
        calibration_n=3,
        calibration_seed=0,
        calibration_benchmark="mmlu_pro",
        calibration_runs_dir=tmp_path,
        calibration_questions=calibration_questions,
        modal_k=modal_k,
        gate_summary=gate_summary,
    )


def test_pride_calibrates_and_debiases_at_k_six(tmp_path):
    cal = [_row(6, 0, f"cal{i}") for i in range(3)]
    runner = _make_runner(
        tmp_path, modal_k=6, calibration_questions=cal,
        gate_summary={"modal_k": 6, "n_total": 10, "n_evaluated": 8, "n_excluded": 2},
    )

    row = runner.run_one(_row(6, 2, "eval"), sample_index=0)

    # Debiased answer is over the full A-F label space.
    assert row["pride_adjusted_choice"] in list("ABCDEF")
    assert row["parsed_choice"] in list("ABCDEF")
    peprior = json.loads(row["peprior_json"])
    assert sorted(peprior.keys()) == list("ABCDEF")

    # Gate accounting is stamped on the row.
    assert row["gate_modal_k"] == 6
    assert row["gate_n_total"] == 10
    assert row["gate_n_evaluated"] == 8
    assert row["gate_n_excluded"] == 2


def test_pride_sidecar_records_six_options(tmp_path):
    cal = [_row(6, 0, f"cal{i}") for i in range(3)]
    runner = _make_runner(tmp_path, modal_k=6, calibration_questions=cal)
    runner.run_one(_row(6, 1, "eval"), sample_index=0)

    sidecar = next(tmp_path.rglob("pride_calibration__*.json"))
    blob = json.loads(sidecar.read_text())
    assert blob["n_options"] == 6
    assert sorted(blob["peprior_probs"].keys()) == list("ABCDEF")


def test_calibration_drops_non_modal_k_questions(tmp_path):
    # Two 6-option calibration rows plus a 4-option row that must be filtered out.
    cal = [_row(6, 0, "a"), _row(6, 1, "b"), _row(4, 0, "wrong_k")]
    runner = _make_runner(tmp_path, modal_k=6, calibration_questions=cal)
    # If the 4-option row were not filtered, calibration would raise on the
    # length mismatch; a clean run proves it was dropped.
    row = runner.run_one(_row(6, 0, "eval"), sample_index=0)
    assert row["pride_adjusted_choice"] in list("ABCDEF")


def test_eval_question_with_wrong_option_count_is_rejected(tmp_path):
    runner = _make_runner(tmp_path, modal_k=6, calibration_questions=[_row(6, 0, "c")])
    with pytest.raises(ValueError, match="PriDe requires exactly 6 valid options"):
        runner.run_one(_row(4, 0, "bad_eval"), sample_index=0)
