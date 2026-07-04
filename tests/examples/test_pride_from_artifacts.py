# tests/examples/test_pride_from_artifacts.py
#
# examples/pride_from_artifacts.py recomputes PriDe entirely from completed
# direct_logprob/cyclic_logprob result CSVs. This fixture is a small,
# hand-computable 5-question, 2-option (A/B) scenario — the pride_math
# equations are option-count-agnostic, so 2 options keeps the arithmetic
# checkable while still exercising every code path (calibration via Eq. 1,
# transfer via Eq. 8, prior averaging via Eq. 7).
#
# Fixture design (see task write-up for the full derivation, cross-checked
# against the real pride_math functions before being hardcoded here):
#   - q4, q5: identical cyclic rollout matrices [[0.9,0.1],[0.9,0.1]] -> each
#     gives an exact Eq. 7 prior of [0.9, 0.1] (softmax(log(p)) == p when p
#     already sums to 1), so the averaged global prior is [0.9, 0.1] exactly.
#     Eq. 1 on that same matrix ties at [0.5, 0.5] -> argmax breaks to index 0
#     ("A") for both.
#   - q1, q2, q3: direct_logprob distributions chosen so dividing by the
#     [0.9, 0.1] prior (Eq. 8) flips q1's naive argmax (A -> B) while leaving
#     q2 and q3 unchanged.
#   - With seed=0, alpha=0.4 (N=5 -> K=2), sample_calibration_ids selects
#     indices [3, 4] of the sorted question_ids ["q1".."q5"], i.e. {q4, q5} —
#     confirmed directly against numpy's RNG, not assumed.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "pride_from_artifacts_under_test",
        _REPO_ROOT / "examples" / "pride_from_artifacts.py",
    )
    module = importlib.util.module_from_spec(spec)
    # dataclass(slots=True) needs its defining module registered in
    # sys.modules (it looks itself up there) — must happen before exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

def _direct_row(qid: str, correct_option: str, dist: list[float], answer_status: str = "success") -> dict:
    return {
        "question_id": qid,
        "correct_option": correct_option,
        "answer_status": answer_status,
        "option_distributions_json": json.dumps([dist]) if answer_status == "success" else None,
    }


def _cyclic_row(qid: str, correct_option: str, matrix: list[list[float]], answer_status: str = "success") -> dict:
    return {
        "question_id": qid,
        "correct_option": correct_option,
        "answer_status": answer_status,
        "option_distributions_json": json.dumps(matrix) if answer_status == "success" else None,
    }


def _direct_df() -> pd.DataFrame:
    rows = [
        _direct_row("q1", "B", [0.6, 0.4]),
        _direct_row("q2", "A", [0.99, 0.01]),
        _direct_row("q3", "B", [0.2, 0.8]),
        _direct_row("q4", "A", [0.7, 0.3]),
        _direct_row("q5", "B", [0.3, 0.7]),
    ]
    return pd.DataFrame(rows).set_index("question_id", drop=False)


def _cyclic_df() -> pd.DataFrame:
    uniform = [[0.5, 0.5], [0.5, 0.5]]
    skewed = [[0.9, 0.1], [0.9, 0.1]]
    rows = [
        _cyclic_row("q1", "B", uniform),
        _cyclic_row("q2", "A", uniform),
        _cyclic_row("q3", "B", uniform),
        _cyclic_row("q4", "A", skewed),
        _cyclic_row("q5", "B", skewed),
    ]
    return pd.DataFrame(rows).set_index("question_id", drop=False)


# ---------------------------------------------------------------------------
# build_question_records
# ---------------------------------------------------------------------------

def test_build_question_records_full_fixture():
    records = mod.build_question_records(_direct_df(), _cyclic_df())

    assert sorted(records) == ["q1", "q2", "q3", "q4", "q5"]
    assert records["q4"].letters == ("A", "B")
    np.testing.assert_allclose(records["q4"].eq7_prior, [0.9, 0.1])
    np.testing.assert_allclose(records["q4"].eq1_probs, [0.5, 0.5])
    np.testing.assert_allclose(records["q1"].default_probs, [0.6, 0.4])


def test_build_question_records_intersection_policy_drops_unmatched_and_failed():
    direct = pd.DataFrame([
        _direct_row("q1", "A", [0.6, 0.4]),
        _direct_row("q2", "A", [0.6, 0.4]),  # only in direct
        _direct_row("q3", "A", [0.6, 0.4]),  # cyclic side failed
    ]).set_index("question_id", drop=False)
    cyclic = pd.DataFrame([
        _cyclic_row("q1", "A", [[0.5, 0.5], [0.5, 0.5]]),
        _cyclic_row("q3", "A", [[0.5, 0.5], [0.5, 0.5]], answer_status="failure"),
        _cyclic_row("q4", "A", [[0.5, 0.5], [0.5, 0.5]]),  # only in cyclic
    ]).set_index("question_id", drop=False)

    records = mod.build_question_records(direct, cyclic)

    assert set(records) == {"q1"}


def test_build_question_records_correct_option_mismatch_raises():
    direct = pd.DataFrame([_direct_row("q1", "A", [0.6, 0.4])]).set_index("question_id", drop=False)
    cyclic = pd.DataFrame(
        [_cyclic_row("q1", "B", [[0.5, 0.5], [0.5, 0.5]])]
    ).set_index("question_id", drop=False)

    with pytest.raises(ValueError, match="mismatched correct_option"):
        mod.build_question_records(direct, cyclic)


def test_build_question_records_mixed_option_counts_raises():
    direct = pd.DataFrame([
        _direct_row("q1", "A", [0.6, 0.4]),
        _direct_row("q2", "A", [0.5, 0.3, 0.2]),
    ]).set_index("question_id", drop=False)
    cyclic = pd.DataFrame([
        _cyclic_row("q1", "A", [[0.5, 0.5], [0.5, 0.5]]),
        _cyclic_row("q2", "A", [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]]),
    ]).set_index("question_id", drop=False)

    with pytest.raises(ValueError, match="mixed option counts"):
        mod.build_question_records(direct, cyclic)


# ---------------------------------------------------------------------------
# sample_calibration_ids
# ---------------------------------------------------------------------------

def test_sample_calibration_ids_selects_expected_indices_for_seed_0():
    full_ids = ["q1", "q2", "q3", "q4", "q5"]
    # floor(0.4 * 5) == 2; verified directly against numpy's RNG (see module
    # docstring at top of this file) that seed=0 picks positions [3, 4].
    assert mod.sample_calibration_ids(full_ids, alpha=0.4, seed=0) == ["q4", "q5"]


def test_sample_calibration_ids_k_zero_hard_errors():
    full_ids = ["q1", "q2", "q3", "q4", "q5"]
    with pytest.raises(ValueError, match="K=0"):
        mod.sample_calibration_ids(full_ids, alpha=0.01, seed=0)


# ---------------------------------------------------------------------------
# compute_pride_cell — exact predictions and metrics for a fixed (alpha, seed)
# ---------------------------------------------------------------------------

def test_compute_pride_cell_exact_predictions_and_metrics():
    records = mod.build_question_records(_direct_df(), _cyclic_df())
    full_ids = sorted(records)

    predictions, summary = mod.compute_pride_cell(records, full_ids, alpha=0.4, seed=0)

    assert predictions == {"q1": "B", "q2": "A", "q3": "B", "q4": "A", "q5": "A"}
    assert summary["n"] == 5
    assert summary["k"] == 2
    assert summary["accuracy"] == pytest.approx(0.8)
    assert summary["rstd"] == pytest.approx(16.66666666666667)
    assert summary["mad"] == pytest.approx(20.0)


def test_compute_pride_cell_is_deterministic():
    records = mod.build_question_records(_direct_df(), _cyclic_df())
    full_ids = sorted(records)

    predictions_a, summary_a = mod.compute_pride_cell(records, full_ids, alpha=0.4, seed=0)
    predictions_b, summary_b = mod.compute_pride_cell(records, full_ids, alpha=0.4, seed=0)

    assert predictions_a == predictions_b
    assert summary_a == summary_b


# ---------------------------------------------------------------------------
# End-to-end run() over on-disk CSVs
# ---------------------------------------------------------------------------

def _write_csv(tmp_path: Path, method_name: str, df: pd.DataFrame) -> None:
    path = tmp_path / f"testrun_{method_name}_dummymodel_toy.csv"
    df.to_csv(path, index=False)


def test_run_end_to_end_over_csvs(tmp_path: Path):
    _write_csv(tmp_path, "direct_logprob", _direct_df())
    _write_csv(tmp_path, "cyclic_logprob", _cyclic_df())

    targets_path = tmp_path / "targets.json"
    targets_path.write_text(json.dumps({
        "targets": {
            "default": {"rstd": 0.0, "acc": 100.0},
            "cyclic_perm": {"rstd": 0.0, "acc": 100.0},
            "pride_40": {"rstd": 0.0, "acc": 100.0},
        }
    }))

    output = mod.run(
        run_id="testrun",
        alphas=[0.4],
        seeds=[0],
        targets_path=targets_path,
        run_dir=tmp_path,
    )

    assert output["run_id"] == "testrun"
    assert output["n_scored"] == 5
    cell = output["cells"]["alpha=0.4/seed=0"]
    assert cell["n"] == 5
    assert cell["k"] == 2
    assert cell["accuracy"] == pytest.approx(0.8)

    agg = output["per_alpha"]["0.4"]
    assert agg["acc_mean"] == pytest.approx(80.0)

    comparison = output["comparison"]
    assert "default" in comparison
    assert "cyclic_perm" in comparison
    assert "pride_40" in comparison
    assert comparison["pride_40"]["delta"]["acc"] == pytest.approx(80.0 - 100.0)


def test_run_raises_when_no_csvs_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="direct_logprob"):
        mod.run(
            run_id="missing_run",
            alphas=[0.4],
            seeds=[0],
            targets_path=tmp_path / "targets.json",
            run_dir=tmp_path,
        )


def test_load_method_frame_duplicate_question_id_raises(tmp_path: Path):
    dup_df = pd.DataFrame([
        _direct_row("q1", "A", [0.6, 0.4]),
        _direct_row("q1", "A", [0.5, 0.5]),
    ])
    _write_csv(tmp_path, "direct_logprob", dup_df)

    with pytest.raises(ValueError, match="duplicate question_id"):
        mod._load_method_frame(tmp_path, "direct_logprob")
