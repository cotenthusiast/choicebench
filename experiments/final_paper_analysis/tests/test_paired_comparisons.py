"""Tests for paired method comparisons (spec section 5)."""

from __future__ import annotations

import pandas as pd
import pytest

from final_paper_analysis.paired_comparisons import (
    PairedComparisonError,
    holm_correction,
    mcnemar_test,
    paired_accuracy_difference,
    paired_bootstrap_ci,
    paired_correctness_table,
    win_loss_tie_counts,
)


def _cell(correctness: dict[str, bool | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {"question_id": list(correctness), "is_correct": list(correctness.values())}
    )


def test_paired_correctness_table_rejects_mismatched_question_sets():
    a = _cell({"q1": True, "q2": False})
    b = _cell({"q1": True, "q3": False})
    with pytest.raises(PairedComparisonError, match="identical question sets"):
        paired_correctness_table(a, b)


def test_paired_correctness_table_coerces_missing_to_false():
    a = _cell({"q1": True, "q2": None})
    b = _cell({"q1": False, "q2": True})
    joined = paired_correctness_table(a, b)
    row_q2 = joined[joined["question_id"] == "q2"].iloc[0]
    assert bool(row_q2["correct_a"]) is False


def test_paired_accuracy_difference():
    a = _cell({"q1": True, "q2": True, "q3": False, "q4": False})
    b = _cell({"q1": True, "q2": False, "q3": False, "q4": False})
    joined = paired_correctness_table(a, b)
    # A: 2/4 = 0.5, B: 1/4 = 0.25 -> diff = 0.25
    assert paired_accuracy_difference(joined) == pytest.approx(0.25)


def test_win_loss_tie_counts():
    a = _cell({"q1": True, "q2": False, "q3": True, "q4": False})
    b = _cell({"q1": False, "q2": True, "q3": True, "q4": False})
    joined = paired_correctness_table(a, b)
    counts = win_loss_tie_counts(joined)
    # q1: A wins, q2: A loses, q3: tie (both correct), q4: tie (both wrong)
    assert counts.wins == 1
    assert counts.losses == 1
    assert counts.ties == 2
    assert counts.n_questions == 4


# --- McNemar exact test -----------------------------------------------------


def test_mcnemar_zero_discordant_pairs_gives_p_value_one():
    a = _cell({"q1": True, "q2": False})
    b = _cell({"q1": True, "q2": False})
    joined = paired_correctness_table(a, b)
    result = mcnemar_test(joined)
    assert result.statistic_b == 0
    assert result.statistic_c == 0
    assert result.p_value == 1.0


def test_mcnemar_symmetric_discordant_pairs_gives_p_value_one():
    # 3 wins, 3 losses -> perfectly symmetric, p should be 1.0.
    a = _cell({f"q{i}": (i < 3) for i in range(6)})
    b = _cell({f"q{i}": (i >= 3) for i in range(6)})
    joined = paired_correctness_table(a, b)
    result = mcnemar_test(joined)
    assert result.statistic_b == 3
    assert result.statistic_c == 3
    assert result.p_value == pytest.approx(1.0)


def test_mcnemar_matches_manual_exact_binomial_reference():
    # b=1, c=9 discordant pairs: exact two-sided p-value from a binomial
    # sign test, computed independently via scipy for cross-checking.
    from scipy.stats import binomtest

    expected = binomtest(1, 10, 0.5, alternative="two-sided").pvalue
    correctness_a = {}
    correctness_b = {}
    # 1 case where A wins (A correct, B incorrect)
    correctness_a["w0"] = True
    correctness_b["w0"] = False
    # 9 cases where B wins (A incorrect, B correct)
    for i in range(9):
        correctness_a[f"l{i}"] = False
        correctness_b[f"l{i}"] = True
    a = _cell(correctness_a)
    b = _cell(correctness_b)
    joined = paired_correctness_table(a, b)
    result = mcnemar_test(joined)
    assert result.statistic_b == 1
    assert result.statistic_c == 9
    assert result.p_value == pytest.approx(expected)
    assert result.p_value < 0.05


# --- Deterministic paired bootstrap -----------------------------------------


def test_bootstrap_is_deterministic_from_seed():
    a = _cell({f"q{i}": (i % 2 == 0) for i in range(20)})
    b = _cell({f"q{i}": (i % 3 == 0) for i in range(20)})
    joined = paired_correctness_table(a, b)
    result1 = paired_bootstrap_ci(joined, seed=42, n_resamples=500)
    result2 = paired_bootstrap_ci(joined, seed=42, n_resamples=500)
    assert result1.replicate_diffs == result2.replicate_diffs
    assert result1.lower == result2.lower
    assert result1.upper == result2.upper


def test_bootstrap_different_seeds_can_differ():
    a = _cell({f"q{i}": (i % 2 == 0) for i in range(20)})
    b = _cell({f"q{i}": (i % 3 == 0) for i in range(20)})
    joined = paired_correctness_table(a, b)
    result1 = paired_bootstrap_ci(joined, seed=1, n_resamples=200)
    result2 = paired_bootstrap_ci(joined, seed=2, n_resamples=200)
    assert result1.replicate_diffs != result2.replicate_diffs


def test_bootstrap_ci_contains_point_estimate_for_a_clear_winner():
    # A always correct, B always wrong -> point estimate 1.0, degenerate CI.
    a = _cell({f"q{i}": True for i in range(30)})
    b = _cell({f"q{i}": False for i in range(30)})
    joined = paired_correctness_table(a, b)
    result = paired_bootstrap_ci(joined, seed=7, n_resamples=1000)
    assert result.point_estimate == pytest.approx(1.0)
    assert result.lower == pytest.approx(1.0)
    assert result.upper == pytest.approx(1.0)


def test_bootstrap_rejects_empty_input():
    empty = pd.DataFrame({"question_id": [], "correct_a": [], "correct_b": []})
    with pytest.raises(PairedComparisonError):
        paired_bootstrap_ci(empty, seed=1)


# --- Holm correction ---------------------------------------------------------


def test_holm_correction_matches_hand_computed_reference():
    # p = [0.01, 0.04, 0.03, 0.02] (original order), m=4.
    # sorted ascending by value: 0.01(idx0), 0.02(idx3), 0.03(idx2), 0.04(idx1)
    # step multipliers (m-rank): 4,3,2,1 -> 0.04, 0.06, 0.06, 0.04
    # running max (monotone): 0.04, 0.06, 0.06, 0.06
    # mapped back to original order [idx0, idx1, idx2, idx3]:
    #   idx0 -> 0.04, idx1 -> 0.06, idx2 -> 0.06, idx3 -> 0.06
    p_values = [0.01, 0.04, 0.03, 0.02]
    adjusted = holm_correction(p_values)
    assert adjusted == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_correction_caps_at_one():
    p_values = [0.9, 0.8, 0.7]
    adjusted = holm_correction(p_values)
    assert all(value <= 1.0 for value in adjusted)


def test_holm_correction_single_p_value_is_unchanged():
    assert holm_correction([0.03]) == pytest.approx([0.03])


def test_holm_correction_empty_list():
    assert holm_correction([]) == []


def test_holm_correction_is_monotonic_in_sorted_order():
    p_values = [0.5, 0.001, 0.2, 0.049]
    adjusted = holm_correction(p_values)
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    sorted_adjusted = [adjusted[i] for i in order]
    assert sorted_adjusted == sorted(sorted_adjusted)
