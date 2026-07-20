"""Tests for core cell metrics (spec section 4)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from final_paper_analysis.metrics import (
    MetricsError,
    accuracy,
    accuracy_by_gold_position,
    clopper_pearson_interval,
    coverage,
    compute_cell_metrics,
    conditional_accuracy,
    fallback_rate,
    parse_failure_rate,
    positional_mad,
    positional_max_min_gap,
    selection_entropy,
    selection_frequency_by_position,
    selection_jensen_shannon_divergence,
    strict_accuracy,
)


def _row(question_id, gold, predicted, *, is_correct=None, parse_failed=False,
         has_prediction=None, fallback_used=None):
    if has_prediction is None:
        has_prediction = predicted is not None
    if is_correct is None and has_prediction and not parse_failed:
        is_correct = predicted == gold
    return {
        "question_id": question_id, "gold_answer": gold, "predicted_option": predicted,
        "is_correct": is_correct, "has_prediction": has_prediction,
        "parse_failed": parse_failed, "fallback_used": fallback_used,
    }


def _frame(rows):
    return pd.DataFrame(rows)


# --- accuracy() ---


def test_accuracy_basic():
    assert accuracy(3, 4) == 0.75


def test_accuracy_zero_total_is_nan():
    assert math.isnan(accuracy(0, 0))


def test_accuracy_rejects_out_of_range_correct():
    with pytest.raises(MetricsError):
        accuracy(5, 4)


# --- Clopper-Pearson: boundary cases 0/n and n/n are explicitly required ---


def test_clopper_pearson_zero_of_n_lower_bound_is_zero():
    result = clopper_pearson_interval(0, 10)
    assert result.lower == 0.0
    assert result.point_estimate == 0.0
    assert 0.0 < result.upper < 1.0


def test_clopper_pearson_n_of_n_upper_bound_is_one():
    result = clopper_pearson_interval(10, 10)
    assert result.upper == 1.0
    assert result.point_estimate == 1.0
    assert 0.0 < result.lower < 1.0


def test_clopper_pearson_known_value_regression():
    # 95% CP interval for 5/10 is a well-known reference value.
    result = clopper_pearson_interval(5, 10)
    assert result.point_estimate == 0.5
    assert result.lower == pytest.approx(0.1871, abs=1e-3)
    assert result.upper == pytest.approx(0.8129, abs=1e-3)


def test_clopper_pearson_rejects_non_positive_total():
    with pytest.raises(MetricsError):
        clopper_pearson_interval(0, 0)


def test_clopper_pearson_rejects_out_of_range_correct():
    with pytest.raises(MetricsError):
        clopper_pearson_interval(11, 10)


# --- strict vs conditional accuracy: denominators must differ -------------


def test_strict_vs_conditional_accuracy_denominators_differ_with_missing_predictions():
    rows = [
        _row("q1", "A", "A"),
        _row("q2", "A", "B"),
        _row("q3", "A", None, has_prediction=False, parse_failed=True),
    ]
    frame = _frame(rows)
    # strict: 1 correct / 3 total = 0.333...
    assert strict_accuracy(frame) == pytest.approx(1 / 3)
    # conditional: 1 correct / 2 valid predictions = 0.5
    assert conditional_accuracy(frame) == pytest.approx(0.5)


def test_conditional_accuracy_is_nan_for_zero_coverage_cell():
    rows = [_row("q1", "A", None, has_prediction=False, parse_failed=True)]
    frame = _frame(rows)
    assert math.isnan(conditional_accuracy(frame))
    # strict accuracy is well-defined (0.0) even when conditional is NaN.
    assert strict_accuracy(frame) == 0.0


def test_all_failed_cell_has_zero_strict_accuracy_and_nan_conditional():
    rows = [_row(f"q{i}", "A", None, has_prediction=False, parse_failed=True) for i in range(5)]
    frame = _frame(rows)
    assert strict_accuracy(frame) == 0.0
    assert math.isnan(conditional_accuracy(frame))
    assert coverage(frame) == 0.0
    assert parse_failure_rate(frame) == 1.0


# --- coverage / parse-failure / fallback rates -----------------------------


def test_coverage_and_parse_failure_rate():
    rows = [
        _row("q1", "A", "A"),
        _row("q2", "A", None, has_prediction=False, parse_failed=True),
    ]
    frame = _frame(rows)
    assert coverage(frame) == 0.5
    assert parse_failure_rate(frame) == 0.5


def test_fallback_rate_nan_when_never_tracked():
    rows = [_row("q1", "A", "A"), _row("q2", "A", "B")]
    frame = _frame(rows)
    assert math.isnan(fallback_rate(frame))


def test_fallback_rate_denominator_excludes_untracked_rows():
    rows = [
        _row("q1", "A", "A", fallback_used=True),
        _row("q2", "A", "A", fallback_used=False),
        _row("q3", "A", "A", fallback_used=None),  # untracked, excluded from denominator
    ]
    frame = _frame(rows)
    assert fallback_rate(frame) == pytest.approx(0.5)


# --- selection frequency / positional accuracy / MAD -----------------------


def test_selection_frequency_by_position():
    rows = [
        _row("q1", "A", "A"), _row("q2", "B", "A"),
        _row("q3", "C", "B"), _row("q4", "D", "B"),
    ]
    frame = _frame(rows)
    freq = selection_frequency_by_position(frame)
    assert freq["A"] == pytest.approx(0.5)
    assert freq["B"] == pytest.approx(0.5)
    assert freq.get("C", 0.0) == 0.0
    assert freq.get("D", 0.0) == 0.0


def test_accuracy_by_gold_position_and_mad_uniform_case():
    # Every position has identical accuracy -> MAD should be 0.
    rows = [
        _row("q1", "A", "A"), _row("q2", "A", "B"),  # position A: 1/2 = 0.5
        _row("q3", "B", "B"), _row("q4", "B", "A"),  # position B: 1/2 = 0.5
    ]
    frame = _frame(rows)
    by_position = accuracy_by_gold_position(frame)
    assert by_position == {"A": 0.5, "B": 0.5}
    assert positional_mad(by_position) == pytest.approx(0.0)
    assert positional_max_min_gap(by_position) == pytest.approx(0.0)


def test_mad_detects_positional_bias():
    rows = [
        _row("q1", "A", "A"), _row("q2", "A", "A"),  # position A: 2/2 = 1.0
        _row("q3", "B", "A"), _row("q4", "B", "C"),  # position B: 0/2 = 0.0
    ]
    frame = _frame(rows)
    by_position = accuracy_by_gold_position(frame)
    assert by_position == {"A": 1.0, "B": 0.0}
    assert positional_mad(by_position) == pytest.approx(0.5)
    assert positional_max_min_gap(by_position) == pytest.approx(1.0)


def test_mad_is_nan_with_fewer_than_two_positions():
    assert math.isnan(positional_mad({"A": 0.5}))
    assert math.isnan(positional_mad({}))


# --- secondary metrics: entropy and Jensen-Shannon divergence --------------


def test_selection_entropy_uniform_four_way_is_two_bits():
    freq = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    assert selection_entropy(freq) == pytest.approx(2.0)


def test_selection_entropy_degenerate_single_choice_is_zero():
    freq = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
    assert selection_entropy(freq) == pytest.approx(0.0)


def test_jensen_shannon_divergence_uniform_selection_matches_uniform_reference():
    freq = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    jsd = selection_jensen_shannon_divergence(freq, option_count=4)
    assert jsd == pytest.approx(0.0, abs=1e-9)


def test_jensen_shannon_divergence_degenerate_selection_is_maximal_bounded():
    freq = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
    jsd = selection_jensen_shannon_divergence(freq, option_count=4)
    assert 0.0 < jsd <= 1.0


def test_jensen_shannon_divergence_respects_three_option_reference():
    # A perfectly uniform 3-way selection should show ~0 divergence from a
    # 3-option uniform reference, not the 4-option one.
    freq = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    jsd = selection_jensen_shannon_divergence(freq, option_count=3)
    assert jsd == pytest.approx(0.0, abs=1e-9)


def test_jensen_shannon_divergence_nan_with_no_selection_data():
    assert math.isnan(selection_jensen_shannon_divergence({}, option_count=4))


# --- compute_cell_metrics: full integration --------------------------------


def test_compute_cell_metrics_integration():
    rows = [
        _row("q1", "A", "A"), _row("q2", "A", "B"),
        _row("q3", "B", "B"), _row("q4", "B", None, has_prediction=False, parse_failed=True),
    ]
    frame = _frame(rows)
    result = compute_cell_metrics(frame, option_count=4)
    assert result.n_total == 4
    assert result.n_correct == 2
    assert result.strict_accuracy == pytest.approx(0.5)
    assert result.conditional_accuracy == pytest.approx(2 / 3)
    assert result.clopper_pearson.n_correct == 2
    assert result.clopper_pearson.n_total == 4
    assert result.coverage == pytest.approx(0.75)
    assert result.parse_failure_rate == pytest.approx(0.25)
    assert not math.isnan(result.positional_mad)


def test_compute_cell_metrics_rejects_empty_frame():
    with pytest.raises(MetricsError):
        compute_cell_metrics(pd.DataFrame(), option_count=4)
