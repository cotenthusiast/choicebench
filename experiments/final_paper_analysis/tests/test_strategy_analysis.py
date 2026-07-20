"""Tests for mentor-driven strategy analysis (spec section 7)."""

from __future__ import annotations

import math

import pytest

from final_paper_analysis.strategy_analysis import (
    CellEffect,
    derive_available_strategies,
    detect_outliers,
    direction_of_effect_consistency,
    heterogeneity_by_group,
    macro_average_effect,
    win_tie_loss_counts,
)


def _effect(model, benchmark, strategy, baseline, strategy_metric, status="included"):
    return CellEffect(
        model_key=model, benchmark_name=benchmark, strategy=strategy,
        baseline_metric=baseline, strategy_metric=strategy_metric, status=status,
    )


def test_macro_average_is_unweighted_across_cells():
    effects = [
        _effect("model-a", "arc", "s1", 0.5, 0.6),  # effect +0.1
        _effect("model-b", "arc", "s1", 0.5, 0.4),  # effect -0.1
    ]
    result = macro_average_effect(effects)
    assert result.macro_average_effect == pytest.approx(0.0)
    assert result.n_usable_cells == 2


def test_macro_average_excludes_incomplete_and_excluded_cells():
    effects = [
        _effect("model-a", "arc", "s1", 0.5, 0.6),  # +0.1, usable
        _effect("model-b", "arc", "s1", 0.5, 0.99, status="incomplete"),  # excluded from average
        _effect("model-c", "arc", "s1", 0.5, 0.01, status="excluded"),  # excluded from average
    ]
    result = macro_average_effect(effects)
    assert result.macro_average_effect == pytest.approx(0.1)
    assert result.n_usable_cells == 1
    assert result.n_incomplete_cells == 1
    assert result.n_excluded_cells == 1


def test_macro_average_nan_when_no_usable_cells():
    effects = [_effect("model-a", "arc", "s1", 0.5, 0.6, status="incomplete")]
    result = macro_average_effect(effects)
    assert math.isnan(result.macro_average_effect)
    assert result.n_usable_cells == 0


def test_direction_of_effect_consistency():
    effects = [
        _effect("model-a", "arc", "s1", 0.5, 0.6),  # improved
        _effect("model-b", "arc", "s1", 0.5, 0.4),  # worsened
        _effect("model-c", "arc", "s1", 0.5, 0.5),  # unchanged
    ]
    result = direction_of_effect_consistency(effects)
    assert result.n_improved == 1
    assert result.n_worsened == 1
    assert result.n_unchanged == 1
    assert result.fraction_improved == pytest.approx(1 / 3)


def test_win_tie_loss_counts_matches_direction_consistency():
    effects = [
        _effect("model-a", "arc", "s1", 0.5, 0.6),
        _effect("model-b", "arc", "s1", 0.5, 0.6),
        _effect("model-c", "arc", "s1", 0.5, 0.4),
    ]
    result = win_tie_loss_counts(effects)
    assert result.wins == 2
    assert result.losses == 1
    assert result.ties == 0
    assert result.n_cells == 3


def test_heterogeneity_by_group_reveals_subgroup_driven_effect():
    effects = [
        _effect("model-a", "arc", "s1", 0.5, 0.9),  # huge improvement
        _effect("model-a", "mmlu", "s1", 0.5, 0.9),
        _effect("model-b", "arc", "s1", 0.5, 0.5),  # no effect
        _effect("model-b", "mmlu", "s1", 0.5, 0.5),
    ]
    by_model = heterogeneity_by_group(effects, group_key=lambda e: e.model_key)
    assert by_model["model-a"].macro_average_effect == pytest.approx(0.4)
    assert by_model["model-b"].macro_average_effect == pytest.approx(0.0)


def test_detect_outliers_flags_extreme_cell():
    effects = [
        _effect("model-a", "arc", "s1", 0.5, 0.51),
        _effect("model-b", "arc", "s1", 0.5, 0.49),
        _effect("model-c", "arc", "s1", 0.5, 0.52),
        _effect("model-d", "arc", "s1", 0.5, 0.98),  # extreme outlier
    ]
    report = detect_outliers(effects, z_threshold=1.0)
    assert len(report.outliers) == 1
    assert report.outliers[0].model_key == "model-d"


def test_detect_outliers_empty_with_fewer_than_two_usable_cells():
    effects = [_effect("model-a", "arc", "s1", 0.5, 0.6)]
    report = detect_outliers(effects)
    assert report.outliers == ()
    assert math.isnan(report.stdev_effect)


def test_detect_outliers_handles_zero_variance():
    effects = [
        _effect("model-a", "arc", "s1", 0.5, 0.6),
        _effect("model-b", "arc", "s1", 0.5, 0.6),
    ]
    report = detect_outliers(effects)
    assert report.outliers == ()
    assert report.stdev_effect == 0.0


def test_derive_available_strategies_only_counts_usable_cells():
    inventory = {
        ("model-a", "arc", "s1"): "included",
        ("model-b", "arc", "s1"): "incomplete",
        ("model-a", "mmlu", "s1"): "qualified",
        ("model-a", "arc", "s2"): "excluded",
    }
    available = derive_available_strategies(inventory)
    assert available["s1"] == {("model-a", "arc"), ("model-a", "mmlu")}
    assert "s2" not in available
