"""Tests for the flip-rate analysis module (spec section 9).

Per methodology lock #12 and this session's own research memory, no real
per-permutation trace data exists yet -- every fixture here is synthetic,
standing in for the future verified trace schema.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from final_paper_analysis.flip_rate import (
    FlipRateError,
    PermutationTrace,
    compare_by_group,
    pearson_correlation,
    per_question_flip_stats,
    permutation_level_correctness,
    question_level_bootstrap_ci,
    traces_to_frame,
    validate_has_permutation_level_data,
)


def _trace(question_id, perm_index, letter, semantic, correct=None, model="gpt-4-1-mini", backend="api", benchmark="arc_challenge"):
    return PermutationTrace(
        question_id=question_id, model_key=model, backend=backend, benchmark_name=benchmark,
        permutation_index=perm_index, predicted_displayed_letter=letter,
        semantic_choice_id=semantic, is_correct=correct,
    )


# --- Fail-closed: reject aggregate-only input --------------------------------


def test_validate_rejects_missing_permutation_index_column():
    frame = pd.DataFrame({
        "question_id": ["q1"], "semantic_choice_id": ["A"], "predicted_displayed_letter": ["A"],
    })
    with pytest.raises(FlipRateError, match="permutation_index"):
        validate_has_permutation_level_data(frame)


def test_validate_rejects_single_row_per_question_as_aggregate_only():
    traces = [_trace("q1", 0, "A", "opt_A"), _trace("q2", 0, "B", "opt_B")]
    frame = traces_to_frame(traces)
    with pytest.raises(FlipRateError, match="aggregate majority output"):
        validate_has_permutation_level_data(frame)


def test_validate_passes_with_genuine_multi_permutation_data():
    traces = [_trace("q1", 0, "A", "opt_A"), _trace("q1", 1, "B", "opt_A")]
    frame = traces_to_frame(traces)
    validate_has_permutation_level_data(frame)  # must not raise


def test_per_question_flip_stats_fails_closed_on_aggregate_input():
    traces = [_trace("q1", 0, "A", "opt_A")]
    frame = traces_to_frame(traces)
    with pytest.raises(FlipRateError):
        per_question_flip_stats(frame)


# --- Semantic stability vs. letter-only cosmetic change ---------------------


def test_letter_only_flips_with_stable_semantic_answer():
    # Same underlying answer (opt_A) every permutation, but displayed at a
    # different letter each time -- semantic_flip_rate must be 0, while
    # displayed_letter_only_flip_rate reflects the cosmetic instability.
    traces = [
        _trace("q1", 0, "A", "opt_A"),
        _trace("q1", 1, "B", "opt_A"),
        _trace("q1", 2, "C", "opt_A"),
        _trace("q1", 3, "D", "opt_A"),
    ]
    frame = traces_to_frame(traces)
    (stats,) = per_question_flip_stats(frame)
    assert stats.semantic_flip_rate == pytest.approx(0.0)
    assert stats.majority_vote_stability == pytest.approx(1.0)
    assert stats.n_distinct_semantic_predictions == 1
    assert stats.displayed_letter_only_flip_rate == pytest.approx(3 / 4)


def test_letter_only_flip_rate_discriminates_split_evenness_not_just_distinct_count():
    # Regression: a naive (distinct_letters - 1) / n formula cannot tell a
    # lopsided 3:1 letter split from an even 2:2 split when both have the
    # same number of distinct letters (2) -- majority-share-based
    # (1 - max_share) correctly reports the even split as more unstable.
    lopsided = [
        _trace("q_lopsided", 0, "A", "opt_A"),
        _trace("q_lopsided", 1, "A", "opt_A"),
        _trace("q_lopsided", 2, "A", "opt_A"),
        _trace("q_lopsided", 3, "B", "opt_A"),
    ]
    even = [
        _trace("q_even", 0, "A", "opt_A"),
        _trace("q_even", 1, "A", "opt_A"),
        _trace("q_even", 2, "B", "opt_A"),
        _trace("q_even", 3, "B", "opt_A"),
    ]
    frame = traces_to_frame(lopsided + even)
    stats_by_question = {stat.question_id: stat for stat in per_question_flip_stats(frame)}

    lopsided_rate = stats_by_question["q_lopsided"].displayed_letter_only_flip_rate
    even_rate = stats_by_question["q_even"].displayed_letter_only_flip_rate

    assert lopsided_rate == pytest.approx(0.25)  # 1 - 3/4
    assert even_rate == pytest.approx(0.5)  # 1 - 2/4
    assert even_rate > lopsided_rate


def test_genuine_semantic_flip_across_permutations():
    traces = [
        _trace("q1", 0, "A", "opt_A"),
        _trace("q1", 1, "A", "opt_A"),
        _trace("q1", 2, "A", "opt_A"),
        _trace("q1", 3, "A", "opt_B"),  # genuine semantic flip
    ]
    frame = traces_to_frame(traces)
    (stats,) = per_question_flip_stats(frame)
    assert stats.n_distinct_semantic_predictions == 2
    assert stats.majority_semantic_choice == "opt_A"
    assert stats.majority_vote_stability == pytest.approx(0.75)
    assert stats.semantic_flip_rate == pytest.approx(0.25)


def test_majority_is_correct_reflects_majority_choice_correctness():
    traces = [
        _trace("q1", 0, "A", "opt_A", correct=True),
        _trace("q1", 1, "B", "opt_A", correct=True),
        _trace("q1", 2, "C", "opt_B", correct=False),
    ]
    frame = traces_to_frame(traces)
    (stats,) = per_question_flip_stats(frame)
    assert stats.majority_semantic_choice == "opt_A"
    assert stats.majority_is_correct is True


def test_unscorable_permutations_excluded_from_semantic_stats_but_counted_in_total():
    traces = [
        _trace("q1", 0, "A", "opt_A"),
        _trace("q1", 1, None, None),  # unparseable this permutation
    ]
    frame = traces_to_frame(traces)
    (stats,) = per_question_flip_stats(frame)
    assert stats.n_permutations == 2
    assert stats.n_distinct_semantic_predictions == 1
    assert stats.majority_vote_stability == pytest.approx(1.0)


def test_all_permutations_unscorable_gives_nan_stats():
    traces = [_trace("q1", 0, None, None), _trace("q1", 1, None, None)]
    frame = traces_to_frame(traces)
    (stats,) = per_question_flip_stats(frame)
    assert stats.n_distinct_semantic_predictions == 0
    assert math.isnan(stats.semantic_flip_rate)
    assert stats.majority_is_correct is None


# --- permutation-level correctness -------------------------------------------


def test_permutation_level_correctness():
    traces = [
        _trace("q1", 0, "A", "opt_A", correct=True),
        _trace("q1", 1, "A", "opt_A", correct=True),
        _trace("q1", 2, "B", "opt_B", correct=False),
    ]
    frame = traces_to_frame(traces)
    assert permutation_level_correctness(frame) == pytest.approx(2 / 3)


def test_permutation_level_correctness_nan_with_no_scored_rows():
    traces = [_trace("q1", 0, "A", "opt_A", correct=None)]
    frame = traces_to_frame(traces)
    assert math.isnan(permutation_level_correctness(frame))


# --- correlation between flip rate and other metrics -------------------------


def test_pearson_correlation_perfect_positive():
    assert pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_pearson_correlation_perfect_negative():
    assert pearson_correlation([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)


def test_pearson_correlation_nan_with_zero_variance():
    assert math.isnan(pearson_correlation([1, 1, 1], [1, 2, 3]))


def test_pearson_correlation_rejects_mismatched_lengths():
    with pytest.raises(FlipRateError):
        pearson_correlation([1, 2], [1, 2, 3])


# --- deterministic question-level bootstrap ----------------------------------


def test_bootstrap_is_deterministic():
    values = [0.1, 0.2, 0.3, 0.0, 0.5]
    r1 = question_level_bootstrap_ci(values, seed=3, n_resamples=500)
    r2 = question_level_bootstrap_ci(values, seed=3, n_resamples=500)
    assert r1 == r2


def test_bootstrap_drops_nan_values():
    values = [0.1, float("nan"), 0.3]
    result = question_level_bootstrap_ci(values, seed=1, n_resamples=200)
    assert result.point_estimate == pytest.approx(0.2)


def test_bootstrap_rejects_all_nan():
    with pytest.raises(FlipRateError):
        question_level_bootstrap_ci([float("nan"), float("nan")], seed=1)


# --- API/local and ARC/MMLU comparisons --------------------------------------


def test_compare_by_group_api_vs_local():
    traces_api = [
        _trace("q1", 0, "A", "opt_A"), _trace("q1", 1, "A", "opt_A"),
        _trace("q1", 2, "B", "opt_B"),
    ]
    traces_local = [
        _trace("q2", 0, "A", "opt_A", backend="local"),
        _trace("q2", 1, "A", "opt_A", backend="local"),
    ]
    frame = traces_to_frame(traces_api + traces_local)
    stats = per_question_flip_stats(frame)
    group_of = {"q1": "api", "q2": "local"}
    result = compare_by_group(stats, group_of=group_of)
    assert result["local"] == pytest.approx(0.0)
    assert result["api"] == pytest.approx(1 / 3)


def test_compare_by_group_omits_groups_with_no_valid_data():
    traces = [_trace("q1", 0, None, None), _trace("q1", 1, None, None)]
    frame = traces_to_frame(traces)
    stats = per_question_flip_stats(frame)
    result = compare_by_group(stats, group_of={"q1": "arc_challenge"})
    assert result == {}
