"""Tests for the 2x2 factorial decomposition (spec section 6)."""

from __future__ import annotations

import pandas as pd
import pytest

from final_paper_analysis.factorial import (
    FactorialError,
    compute_factorial_diagnostic_diffs,
    compute_factorial_effects,
    join_factorial_conditions,
    verify_shared_stage1_response,
)


def _role_frame(question_ids, correctness, *, has_prediction=None, parse_failed=None, fallback_used=None):
    n = len(question_ids)
    return pd.DataFrame({
        "question_id": question_ids,
        "is_correct": correctness,
        "has_prediction": has_prediction if has_prediction is not None else [True] * n,
        "parse_failed": parse_failed if parse_failed is not None else [False] * n,
        "fallback_used": fallback_used if fallback_used is not None else [None] * n,
    })


def test_join_factorial_conditions_rejects_missing_role():
    qids = ["q1", "q2"]
    cells = {
        "hidden_embedding": _role_frame(qids, [True, False]),
        "hidden_llm": _role_frame(qids, [True, True]),
        "visible_llm": _role_frame(qids, [False, False]),
    }
    with pytest.raises(FactorialError, match="Missing required factorial role"):
        join_factorial_conditions(cells)


def test_join_factorial_conditions_rejects_mismatched_question_sets():
    cells = {
        "hidden_embedding": _role_frame(["q1", "q2"], [True, False]),
        "visible_embedding": _role_frame(["q1", "q2"], [True, True]),
        "hidden_llm": _role_frame(["q1", "q2"], [True, True]),
        "visible_llm": _role_frame(["q1", "q3"], [True, True]),
    }
    with pytest.raises(FactorialError, match="identical question_id set"):
        join_factorial_conditions(cells)


def _known_4x4_cells():
    # 4 questions, deterministic correctness per role, chosen so every
    # effect is hand-computable:
    #   hidden_embedding: [T,T,F,F] -> acc 0.5
    #   visible_embedding:[T,T,T,F] -> acc 0.75
    #   hidden_llm:       [T,F,F,F] -> acc 0.25
    #   visible_llm:      [T,T,T,T] -> acc 1.0
    qids = ["q1", "q2", "q3", "q4"]
    cells = {
        "hidden_embedding": _role_frame(qids, [True, True, False, False]),
        "visible_embedding": _role_frame(qids, [True, True, True, False]),
        "hidden_llm": _role_frame(qids, [True, False, False, False]),
        "visible_llm": _role_frame(qids, [True, True, True, True]),
    }
    return cells


def test_compute_factorial_effects_matches_hand_computed_point_estimates():
    joined = join_factorial_conditions(_known_4x4_cells())
    result = compute_factorial_effects(joined, seed=1, n_resamples=200)

    assert result.accuracy_by_role["hidden_embedding"] == pytest.approx(0.5)
    assert result.accuracy_by_role["visible_embedding"] == pytest.approx(0.75)
    assert result.accuracy_by_role["hidden_llm"] == pytest.approx(0.25)
    assert result.accuracy_by_role["visible_llm"] == pytest.approx(1.0)

    # main_effect_visible = 0.5*((1.0-0.25) + (0.75-0.5)) = 0.5*(0.75+0.25) = 0.5
    assert result.main_effect_visible_options.point_estimate == pytest.approx(0.5)
    # main_effect_llm = 0.5*((0.25-0.5) + (1.0-0.75)) = 0.5*(-0.25+0.25) = 0.0
    assert result.main_effect_llm_matcher.point_estimate == pytest.approx(0.0)
    # interaction = (1.0-0.25) - (0.75-0.5) = 0.75 - 0.25 = 0.5
    assert result.interaction_effect.point_estimate == pytest.approx(0.5)
    assert result.n_questions == 4


def test_compute_factorial_effects_ci_contains_point_estimate():
    joined = join_factorial_conditions(_known_4x4_cells())
    result = compute_factorial_effects(joined, seed=7, n_resamples=1000)
    for contrast in (
        result.main_effect_visible_options, result.main_effect_llm_matcher, result.interaction_effect,
    ):
        assert contrast.lower <= contrast.point_estimate <= contrast.upper


def test_compute_factorial_effects_no_effect_case_is_all_zero():
    qids = ["q1", "q2", "q3", "q4"]
    identical = [True, False, True, False]
    cells = {role: _role_frame(qids, identical) for role in (
        "hidden_embedding", "visible_embedding", "hidden_llm", "visible_llm"
    )}
    joined = join_factorial_conditions(cells)
    result = compute_factorial_effects(joined, seed=1, n_resamples=200)
    assert result.main_effect_visible_options.point_estimate == pytest.approx(0.0)
    assert result.main_effect_llm_matcher.point_estimate == pytest.approx(0.0)
    assert result.interaction_effect.point_estimate == pytest.approx(0.0)


def test_compute_factorial_effects_is_deterministic_from_seed():
    joined = join_factorial_conditions(_known_4x4_cells())
    result1 = compute_factorial_effects(joined, seed=99, n_resamples=300)
    result2 = compute_factorial_effects(joined, seed=99, n_resamples=300)
    assert result1.main_effect_visible_options == result2.main_effect_visible_options
    assert result1.interaction_effect == result2.interaction_effect


def test_compute_factorial_effects_rejects_empty_join():
    empty = pd.DataFrame({f"{role}__correct": [] for role in (
        "hidden_embedding", "visible_embedding", "hidden_llm", "visible_llm"
    )})
    with pytest.raises(FactorialError):
        compute_factorial_effects(empty, seed=1)


def test_diagnostic_diffs_reports_coverage_and_parse_failure_per_role():
    qids = ["q1", "q2"]
    cells = {
        "hidden_embedding": _role_frame(qids, [True, False], has_prediction=[True, False], parse_failed=[False, True]),
        "visible_embedding": _role_frame(qids, [True, True]),
        "hidden_llm": _role_frame(qids, [True, True]),
        "visible_llm": _role_frame(qids, [True, True]),
    }
    joined = join_factorial_conditions(cells)
    diffs = compute_factorial_diagnostic_diffs(joined)
    assert diffs.coverage_by_role["hidden_embedding"] == pytest.approx(0.5)
    assert diffs.parse_failure_rate_by_role["hidden_embedding"] == pytest.approx(0.5)
    assert diffs.coverage_by_role["visible_llm"] == pytest.approx(1.0)


# --- Stage-1 response reuse verification ------------------------------------


def test_verify_shared_stage1_response_passes_when_identical():
    responses = {"q1": "the answer is A", "q2": "the answer is B"}
    verify_shared_stage1_response(responses, dict(responses))


def test_verify_shared_stage1_response_rejects_divergent_text():
    llm_responses = {"q1": "the answer is A"}
    embedding_responses = {"q1": "a completely different response"}
    with pytest.raises(FactorialError, match="do not share the exact same Stage-1"):
        verify_shared_stage1_response(llm_responses, embedding_responses)


def test_verify_shared_stage1_response_rejects_missing_question():
    llm_responses = {"q1": "x", "q2": "y"}
    embedding_responses = {"q1": "x"}
    with pytest.raises(FactorialError, match="response sets differ"):
        verify_shared_stage1_response(llm_responses, embedding_responses)
