"""Tests for sensitivity analyses (spec section 8)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from final_paper_analysis.sensitivity import (
    SensitivityError,
    api_vs_local_summary,
    included_only_vs_included_plus_qualified,
    macro_average,
    missing_prediction_policy_sensitivity,
    mmlu_duplicate_sensitivity,
    per_group_aggregate,
    strict_vs_conditional,
    with_and_without_repaired_questions,
)


def _frame(rows):
    return pd.DataFrame(rows)


def test_with_and_without_repaired_questions():
    rows = [
        {"question_id": "q1", "is_correct": True},
        {"question_id": "q2", "is_correct": False},  # repaired, wrong
        {"question_id": "q3", "is_correct": True},
    ]
    frame = _frame(rows)
    result = with_and_without_repaired_questions(frame, repaired_question_ids=["q2"])
    assert result.value_a == pytest.approx(2 / 3)  # with repaired q2 (wrong)
    assert result.value_b == pytest.approx(1.0)  # without q2, both remaining correct
    assert result.difference == pytest.approx(2 / 3 - 1.0)


def test_strict_vs_conditional():
    rows = [
        {"question_id": "q1", "is_correct": True},
        {"question_id": "q2", "is_correct": None},
    ]
    frame = _frame(rows)
    result = strict_vs_conditional(frame)
    assert result.value_a == pytest.approx(0.5)  # strict: 1/2
    assert result.value_b == pytest.approx(1.0)  # conditional: 1/1


def test_included_only_vs_included_plus_qualified():
    rows = [
        {"question_id": "q1", "is_correct": True, "status": "included"},
        {"question_id": "q2", "is_correct": False, "status": "qualified"},
        {"question_id": "q3", "is_correct": True, "status": "qualified"},
    ]
    frame = _frame(rows)
    result = included_only_vs_included_plus_qualified(frame)
    assert result.value_a == pytest.approx(1.0)  # included only: q1 correct
    assert result.value_b == pytest.approx(2 / 3)  # + qualified: 2/3 correct


def test_included_only_requires_status_column():
    with pytest.raises(SensitivityError, match="status"):
        included_only_vs_included_plus_qualified(_frame([{"question_id": "q1", "is_correct": True}]))


def test_mmlu_duplicate_sensitivity():
    rows = [
        {"question_id": "q1", "is_correct": True},
        {"question_id": "q2_dup", "is_correct": False},
        {"question_id": "q3", "is_correct": True},
    ]
    frame = _frame(rows)
    result = mmlu_duplicate_sensitivity(frame, duplicate_question_ids=["q2_dup"])
    assert result.value_a == pytest.approx(2 / 3)
    assert result.value_b == pytest.approx(1.0)


def test_api_vs_local_summary_kept_separate():
    rows = [
        {"question_id": "q1", "is_correct": True, "backend": "api"},
        {"question_id": "q2", "is_correct": False, "backend": "api"},
        {"question_id": "q3", "is_correct": True, "backend": "local"},
    ]
    frame = _frame(rows)
    result = api_vs_local_summary(frame)
    assert result["api"] == pytest.approx(0.5)
    assert result["local"] == pytest.approx(1.0)


def test_missing_prediction_policy_sensitivity_reports_all_policies():
    rows = [
        {"question_id": "q1", "is_correct": True},
        {"question_id": "q2", "is_correct": None},
    ]
    frame = _frame(rows)
    result = missing_prediction_policy_sensitivity(frame)
    assert set(result) == {"strict_missing_as_incorrect", "conditional_exclude_missing"}
    assert result["strict_missing_as_incorrect"] == pytest.approx(0.5)
    assert result["conditional_exclude_missing"] == pytest.approx(1.0)


def test_per_group_aggregate_and_macro_average_unweighted():
    rows = [
        {"question_id": "q1", "is_correct": True, "model_key": "model-a"},
        {"question_id": "q2", "is_correct": True, "model_key": "model-a"},
        {"question_id": "q3", "is_correct": False, "model_key": "model-a"},
        {"question_id": "q4", "is_correct": True, "model_key": "model-a"},
        # model-b: 1 question, wrong -- should NOT dominate the macro average
        # despite model-a having 4x as many rows.
        {"question_id": "q5", "is_correct": False, "model_key": "model-b"},
    ]
    frame = _frame(rows)

    def _strict(group):
        return (group["is_correct"] == True).sum() / len(group)  # noqa: E712

    per_model = per_group_aggregate(frame, group_column="model_key", metric=_strict)
    assert per_model["model-a"] == pytest.approx(0.75)
    assert per_model["model-b"] == pytest.approx(0.0)
    # unweighted macro average: (0.75 + 0.0) / 2 = 0.375, NOT the pooled
    # 3/5 = 0.6 that a row-weighted average would give.
    assert macro_average(per_model) == pytest.approx(0.375)


def test_macro_average_drops_nan_groups():
    assert macro_average({"a": 0.5, "b": float("nan")}) == pytest.approx(0.5)


def test_macro_average_all_nan_is_nan():
    assert math.isnan(macro_average({"a": float("nan")}))
