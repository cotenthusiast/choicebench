"""Tests for the canonical row-level analysis table (spec section 3)."""

from __future__ import annotations

import pytest

from final_paper_analysis.canonical_table import (
    CanonicalRowSource,
    CanonicalTableError,
    build_canonical_table,
)


def _source(**overrides) -> CanonicalRowSource:
    defaults = dict(
        question_id="q1",
        model_key="gpt-4-1-mini",
        backend="api",
        benchmark_name="arc_challenge",
        method_key="baseline",
        condition_id="cond_1",
        realization_id="real_1",
        prediction_origin="external_historical_inference",
        derivation_origin="external_import",
        evidence_status="complete",
        scope_disposition="included",
        correct_option="B",
        predicted_option="B",
        choices_json='{"A": "a", "B": "b", "C": "c", "D": "d"}',
    )
    defaults.update(overrides)
    return CanonicalRowSource(**defaults)


def test_correct_prediction_row():
    frame = build_canonical_table([_source()])
    row = frame.iloc[0]
    assert bool(row["is_correct"]) is True
    assert bool(row["has_prediction"]) is True
    assert bool(row["parse_failed"]) is False
    assert row["status"] == "included"
    assert row["option_count"] == 4
    assert row["displayed_gold_position"] == "B"


def test_incorrect_prediction_row():
    frame = build_canonical_table([_source(predicted_option="C")])
    row = frame.iloc[0]
    assert bool(row["is_correct"]) is False


def test_missing_prediction_is_not_coerced_to_incorrect():
    frame = build_canonical_table([_source(predicted_option=None, parse_status="parse_missing")])
    row = frame.iloc[0]
    assert row["is_correct"] is None
    assert bool(row["has_prediction"]) is False
    assert bool(row["parse_failed"]) is True


def test_three_option_arc_question_has_option_count_three():
    frame = build_canonical_table(
        [_source(choices_json='{"A": "a", "B": "b", "C": "c", "D": ""}')]
    )
    assert frame.iloc[0]["option_count"] == 3


def test_excluded_pride_cell_status_is_excluded():
    frame = build_canonical_table([_source(scope_disposition="excluded_from_paper_matrix")])
    assert frame.iloc[0]["status"] == "excluded"


def test_held_cell_status_is_incomplete():
    frame = build_canonical_table([_source(scope_disposition="held", evidence_status="partial")])
    assert frame.iloc[0]["status"] == "incomplete"


def test_qualified_evidence_status_maps_to_qualified():
    frame = build_canonical_table([_source(evidence_status="qualified")])
    assert frame.iloc[0]["status"] == "qualified"


@pytest.mark.parametrize("evidence_status", ["partial", "malformed", "recoverable", "failed"])
def test_incomplete_evidence_statuses_map_to_incomplete(evidence_status):
    frame = build_canonical_table([_source(evidence_status=evidence_status)])
    assert frame.iloc[0]["status"] == "incomplete"


def test_local_and_api_same_model_name_are_not_collapsed():
    rows = [
        _source(model_key="Qwen/Qwen2.5-7B-Instruct", backend="local"),
        _source(model_key="Qwen/Qwen2.5-7B-Instruct-Turbo", backend="api"),
    ]
    frame = build_canonical_table(rows)
    assert len(frame) == 2
    assert set(frame["backend"]) == {"local", "api"}


def test_two_stage_versions_are_not_collapsed():
    rows = [
        _source(method_key="two_stage_v1", condition_id="cond_v1", realization_id="real_v1"),
        _source(method_key="two_stage_v2", condition_id="cond_v2", realization_id="real_v2"),
        _source(method_key="two_stage_v3", condition_id="cond_v3", realization_id="real_v3"),
    ]
    frame = build_canonical_table(rows)
    assert len(frame) == 3
    assert set(frame["method_key"]) == {"two_stage_v1", "two_stage_v2", "two_stage_v3"}


def test_duplicate_row_key_is_rejected():
    with pytest.raises(CanonicalTableError, match="Duplicate canonical row key"):
        build_canonical_table([_source(), _source()])


def test_parse_status_failure_forces_is_correct_none_even_with_a_stray_predicted_option():
    # Defensive: even if predicted_option is somehow non-null alongside a
    # failure-status parse_status, correctness must not be silently computed.
    frame = build_canonical_table(
        [_source(predicted_option="B", parse_status="score_unscorable")]
    )
    assert frame.iloc[0]["is_correct"] is None
    assert bool(frame.iloc[0]["parse_failed"]) is True


def test_fallback_used_is_passed_through():
    frame = build_canonical_table([_source(fallback_used=True)])
    assert bool(frame.iloc[0]["fallback_used"]) is True
