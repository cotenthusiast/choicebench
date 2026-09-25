# tests/analysis/test_agreement.py

import math

import pandas as pd
import pytest

from choicebench.analysis.agreement import compute_agreement_rate


def _row(question_id, repetition_index, parsed_choice):
    return {"question_id": question_id, "repetition_index": repetition_index, "parsed_choice": parsed_choice}


class TestComputeAgreementRate:
    def test_unanimous_group_counts_as_agreement(self):
        df = pd.DataFrame([
            _row("q1", 0, "C"), _row("q1", 1, "C"), _row("q1", 2, "C"), _row("q1", 3, "C"),
        ])
        out = compute_agreement_rate(df)
        assert out["agreement_rate"] == 1.0
        assert out["n_groups"] == 1

    def test_disagreeing_group_counts_as_not_agreement(self):
        df = pd.DataFrame([
            _row("q1", 0, "C"), _row("q1", 1, "C"), _row("q1", 2, "A"), _row("q1", 3, "C"),
        ])
        out = compute_agreement_rate(df)
        assert out["agreement_rate"] == 0.0

    def test_mixed_groups_computes_a_fraction(self):
        df = pd.DataFrame([
            _row("q1", 0, "C"), _row("q1", 1, "C"),  # agrees
            _row("q2", 0, "A"), _row("q2", 1, "B"),  # disagrees
        ])
        out = compute_agreement_rate(df)
        assert out["agreement_rate"] == 0.5
        assert out["n_groups"] == 2

    def test_a_group_with_a_single_row_is_excluded(self):
        """No comparison is possible with only one observation -- must not
        be silently counted as "agreement" (or disagreement)."""
        df = pd.DataFrame([
            _row("q1", 0, "C"), _row("q1", 1, "C"),  # 2 rows, agrees
            _row("q2", 0, "A"),  # only 1 row
        ])
        out = compute_agreement_rate(df)
        assert out["n_groups"] == 1
        assert out["agreement_rate"] == 1.0

    def test_unanimous_none_counts_as_agreement_but_is_flagged(self):
        """All four repetitions failing to parse an answer is technically
        "unanimous" (nunique==1), but it's not informative agreement about
        the model's actual answer -- callers must be able to tell these
        apart from genuine answer agreement."""
        df = pd.DataFrame([
            _row("q1", 0, None), _row("q1", 1, None), _row("q1", 2, None), _row("q1", 3, None),
        ])
        out = compute_agreement_rate(df)
        assert out["agreement_rate"] == 1.0
        assert out["n_fully_missing_groups"] == 1

    def test_a_none_among_real_answers_breaks_agreement(self):
        df = pd.DataFrame([
            _row("q1", 0, "C"), _row("q1", 1, "C"), _row("q1", 2, None), _row("q1", 3, "C"),
        ])
        out = compute_agreement_rate(df)
        assert out["agreement_rate"] == 0.0

    def test_empty_dataframe_returns_nan_not_a_crash(self):
        df = pd.DataFrame(columns=["question_id", "repetition_index", "parsed_choice"])
        out = compute_agreement_rate(df)
        assert math.isnan(out["agreement_rate"])
        assert out["n_groups"] == 0

    def test_custom_group_and_answer_columns(self):
        df = pd.DataFrame([
            {"qid": "q1", "rep": 0, "answer": "C"},
            {"qid": "q1", "rep": 1, "answer": "C"},
        ])
        out = compute_agreement_rate(df, group_by="qid", answer_col="answer")
        assert out["agreement_rate"] == 1.0


class TestRepetitionIndexCompleteness:
    """Confirmed conformance gap: a group missing a repetition_index, or
    with a duplicate one, was silently accepted as a valid group."""

    def test_missing_repetition_index_raises(self):
        df = pd.DataFrame([
            _row("q1", 0, "A"), _row("q1", 1, "A"), _row("q1", 3, "A"),  # missing index 2
        ])
        with pytest.raises(ValueError, match="repetition_index"):
            compute_agreement_rate(df)

    def test_duplicate_repetition_index_raises(self):
        df = pd.DataFrame([
            _row("q1", 0, "A"), _row("q1", 1, "A"), _row("q1", 2, "A"),
            _row("q1", 3, "A"), _row("q1", 3, "B"),  # 3 duplicated, 4 missing
        ])
        with pytest.raises(ValueError, match="repetition_index"):
            compute_agreement_rate(df)

    def test_complete_0_to_3_range_still_works(self):
        """Regression guard: the check itself must not reject a genuinely
        complete, duplicate-free group."""
        df = pd.DataFrame([
            _row("q1", 0, "A"), _row("q1", 3, "A"), _row("q1", 1, "A"), _row("q1", 2, "A"),
        ])
        out = compute_agreement_rate(df)
        assert out["agreement_rate"] == 1.0
        assert out["n_groups"] == 1

    def test_absent_repetition_index_column_is_unchecked(self):
        """Backward compatibility: a caller not using repetition_index at
        all (any non-stochasticity agreement use case) is unaffected."""
        df = pd.DataFrame([
            {"question_id": "q1", "parsed_choice": "A"},
            {"question_id": "q1", "parsed_choice": "A"},
        ])
        out = compute_agreement_rate(df)  # must not raise
        assert out["agreement_rate"] == 1.0

    def test_check_only_applies_to_counted_groups(self):
        """A single-row group is excluded before the repetition_index
        check runs -- its own repetition_index value (even if it "looks
        wrong" in isolation, e.g. index 2 alone) must not raise."""
        df = pd.DataFrame([
            _row("q1", 0, "A"), _row("q1", 1, "A"),  # valid 2-observation group
            _row("q2", 2, "A"),  # single row, excluded regardless of its index
        ])
        out = compute_agreement_rate(df)  # must not raise
        assert out["n_groups"] == 1
