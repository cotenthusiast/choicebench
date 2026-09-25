# tests/analysis/test_paired_tests.py

import pandas as pd
import pytest

from choicebench.analysis.paired_tests import mcnemar_exact_test


def _df(rows: list[tuple[str, bool | None]]) -> pd.DataFrame:
    return pd.DataFrame([{"question_id": qid, "is_correct": correct} for qid, correct in rows])


class TestMcnemarExactTest:
    def test_hand_checkable_p_value_8_vs_2_discordant(self):
        """8 questions where A is correct and B is not, 2 where B is
        correct and A is not -- exact two-sided binomial test of
        min(8,2)=2 successes in 10 trials at p=0.5. Hand-verified via
        scipy directly: binomtest(2, 10, 0.5).pvalue == 0.109375."""
        rows_a = [(f"q{i}", True) for i in range(8)] + [(f"q{i}", False) for i in range(8, 10)]
        rows_b = [(f"q{i}", False) for i in range(8)] + [(f"q{i}", True) for i in range(8, 10)]
        result = mcnemar_exact_test(_df(rows_a), _df(rows_b))
        assert result["b"] == 8
        assert result["c"] == 2
        assert result["n_discordant"] == 10
        assert result["p_value"] == pytest.approx(0.109375, abs=1e-9)

    def test_hand_checkable_p_value_4_vs_1_discordant(self):
        """binomtest(1, 5, 0.5).pvalue == 0.375, hand-verified via scipy."""
        rows_a = [(f"q{i}", True) for i in range(4)] + [("q4", False)]
        rows_b = [(f"q{i}", False) for i in range(4)] + [("q4", True)]
        result = mcnemar_exact_test(_df(rows_a), _df(rows_b))
        assert result["b"] == 4
        assert result["c"] == 1
        assert result["p_value"] == pytest.approx(0.375, abs=1e-9)

    def test_no_discordant_pairs_gives_p_value_one(self):
        rows = [("q1", True), ("q2", False), ("q3", True)]
        result = mcnemar_exact_test(_df(rows), _df(rows))
        assert result["n_discordant"] == 0
        assert result["p_value"] == 1.0

    def test_concordant_pairs_do_not_affect_the_test(self):
        """Questions where A and B agree (both correct or both incorrect)
        are irrelevant to McNemar -- only discordant pairs matter."""
        rows_a = [("q1", True), ("q2", True), ("q3", False), ("q4", True), ("q5", False)]
        rows_b = [("q1", True), ("q2", False), ("q3", False), ("q4", False), ("q5", True)]
        result = mcnemar_exact_test(_df(rows_a), _df(rows_b))
        # q1: both correct (concordant), q3: both incorrect (concordant)
        # q2: A correct, B incorrect -> b; q4: A correct, B incorrect -> b
        # q5: A incorrect, B correct -> c
        assert result["b"] == 2
        assert result["c"] == 1
        assert result["n_discordant"] == 3

    def test_question_present_only_in_a_is_excluded_and_counted(self):
        rows_a = [("q1", True), ("q2", False), ("qX", True)]
        rows_b = [("q1", True), ("q2", True)]
        result = mcnemar_exact_test(_df(rows_a), _df(rows_b))
        assert result["n_paired"] == 2
        assert result["n_a_only"] == 1
        assert result["n_b_only"] == 0

    def test_question_present_only_in_b_is_excluded_and_counted(self):
        rows_a = [("q1", True)]
        rows_b = [("q1", True), ("q2", False)]
        result = mcnemar_exact_test(_df(rows_a), _df(rows_b))
        assert result["n_paired"] == 1
        assert result["n_a_only"] == 0
        assert result["n_b_only"] == 1

    def test_unscorable_row_counts_as_incorrect_not_excluded(self):
        """None/NaN is_correct (unscorable) must count as incorrect,
        matching metrics.accuracy's own convention -- never silently
        excluded from the paired comparison."""
        rows_a = [("q1", None)]  # unscorable -> treated as incorrect
        rows_b = [("q1", True)]
        result = mcnemar_exact_test(_df(rows_a), _df(rows_b))
        assert result["n_paired"] == 1
        assert result["c"] == 1  # a incorrect, b correct
        assert result["b"] == 0

    def test_duplicate_question_id_in_a_raises(self):
        rows_a = [("q1", True), ("q1", False)]
        rows_b = [("q1", True)]
        with pytest.raises(ValueError, match="duplicate question_id"):
            mcnemar_exact_test(_df(rows_a), _df(rows_b))

    def test_duplicate_question_id_in_b_raises(self):
        rows_a = [("q1", True)]
        rows_b = [("q1", True), ("q1", False)]
        with pytest.raises(ValueError, match="duplicate question_id"):
            mcnemar_exact_test(_df(rows_a), _df(rows_b))
