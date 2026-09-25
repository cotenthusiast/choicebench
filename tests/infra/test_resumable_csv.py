# tests/infra/test_resumable_csv.py

import pandas as pd
import pytest

from choicebench.infra.resumable_csv import check_resume_compatible


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


class TestCheckResumeCompatibleNoExistingFile:
    def test_no_op_when_output_does_not_exist(self, tmp_path):
        check_resume_compatible(tmp_path / "missing.csv", exact_columns=["a", "b"])


class TestExactColumnsMode:
    """Used by scripts whose rows are always schema-uniform (the
    single-stage rotation scripts) -- any column difference means a stale
    or wrong-version file."""

    def test_passes_when_columns_match_exactly(self, tmp_path):
        path = tmp_path / "out.csv"
        _write(path, [{"question_id": "q1", "parsed_choice": "C"}])
        check_resume_compatible(path, exact_columns=["question_id", "parsed_choice"])

    def test_raises_when_a_column_is_missing(self, tmp_path):
        path = tmp_path / "out.csv"
        _write(path, [{"question_id": "q1"}])
        with pytest.raises(ValueError, match="schema"):
            check_resume_compatible(path, exact_columns=["question_id", "parsed_choice"])

    def test_raises_when_there_is_an_extra_column(self, tmp_path):
        path = tmp_path / "out.csv"
        _write(path, [{"question_id": "q1", "parsed_choice": "C", "extra": "x"}])
        with pytest.raises(ValueError, match="schema"):
            check_resume_compatible(path, exact_columns=["question_id", "parsed_choice"])

    def test_raises_when_column_order_differs(self, tmp_path):
        """DictWriter appends rows in fieldnames order -- a reordered
        header is just as unsafe to append into as a missing column."""
        path = tmp_path / "out.csv"
        _write(path, [{"parsed_choice": "C", "question_id": "q1"}])
        with pytest.raises(ValueError, match="schema"):
            check_resume_compatible(path, exact_columns=["question_id", "parsed_choice"])


class TestRequiredColumnsMode:
    """Used by scripts whose rows are legitimately heterogeneous across
    repetitions (stochasticity's reused observation-0 rows vs. freshly
    computed rows) -- only a MINIMUM set of columns is enforced, not an
    exact match."""

    def test_passes_when_required_columns_are_present_plus_extras(self, tmp_path):
        path = tmp_path / "out.csv"
        _write(path, [{"question_id": "q1", "repetition_index": 0, "method_name": "direct_mcq", "extra": "x"}])
        check_resume_compatible(path, required_columns=["question_id", "repetition_index"])

    def test_raises_when_a_required_column_is_missing(self, tmp_path):
        path = tmp_path / "out.csv"
        _write(path, [{"question_id": "q1"}])
        with pytest.raises(ValueError, match="schema"):
            check_resume_compatible(path, required_columns=["question_id", "repetition_index"])


class TestMethodNameIdentityCheck:
    def test_passes_when_all_rows_match_expected_method(self, tmp_path):
        path = tmp_path / "out.csv"
        _write(path, [
            {"question_id": "q1", "method_name": "direct_mcq"},
            {"question_id": "q2", "method_name": "direct_mcq"},
        ])
        check_resume_compatible(path, required_columns=["question_id"], expected_method_name="direct_mcq")

    def test_raises_when_a_row_has_a_different_method(self, tmp_path):
        """Refuses to mix runs -- this is the 'wrong file entirely'
        case Karl explicitly asked to guard against, distinct from the
        legitimate schema-heterogeneity case above."""
        path = tmp_path / "out.csv"
        _write(path, [
            {"question_id": "q1", "method_name": "direct_mcq"},
            {"question_id": "q2", "method_name": "two_stage"},
        ])
        with pytest.raises(ValueError, match="method_name"):
            check_resume_compatible(path, required_columns=["question_id"], expected_method_name="direct_mcq")

    def test_ignored_when_method_name_column_absent(self, tmp_path):
        path = tmp_path / "out.csv"
        _write(path, [{"question_id": "q1"}])
        check_resume_compatible(path, required_columns=["question_id"], expected_method_name="direct_mcq")

    def test_ignored_when_expected_method_name_not_given(self, tmp_path):
        path = tmp_path / "out.csv"
        _write(path, [{"question_id": "q1", "method_name": "anything"}])
        check_resume_compatible(path, required_columns=["question_id"])
