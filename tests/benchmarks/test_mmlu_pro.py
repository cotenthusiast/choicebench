# tests/benchmarks/test_mmlu_pro.py

import pytest
import pandas as pd

from choicebench.benchmarks.mmlu_pro import normalize_row, build_normalized_dataframe

_SCHEMA_COLS = {
    "question_id",
    "subject",
    "question_text",
    "choice_a",
    "choice_b",
    "choice_c",
    "choice_d",
    "correct_option",
    "correct_answer_text",
}

# Correct answer is options[2] → answer_index=2 → "C"
_ROW_VALID = {
    "question": "What is 2 + 2?",
    "options": ["1", "2", "4", "3", "5", "6"],
    "answer": "C",
    "answer_index": 2,
    "category": "math",
    "src": "test_src",
}

# Correct answer is options[0] → answer_index=0 → "A"
_ROW_VALID_A = {
    "question": "What color is the sky?",
    "options": ["blue", "red", "green", "yellow"],
    "answer": "A",
    "answer_index": 0,
    "category": "science",
    "src": "test_src",
}

# Correct answer is options[4] → answer_index=4 → must be skipped
_ROW_SKIP = {
    "question": "What is 3 + 3?",
    "options": ["1", "2", "4", "3", "6", "7"],
    "answer": "E",
    "answer_index": 4,
    "category": "math",
    "src": "test_src",
}


class TestNormalizeRow:
    def test_returns_all_schema_keys(self):
        result = normalize_row(_ROW_VALID)
        assert set(result.keys()) == _SCHEMA_COLS

    def test_subject_is_category(self):
        assert normalize_row(_ROW_VALID)["subject"] == "math"

    def test_question_text_preserved(self):
        assert normalize_row(_ROW_VALID)["question_text"] == _ROW_VALID["question"]

    def test_choices_are_first_four_options(self):
        result = normalize_row(_ROW_VALID)
        assert result["choice_a"] == "1"
        assert result["choice_b"] == "2"
        assert result["choice_c"] == "4"
        assert result["choice_d"] == "3"

    def test_correct_option_derived_from_answer_index(self):
        assert normalize_row(_ROW_VALID)["correct_option"] == "C"

    def test_correct_answer_text_matches_option(self):
        result = normalize_row(_ROW_VALID)
        assert result["correct_answer_text"] == "4"

    @pytest.mark.parametrize("answer_index,expected_option", [
        (0, "A"), (1, "B"), (2, "C"), (3, "D"),
    ])
    def test_all_valid_answer_indices(self, answer_index, expected_option):
        row = {**_ROW_VALID, "answer_index": answer_index}
        result = normalize_row(row)
        assert result["correct_option"] == expected_option

    def test_correct_answer_text_matches_selected_choice(self):
        for idx, field in [(0, "choice_a"), (1, "choice_b"), (2, "choice_c"), (3, "choice_d")]:
            row = {**_ROW_VALID, "answer_index": idx}
            result = normalize_row(row)
            assert result["correct_answer_text"] == result[field]

    def test_question_id_is_16_char_hex_string(self):
        qid = normalize_row(_ROW_VALID)["question_id"]
        assert len(qid) == 16
        int(qid, 16)

    def test_question_id_is_deterministic(self):
        assert normalize_row(_ROW_VALID)["question_id"] == normalize_row(_ROW_VALID)["question_id"]

    def test_different_questions_produce_different_ids(self):
        r1 = normalize_row(_ROW_VALID)
        r2 = normalize_row(_ROW_VALID_A)
        assert r1["question_id"] != r2["question_id"]

    def test_skips_when_answer_index_greater_than_3(self):
        assert normalize_row(_ROW_SKIP) is None

    def test_skips_answer_index_4(self):
        row = {**_ROW_VALID, "answer_index": 4}
        assert normalize_row(row) is None

    def test_skips_answer_index_9(self):
        row = {**_ROW_VALID, "answer_index": 9}
        assert normalize_row(row) is None


class TestBuildNormalizedDataframe:
    def test_returns_dataframe(self):
        df_raw = pd.DataFrame([_ROW_VALID])
        assert isinstance(build_normalized_dataframe(df_raw), pd.DataFrame)

    def test_output_has_all_schema_columns(self):
        df_raw = pd.DataFrame([_ROW_VALID])
        result = build_normalized_dataframe(df_raw)
        assert _SCHEMA_COLS.issubset(set(result.columns))

    def test_valid_rows_are_kept(self):
        df_raw = pd.DataFrame([_ROW_VALID, _ROW_VALID_A])
        result = build_normalized_dataframe(df_raw)
        assert len(result) == 2

    def test_skipped_rows_are_dropped(self):
        df_raw = pd.DataFrame([_ROW_VALID, _ROW_SKIP, _ROW_VALID_A])
        result = build_normalized_dataframe(df_raw)
        assert len(result) == 2

    def test_all_skipped_produces_empty(self):
        df_raw = pd.DataFrame([_ROW_SKIP])
        result = build_normalized_dataframe(df_raw)
        assert len(result) == 0

    def test_empty_input_produces_empty_output(self):
        df_raw = pd.DataFrame(columns=["question", "options", "answer", "answer_index", "category", "src"])
        result = build_normalized_dataframe(df_raw)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_correct_option_is_always_abcd(self):
        df_raw = pd.DataFrame([_ROW_VALID, _ROW_VALID_A])
        result = build_normalized_dataframe(df_raw)
        assert result["correct_option"].isin(["A", "B", "C", "D"]).all()

    def test_question_ids_are_16_char_hex(self):
        df_raw = pd.DataFrame([_ROW_VALID, _ROW_VALID_A])
        result = build_normalized_dataframe(df_raw)
        for qid in result["question_id"]:
            assert len(qid) == 16
            int(qid, 16)

    def test_question_ids_are_unique(self):
        df_raw = pd.DataFrame([_ROW_VALID, _ROW_VALID_A])
        result = build_normalized_dataframe(df_raw)
        assert result["question_id"].nunique() == len(result)
