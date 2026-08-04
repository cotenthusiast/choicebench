# tests/benchmarks/test_mmlu_pro.py

import pandas as pd
import pytest

from choicebench.benchmarks.mmlu_pro import normalize_row
from choicebench.benchmarks.registry import BENCHMARK_REGISTRY
from choicebench.pipeline.options import build_option_map

build_normalized_dataframe = BENCHMARK_REGISTRY["mmlu_pro"].normalizer

_SCHEMA_COLS = {
    "question_id",
    "subject",
    "question_text",
    "choices_json",
    "correct_index",
    "correct_option",
    "correct_answer_text",
    "n_choices",
}


def _opts(row) -> dict:
    return build_option_map(row if isinstance(row, dict) else row.to_dict())


# 6 options; correct is options[2] → answer_index=2 → "C"
_ROW_VALID = {
    "question": "What is 2 + 2?",
    "options": ["1", "2", "4", "3", "5", "6"],
    "answer": "C",
    "answer_index": 2,
    "category": "math",
    "src": "test_src",
}

# 4 options; correct is options[0] → answer_index=0 → "A"
_ROW_VALID_A = {
    "question": "What color is the sky?",
    "options": ["blue", "red", "green", "yellow"],
    "answer": "A",
    "answer_index": 0,
    "category": "science",
    "src": "test_src",
}

# 6 options; correct is options[4] → answer_index=4 → "E" (previously dropped).
_ROW_CORRECT_BEYOND_D = {
    "question": "What is 3 + 3?",
    "options": ["1", "2", "4", "3", "6", "7"],
    "answer": "E",
    "answer_index": 4,
    "category": "math",
    "src": "test_src",
}

# A full 10-option MMLU-Pro-style question with the answer at index 9 ("J").
_ROW_TEN_OPTIONS = {
    "question": "Which is the ninth option?",
    "options": ["o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9"],
    "answer": "J",
    "answer_index": 9,
    "category": "logic",
    "src": "test_src",
}


class TestNormalizeRow:
    def test_returns_all_schema_keys(self):
        assert set(normalize_row(_ROW_VALID).keys()) == _SCHEMA_COLS

    def test_subject_is_category(self):
        assert normalize_row(_ROW_VALID)["subject"] == "math"

    def test_question_text_preserved(self):
        assert normalize_row(_ROW_VALID)["question_text"] == _ROW_VALID["question"]

    def test_all_options_preserved(self):
        opts = _opts(normalize_row(_ROW_VALID))
        assert opts == {"A": "1", "B": "2", "C": "4", "D": "3", "E": "5", "F": "6"}

    def test_n_choices_matches_option_count(self):
        assert normalize_row(_ROW_VALID)["n_choices"] == 6

    def test_correct_option_derived_from_answer_index(self):
        assert normalize_row(_ROW_VALID)["correct_option"] == "C"
        assert normalize_row(_ROW_VALID)["correct_index"] == 2

    def test_correct_answer_text_matches_option(self):
        assert normalize_row(_ROW_VALID)["correct_answer_text"] == "4"

    @pytest.mark.parametrize("answer_index,expected_option", [
        (0, "A"), (1, "B"), (2, "C"), (3, "D"), (4, "E"), (5, "F"),
    ])
    def test_all_valid_answer_indices(self, answer_index, expected_option):
        row = {**_ROW_VALID, "answer_index": answer_index}
        result = normalize_row(row)
        assert result["correct_option"] == expected_option

    def test_answer_beyond_d_is_preserved_not_skipped(self):
        result = normalize_row(_ROW_CORRECT_BEYOND_D)
        assert result is not None
        assert result["correct_option"] == "E"
        assert result["correct_answer_text"] == "6"
        assert result["n_choices"] == 6

    def test_ten_option_question_normalizes_with_full_choice_set(self):
        result = normalize_row(_ROW_TEN_OPTIONS)
        assert result["n_choices"] == 10
        assert result["correct_option"] == "J"
        assert result["correct_index"] == 9
        assert result["correct_answer_text"] == "o9"
        opts = _opts(result)
        assert len(opts) == 10
        assert opts["J"] == "o9"

    def test_question_id_is_16_char_hex_string(self):
        qid = normalize_row(_ROW_VALID)["question_id"]
        assert len(qid) == 16
        int(qid, 16)

    def test_different_questions_produce_different_ids(self):
        assert normalize_row(_ROW_VALID)["question_id"] != normalize_row(_ROW_VALID_A)["question_id"]


class TestBuildNormalizedDataframe:
    def test_returns_dataframe(self):
        df_raw = pd.DataFrame([_ROW_VALID])
        assert isinstance(build_normalized_dataframe(df_raw), pd.DataFrame)

    def test_output_has_all_schema_columns(self):
        df_raw = pd.DataFrame([_ROW_VALID])
        result = build_normalized_dataframe(df_raw)
        assert _SCHEMA_COLS.issubset(set(result.columns))

    def test_no_rows_are_dropped(self):
        # Every row survives — including one whose answer is beyond D, which the
        # old adapter would have silently skipped.
        df_raw = pd.DataFrame([_ROW_VALID, _ROW_CORRECT_BEYOND_D, _ROW_VALID_A, _ROW_TEN_OPTIONS])
        result = build_normalized_dataframe(df_raw)
        assert len(result) == 4

    def test_full_choice_sets_survive_normalization(self):
        df_raw = pd.DataFrame([_ROW_VALID, _ROW_TEN_OPTIONS])
        result = build_normalized_dataframe(df_raw)
        assert set(result["n_choices"]) == {6, 10}

    def test_empty_input_produces_empty_output(self):
        df_raw = pd.DataFrame(columns=["question", "options", "answer", "answer_index", "category", "src"])
        result = build_normalized_dataframe(df_raw)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_correct_option_can_exceed_d(self):
        df_raw = pd.DataFrame([_ROW_VALID, _ROW_CORRECT_BEYOND_D, _ROW_TEN_OPTIONS])
        result = build_normalized_dataframe(df_raw)
        assert set(result["correct_option"]) == {"C", "E", "J"}

    def test_question_ids_are_unique(self):
        df_raw = pd.DataFrame([_ROW_VALID, _ROW_VALID_A])
        result = build_normalized_dataframe(df_raw)
        assert result["question_id"].nunique() == len(result)
