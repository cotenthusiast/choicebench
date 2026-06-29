# tests/benchmarks/test_truthful_qa.py

import pytest
import pandas as pd

from choicebench.benchmarks.truthful_qa import normalize_row, build_normalized_dataframe

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

# Correct answer is choices[0] → label index 0 → "A"
_ROW_VALID_A = {
    "question": "What causes the seasons?",
    "mc1_targets": {
        "choices": ["Earth's axial tilt", "Distance from sun", "Solar flares", "Moon gravity", "Meteor showers"],
        "labels": [1, 0, 0, 0, 0],
    },
    "mc2_targets": {"choices": [], "labels": []},
}

# Correct answer is choices[2] → label index 2 → "C"
_ROW_VALID_C = {
    "question": "What is the boiling point of water at sea level?",
    "mc1_targets": {
        "choices": ["50 °C", "75 °C", "100 °C", "125 °C"],
        "labels": [0, 0, 1, 0],
    },
    "mc2_targets": {"choices": [], "labels": []},
}

# Correct answer is choices[4] → must be skipped
_ROW_SKIP = {
    "question": "What is the speed of light?",
    "mc1_targets": {
        "choices": ["100 m/s", "1000 m/s", "10000 m/s", "100000 m/s", "3e8 m/s"],
        "labels": [0, 0, 0, 0, 1],
    },
    "mc2_targets": {"choices": [], "labels": []},
}


class TestNormalizeRow:
    def test_returns_all_schema_keys(self):
        result = normalize_row(_ROW_VALID_A)
        assert set(result.keys()) == _SCHEMA_COLS

    def test_subject_is_constant_truthful_qa(self):
        assert normalize_row(_ROW_VALID_A)["subject"] == "truthful_qa"
        assert normalize_row(_ROW_VALID_C)["subject"] == "truthful_qa"

    def test_question_text_preserved(self):
        assert normalize_row(_ROW_VALID_A)["question_text"] == _ROW_VALID_A["question"]

    def test_choices_are_first_four_mc1_choices(self):
        result = normalize_row(_ROW_VALID_A)
        assert result["choice_a"] == "Earth's axial tilt"
        assert result["choice_b"] == "Distance from sun"
        assert result["choice_c"] == "Solar flares"
        assert result["choice_d"] == "Moon gravity"

    def test_correct_option_derived_from_label(self):
        assert normalize_row(_ROW_VALID_A)["correct_option"] == "A"
        assert normalize_row(_ROW_VALID_C)["correct_option"] == "C"

    def test_correct_answer_text_matches_choice(self):
        result = normalize_row(_ROW_VALID_A)
        assert result["correct_answer_text"] == "Earth's axial tilt"

    @pytest.mark.parametrize("correct_idx,expected_option", [
        (0, "A"), (1, "B"), (2, "C"), (3, "D"),
    ])
    def test_all_valid_correct_indices(self, correct_idx, expected_option):
        labels = [0, 0, 0, 0]
        labels[correct_idx] = 1
        row = {
            "question": "Test?",
            "mc1_targets": {
                "choices": ["w", "x", "y", "z"],
                "labels": labels,
            },
            "mc2_targets": {"choices": [], "labels": []},
        }
        result = normalize_row(row)
        assert result["correct_option"] == expected_option

    def test_correct_answer_text_matches_selected_choice(self):
        for idx, field in [(0, "choice_a"), (1, "choice_b"), (2, "choice_c"), (3, "choice_d")]:
            labels = [0, 0, 0, 0]
            labels[idx] = 1
            row = {
                "question": "Test?",
                "mc1_targets": {"choices": ["alpha", "beta", "gamma", "delta"], "labels": labels},
                "mc2_targets": {"choices": [], "labels": []},
            }
            result = normalize_row(row)
            assert result["correct_answer_text"] == result[field]

    def test_question_id_is_16_char_hex_string(self):
        qid = normalize_row(_ROW_VALID_A)["question_id"]
        assert len(qid) == 16
        int(qid, 16)

    def test_question_id_is_deterministic(self):
        r1 = normalize_row(_ROW_VALID_A)
        r2 = normalize_row(_ROW_VALID_A)
        assert r1["question_id"] == r2["question_id"]

    def test_different_questions_produce_different_ids(self):
        r1 = normalize_row(_ROW_VALID_A)
        r2 = normalize_row(_ROW_VALID_C)
        assert r1["question_id"] != r2["question_id"]

    def test_skips_when_correct_index_greater_than_3(self):
        assert normalize_row(_ROW_SKIP) is None

    def test_skips_correct_index_4(self):
        labels = [0, 0, 0, 0, 1]
        row = {
            "question": "Skip me?",
            "mc1_targets": {"choices": ["a", "b", "c", "d", "e"], "labels": labels},
            "mc2_targets": {"choices": [], "labels": []},
        }
        assert normalize_row(row) is None


class TestBuildNormalizedDataframe:
    def test_returns_dataframe(self):
        df_raw = pd.DataFrame([_ROW_VALID_A])
        assert isinstance(build_normalized_dataframe(df_raw), pd.DataFrame)

    def test_output_has_all_schema_columns(self):
        df_raw = pd.DataFrame([_ROW_VALID_A])
        result = build_normalized_dataframe(df_raw)
        assert _SCHEMA_COLS.issubset(set(result.columns))

    def test_valid_rows_are_kept(self):
        df_raw = pd.DataFrame([_ROW_VALID_A, _ROW_VALID_C])
        result = build_normalized_dataframe(df_raw)
        assert len(result) == 2

    def test_skipped_rows_are_dropped(self):
        df_raw = pd.DataFrame([_ROW_VALID_A, _ROW_SKIP, _ROW_VALID_C])
        result = build_normalized_dataframe(df_raw)
        assert len(result) == 2

    def test_all_skipped_produces_empty(self):
        df_raw = pd.DataFrame([_ROW_SKIP])
        result = build_normalized_dataframe(df_raw)
        assert len(result) == 0

    def test_empty_input_produces_empty_output(self):
        df_raw = pd.DataFrame(columns=["question", "mc1_targets", "mc2_targets"])
        result = build_normalized_dataframe(df_raw)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_all_subjects_are_truthful_qa(self):
        df_raw = pd.DataFrame([_ROW_VALID_A, _ROW_VALID_C])
        result = build_normalized_dataframe(df_raw)
        assert (result["subject"] == "truthful_qa").all()

    def test_correct_option_is_always_abcd(self):
        df_raw = pd.DataFrame([_ROW_VALID_A, _ROW_VALID_C])
        result = build_normalized_dataframe(df_raw)
        assert result["correct_option"].isin(["A", "B", "C", "D"]).all()

    def test_question_ids_are_16_char_hex(self):
        df_raw = pd.DataFrame([_ROW_VALID_A, _ROW_VALID_C])
        result = build_normalized_dataframe(df_raw)
        for qid in result["question_id"]:
            assert len(qid) == 16
            int(qid, 16)

    def test_question_ids_are_unique(self):
        df_raw = pd.DataFrame([_ROW_VALID_A, _ROW_VALID_C])
        result = build_normalized_dataframe(df_raw)
        assert result["question_id"].nunique() == len(result)
