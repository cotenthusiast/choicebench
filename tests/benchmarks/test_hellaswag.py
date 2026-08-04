# tests/benchmarks/test_hellaswag.py

import pytest
import pandas as pd

from choicebench.benchmarks.hellaswag import normalize_row
from choicebench.benchmarks.registry import BENCHMARK_REGISTRY
from choicebench.pipeline.options import build_option_map

build_normalized_dataframe = BENCHMARK_REGISTRY["hellaswag"].normalizer

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

_ROW_LABEL_1 = {
    "ind": "1",
    "activity_label": "Cooking",
    "ctx": "She poured water into the pot and",
    "endings": [
        "left it cold.",
        "heated it on the stove.",
        "put it in the fridge.",
        "ate it raw.",
    ],
    "label": 1,
}

_ROW_LABEL_0 = {
    "ind": "2",
    "activity_label": "Running",
    "ctx": "He laced up his shoes and",
    "endings": [
        "started jogging.",
        "sat down.",
        "read a book.",
        "watched TV.",
    ],
    "label": 0,
}


class TestNormalizeRow:
    def test_returns_all_schema_keys(self):
        result = normalize_row(_ROW_LABEL_1)
        assert set(result.keys()) == _SCHEMA_COLS

    def test_subject_is_activity_label(self):
        assert normalize_row(_ROW_LABEL_1)["subject"] == "Cooking"

    def test_question_text_is_ctx(self):
        assert normalize_row(_ROW_LABEL_1)["question_text"] == _ROW_LABEL_1["ctx"]

    def test_choices_are_endings(self):
        opts = _opts(normalize_row(_ROW_LABEL_1))
        assert opts["A"] == "left it cold."
        assert opts["B"] == "heated it on the stove."
        assert opts["C"] == "put it in the fridge."
        assert opts["D"] == "ate it raw."

    def test_correct_option_derived_from_label(self):
        assert normalize_row(_ROW_LABEL_1)["correct_option"] == "B"
        assert normalize_row(_ROW_LABEL_0)["correct_option"] == "A"

    def test_correct_answer_text_matches_ending(self):
        result = normalize_row(_ROW_LABEL_1)
        assert result["correct_answer_text"] == "heated it on the stove."

    @pytest.mark.parametrize("label,expected_option", [
        (0, "A"), (1, "B"), (2, "C"), (3, "D"),
    ])
    def test_all_valid_labels(self, label, expected_option):
        row = {**_ROW_LABEL_1, "label": label}
        result = normalize_row(row)
        assert result["correct_option"] == expected_option

    def test_correct_answer_text_matches_selected_choice(self):
        for label, letter in [(0, "A"), (1, "B"), (2, "C"), (3, "D")]:
            row = {**_ROW_LABEL_1, "label": label}
            result = normalize_row(row)
            assert result["correct_answer_text"] == _opts(result)[letter]

    def test_question_id_is_16_char_hex_string(self):
        qid = normalize_row(_ROW_LABEL_1)["question_id"]
        assert len(qid) == 16
        int(qid, 16)

    def test_question_id_is_deterministic(self):
        r1 = normalize_row(_ROW_LABEL_1)
        r2 = normalize_row(_ROW_LABEL_1)
        assert r1["question_id"] == r2["question_id"]

    def test_different_questions_produce_different_ids(self):
        r1 = normalize_row(_ROW_LABEL_1)
        r2 = normalize_row(_ROW_LABEL_0)
        assert r1["question_id"] != r2["question_id"]

    def test_ind_field_not_used_for_question_id(self):
        row_copy = {**_ROW_LABEL_1, "ind": "999"}
        r1 = normalize_row(_ROW_LABEL_1)
        r2 = normalize_row(row_copy)
        assert r1["question_id"] == r2["question_id"]


class TestBuildNormalizedDataframe:
    def test_returns_dataframe(self):
        df_raw = pd.DataFrame([_ROW_LABEL_1])
        assert isinstance(build_normalized_dataframe(df_raw), pd.DataFrame)

    def test_row_count_matches_input(self):
        df_raw = pd.DataFrame([_ROW_LABEL_1, _ROW_LABEL_0])
        result = build_normalized_dataframe(df_raw)
        assert len(result) == 2

    def test_output_has_all_schema_columns(self):
        df_raw = pd.DataFrame([_ROW_LABEL_1])
        result = build_normalized_dataframe(df_raw)
        assert _SCHEMA_COLS.issubset(set(result.columns))

    def test_empty_input_produces_empty_output(self):
        df_raw = pd.DataFrame(columns=["ind", "activity_label", "ctx", "endings", "label"])
        result = build_normalized_dataframe(df_raw)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_correct_option_is_always_abcd(self):
        df_raw = pd.DataFrame([_ROW_LABEL_1, _ROW_LABEL_0])
        result = build_normalized_dataframe(df_raw)
        assert result["correct_option"].isin(["A", "B", "C", "D"]).all()

    def test_question_ids_are_16_char_hex(self):
        df_raw = pd.DataFrame([_ROW_LABEL_1, _ROW_LABEL_0])
        result = build_normalized_dataframe(df_raw)
        for qid in result["question_id"]:
            assert len(qid) == 16
            int(qid, 16)

    def test_question_ids_are_unique(self):
        df_raw = pd.DataFrame([_ROW_LABEL_1, _ROW_LABEL_0])
        result = build_normalized_dataframe(df_raw)
        assert result["question_id"].nunique() == len(result)

    def test_single_row_round_trips_correctly(self):
        df_raw = pd.DataFrame([_ROW_LABEL_1])
        result = build_normalized_dataframe(df_raw)
        row = result.iloc[0]
        assert row["subject"] == "Cooking"
        assert row["question_text"] == "She poured water into the pot and"
        assert _opts(row)["B"] == "heated it on the stove."
        assert row["correct_option"] == "B"
        assert row["correct_answer_text"] == "heated it on the stove."
