# tests/benchmarks/test_arc.py

import pytest
import pandas as pd

from choicebench.benchmarks.arc import normalize_row, build_normalized_dataframe
from choicebench.pipeline.options import build_option_map

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
    """Label -> text map for a normalized row (new variable-choice schema)."""
    return build_option_map(row if isinstance(row, dict) else row.to_dict())

# ---------------------------------------------------------------------------
# Representative raw rows
# ---------------------------------------------------------------------------

_ROW_LETTER_KEYS = {
    "id": "Mercury_7220990",
    "question": "Which factor will most likely cause a person to develop a fever?",
    "choices": {
        "text": [
            "taking a shower in cold water",
            "having a flu virus in the body",
            "smoking cigarettes",
            "eating salted crackers",
        ],
        "label": ["A", "B", "C", "D"],
    },
    "answerKey": "B",
}

_ROW_NUMERIC_KEYS = {
    "id": "NYSEDREGENTS_2015_8_14",
    "question": "What is the primary source of energy for most ecosystems?",
    "choices": {
        "text": ["soil nutrients", "the sun", "decomposers", "producers"],
        "label": ["1", "2", "3", "4"],
    },
    "answerKey": "2",
}


# ---------------------------------------------------------------------------
# normalize_row
# ---------------------------------------------------------------------------


class TestNormalizeRow:
    def test_returns_all_schema_keys(self):
        result = normalize_row(_ROW_LETTER_KEYS)
        assert set(result.keys()) == _SCHEMA_COLS

    def test_subject_is_arc_challenge(self):
        assert normalize_row(_ROW_LETTER_KEYS)["subject"] == "arc_challenge"

    def test_question_text_preserved(self):
        result = normalize_row(_ROW_LETTER_KEYS)
        assert result["question_text"] == _ROW_LETTER_KEYS["question"]

    def test_choices_mapped_correctly_with_letter_labels(self):
        opts = _opts(normalize_row(_ROW_LETTER_KEYS))
        assert opts["A"] == "taking a shower in cold water"
        assert opts["B"] == "having a flu virus in the body"
        assert opts["C"] == "smoking cigarettes"
        assert opts["D"] == "eating salted crackers"

    def test_correct_option_letter_label(self):
        assert normalize_row(_ROW_LETTER_KEYS)["correct_option"] == "B"

    def test_correct_answer_text_letter_label(self):
        result = normalize_row(_ROW_LETTER_KEYS)
        assert result["correct_answer_text"] == "having a flu virus in the body"

    def test_numeric_labels_converted_to_letters(self):
        opts = _opts(normalize_row(_ROW_NUMERIC_KEYS))
        assert opts["A"] == "soil nutrients"
        assert opts["B"] == "the sun"
        assert opts["C"] == "decomposers"
        assert opts["D"] == "producers"

    def test_correct_option_numeric_label_converted(self):
        assert normalize_row(_ROW_NUMERIC_KEYS)["correct_option"] == "B"

    def test_correct_answer_text_numeric_label(self):
        assert normalize_row(_ROW_NUMERIC_KEYS)["correct_answer_text"] == "the sun"

    def test_question_id_is_16_char_hex_string(self):
        qid = normalize_row(_ROW_LETTER_KEYS)["question_id"]
        assert len(qid) == 16
        int(qid, 16)  # raises ValueError if not valid hex

    def test_question_id_is_deterministic(self):
        r1 = normalize_row(_ROW_LETTER_KEYS)
        r2 = normalize_row(_ROW_LETTER_KEYS)
        assert r1["question_id"] == r2["question_id"]

    def test_different_questions_produce_different_ids(self):
        r1 = normalize_row(_ROW_LETTER_KEYS)
        r2 = normalize_row(_ROW_NUMERIC_KEYS)
        assert r1["question_id"] != r2["question_id"]

    def test_question_id_is_content_based_not_arc_id_based(self):
        """Changing the ARC `id` field but keeping content should not change question_id."""
        row_copy = {**_ROW_LETTER_KEYS, "id": "some_completely_different_arc_id"}
        r1 = normalize_row(_ROW_LETTER_KEYS)
        r2 = normalize_row(row_copy)
        assert r1["question_id"] == r2["question_id"]

    @pytest.mark.parametrize("answer_key", ["A", "B", "C", "D"])
    def test_all_valid_answer_keys(self, answer_key):
        row = {
            "id": f"test_{answer_key}",
            "question": "Test question?",
            "choices": {
                "text": ["opt1", "opt2", "opt3", "opt4"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": answer_key,
        }
        result = normalize_row(row)
        assert result["correct_option"] == answer_key

    def test_correct_answer_text_matches_selected_choice(self):
        for letter in ["A", "B", "C", "D"]:
            row = {
                "id": f"test_{letter}",
                "question": "Test?",
                "choices": {
                    "text": ["alpha", "beta", "gamma", "delta"],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": letter,
            }
            result = normalize_row(row)
            assert result["correct_answer_text"] == _opts(result)[letter]

    def test_numeric_answer_key_converted(self):
        for num, letter in [("1", "A"), ("2", "B"), ("3", "C"), ("4", "D")]:
            row = {
                "id": f"test_num_{num}",
                "question": "Test?",
                "choices": {
                    "text": ["w", "x", "y", "z"],
                    "label": ["1", "2", "3", "4"],
                },
                "answerKey": num,
            }
            result = normalize_row(row)
            assert result["correct_option"] == letter


# ---------------------------------------------------------------------------
# build_normalized_dataframe
# ---------------------------------------------------------------------------


class TestBuildNormalizedDataframe:
    def test_returns_dataframe(self):
        df_raw = pd.DataFrame([_ROW_LETTER_KEYS, _ROW_NUMERIC_KEYS])
        assert isinstance(build_normalized_dataframe(df_raw), pd.DataFrame)

    def test_row_count_matches_input(self):
        df_raw = pd.DataFrame([_ROW_LETTER_KEYS, _ROW_NUMERIC_KEYS])
        result = build_normalized_dataframe(df_raw)
        assert len(result) == 2

    def test_output_has_all_schema_columns(self):
        df_raw = pd.DataFrame([_ROW_LETTER_KEYS])
        result = build_normalized_dataframe(df_raw)
        assert _SCHEMA_COLS.issubset(set(result.columns))

    def test_all_subjects_are_arc_challenge(self):
        df_raw = pd.DataFrame([_ROW_LETTER_KEYS, _ROW_NUMERIC_KEYS])
        result = build_normalized_dataframe(df_raw)
        assert (result["subject"] == "arc_challenge").all()

    def test_single_row_round_trips_correctly(self):
        df_raw = pd.DataFrame([_ROW_LETTER_KEYS])
        result = build_normalized_dataframe(df_raw)
        row = result.iloc[0]
        opts = _opts(row)
        assert opts["A"] == "taking a shower in cold water"
        assert opts["B"] == "having a flu virus in the body"
        assert row["correct_option"] == "B"
        assert row["correct_answer_text"] == "having a flu virus in the body"

    def test_all_question_ids_are_unique(self):
        df_raw = pd.DataFrame([_ROW_LETTER_KEYS, _ROW_NUMERIC_KEYS])
        result = build_normalized_dataframe(df_raw)
        assert result["question_id"].nunique() == len(result)
