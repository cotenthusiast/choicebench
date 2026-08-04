# tests/benchmarks/test_truthful_qa.py

import pandas as pd

from choicebench.benchmarks.truthful_qa import normalize_row
from choicebench.benchmarks.registry import BENCHMARK_REGISTRY
from choicebench.pipeline.options import build_option_map

build_normalized_dataframe = BENCHMARK_REGISTRY["truthful_qa"].normalizer

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


def _raw_row(n: int, question: str | None = None) -> dict:
    """A raw TruthfulQA-style row with n mc1 choices; gold is always first."""
    labels = [0] * n
    labels[0] = 1
    return {
        "question": question or f"Question with {n} choices?",
        "mc1_targets": {
            "choices": [f"choice_{i}" for i in range(n)],
            "labels": labels,
        },
        "mc2_targets": {"choices": [], "labels": []},
    }


# A real 5-choice example (correct answer first, as in the raw dataset).
_ROW_5 = {
    "question": "What causes the seasons?",
    "mc1_targets": {
        "choices": [
            "Earth's axial tilt",
            "Distance from the sun",
            "Solar flares",
            "The moon's gravity",
            "Meteor showers",
        ],
        "labels": [1, 0, 0, 0, 0],
    },
    "mc2_targets": {"choices": [], "labels": []},
}


class TestNormalizeRow:
    def test_returns_all_schema_keys(self):
        assert set(normalize_row(_ROW_5).keys()) == _SCHEMA_COLS

    def test_subject_is_constant_truthful_qa(self):
        assert normalize_row(_ROW_5)["subject"] == "truthful_qa"

    def test_question_text_preserved(self):
        assert normalize_row(_ROW_5)["question_text"] == _ROW_5["question"]

    def test_all_mc1_choices_preserved(self):
        # No truncation: all five options survive with labels A-E.
        opts = _opts(normalize_row(_ROW_5))
        assert opts == {
            "A": "Earth's axial tilt",
            "B": "Distance from the sun",
            "C": "Solar flares",
            "D": "The moon's gravity",
            "E": "Meteor showers",
        }

    def test_correct_index_resolves_to_zero_when_gold_first(self):
        # Real-data shape: gold is the first-listed choice → index 0.
        result = normalize_row(_ROW_5)
        assert result["correct_index"] == 0
        assert result["correct_option"] == "A"

    def test_correct_index_derived_from_labels_when_gold_not_first(self):
        # Drift guard: if a future row ever lists gold at a non-zero position,
        # correct_index resolves to the actual gold, not 0 (self-correcting
        # rather than silently mislabeling).
        row = {
            "question": "A drifted row?",
            "mc1_targets": {"choices": ["wrong0", "gold", "wrong2"], "labels": [0, 1, 0]},
            "mc2_targets": {"choices": [], "labels": []},
        }
        result = normalize_row(row)
        assert result["correct_index"] == 1
        assert result["correct_option"] == "B"
        assert result["correct_answer_text"] == "gold"

    def test_correct_answer_text_is_first_choice(self):
        assert normalize_row(_ROW_5)["correct_answer_text"] == "Earth's axial tilt"

    def test_no_truncation_for_many_choices(self):
        # A 13-choice question keeps all 13 options (the raw max in the split).
        result = normalize_row(_raw_row(13))
        assert result["n_choices"] == 13
        assert len(_opts(result)) == 13

    def test_short_question_kept_as_is(self):
        result = normalize_row(_raw_row(2))
        assert result["n_choices"] == 2

    def test_question_id_is_16_char_hex_string(self):
        qid = normalize_row(_ROW_5)["question_id"]
        assert len(qid) == 16
        int(qid, 16)

    def test_question_id_is_deterministic(self):
        assert normalize_row(_ROW_5)["question_id"] == normalize_row(_ROW_5)["question_id"]


class TestBuildNormalizedDataframe:
    def test_returns_dataframe(self):
        df_raw = pd.DataFrame([_ROW_5])
        assert isinstance(build_normalized_dataframe(df_raw), pd.DataFrame)

    def test_output_has_all_schema_columns(self):
        df_raw = pd.DataFrame([_ROW_5])
        result = build_normalized_dataframe(df_raw)
        assert _SCHEMA_COLS.issubset(set(result.columns))

    def test_no_rows_dropped(self):
        # One row per choice-count 2..13 — every one survives.
        df_raw = pd.DataFrame([_raw_row(n) for n in range(2, 14)])
        result = build_normalized_dataframe(df_raw)
        assert len(result) == len(df_raw) == 12

    def test_no_options_discarded_and_n_choices_matches_input(self):
        counts = list(range(2, 14))
        df_raw = pd.DataFrame([_raw_row(n) for n in counts])
        result = build_normalized_dataframe(df_raw)
        # n_choices per row equals the raw choice count exactly — nothing cut.
        assert sorted(result["n_choices"].tolist()) == sorted(counts)
        # Total option count preserved across the whole frame.
        assert result["n_choices"].sum() == sum(counts)

    def test_n_choices_distribution_matches_raw_distribution(self):
        # Mimic the real raw distribution shape (multiplicities per count).
        raw_dist = {2: 3, 3: 2, 4: 5, 5: 4, 10: 1}
        rows = [_raw_row(n) for n, m in raw_dist.items() for _ in range(m)]
        result = build_normalized_dataframe(pd.DataFrame(rows))
        got = result["n_choices"].value_counts().to_dict()
        assert got == raw_dist

    def test_correct_index_zero_for_gold_first_rows(self):
        # All these rows list gold first (real-data shape), so they resolve to 0.
        df_raw = pd.DataFrame([_raw_row(n) for n in range(2, 14)])
        result = build_normalized_dataframe(df_raw)
        assert (result["correct_index"] == 0).all()
        assert (result["correct_option"] == "A").all()

    def test_all_subjects_are_truthful_qa(self):
        df_raw = pd.DataFrame([_ROW_5, _raw_row(6)])
        result = build_normalized_dataframe(df_raw)
        assert (result["subject"] == "truthful_qa").all()

    def test_question_ids_are_unique(self):
        df_raw = pd.DataFrame([_raw_row(n, question=f"q{n}") for n in range(2, 14)])
        result = build_normalized_dataframe(df_raw)
        assert result["question_id"].nunique() == len(result)
