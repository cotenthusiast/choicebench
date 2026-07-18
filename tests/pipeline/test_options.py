# tests/pipeline/test_options.py

import json
from pathlib import Path

import pandas as pd
import pytest

from choicebench.benchmarks.base import make_normalized_row
from choicebench.pipeline.options import (
    build_choices,
    build_option_map,
    correct_option_for_row,
    serialize_choices,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_LEGACY_CSV = _FIXTURES / "legacy_4choice_normalized.csv"


# ---------------------------------------------------------------------------
# New variable-choice schema
# ---------------------------------------------------------------------------

def test_build_option_map_new_schema_six_options():
    row = make_normalized_row("cat", "q", ["o0", "o1", "o2", "o3", "o4", "o5"], correct_index=4)
    assert build_option_map(row) == {
        "A": "o0", "B": "o1", "C": "o2", "D": "o3", "E": "o4", "F": "o5",
    }
    assert row["correct_option"] == "E"


def test_labels_derived_from_order_not_source_index():
    # Source indices are non-contiguous / out of order, but render labels are
    # always A, B, C... by position.
    choices_json = json.dumps([
        {"text": "x", "source_index": 7},
        {"text": "y", "source_index": 2},
        {"text": "z", "source_index": 5},
    ])
    row = {"question_id": "q", "choices_json": choices_json, "correct_option": "B"}
    choices = build_choices(row)
    assert [c["label"] for c in choices] == ["A", "B", "C"]
    assert [c["source_index"] for c in choices] == [7, 2, 5]
    assert build_option_map(row) == {"A": "x", "B": "y", "C": "z"}


def test_correct_option_derived_from_correct_index_when_letter_absent():
    row = make_normalized_row("cat", "q", ["a", "b", "c", "d", "e"], correct_index=3)
    del row["correct_option"]  # only correct_index persisted
    assert correct_option_for_row(row) == "D"


def test_serialize_choices_round_trips_text_and_source_index():
    row = make_normalized_row("cat", "q", ["a", "b", "c"], correct_index=0)
    reparsed = json.loads(serialize_choices(build_choices(row)))
    assert reparsed == [
        {"text": "a", "source_index": 0},
        {"text": "b", "source_index": 1},
        {"text": "c", "source_index": 2},
    ]


# ---------------------------------------------------------------------------
# Legacy choice_a..choice_d schema (backwards compatibility)
# ---------------------------------------------------------------------------

def test_build_option_map_legacy_four_options():
    row = {
        "question_id": "x", "correct_option": "C",
        "choice_a": "A0", "choice_b": "B1", "choice_c": "C2", "choice_d": "D3",
    }
    assert build_option_map(row) == {"A": "A0", "B": "B1", "C": "C2", "D": "D3"}


def test_build_option_map_legacy_drops_empty_trailing_option():
    row = {
        "question_id": "x", "correct_option": "C",
        "choice_a": "one", "choice_b": "two", "choice_c": "three", "choice_d": "",
    }
    assert build_option_map(row) == {"A": "one", "B": "two", "C": "three"}


def test_build_option_map_rejects_fewer_than_two_options():
    row = {"question_id": "x", "correct_option": "A", "choice_a": "only", "choice_b": ""}
    with pytest.raises(ValueError, match="fewer than 2 valid answer options"):
        build_option_map(row)


def test_build_option_map_rejects_correct_option_not_in_options():
    row = {"question_id": "x", "correct_option": "D", "choice_a": "a", "choice_b": "b", "choice_c": "c"}
    with pytest.raises(ValueError, match="correct_option"):
        build_option_map(row)


# ---------------------------------------------------------------------------
# Step 1.3: an already-existing (legacy) normalized CSV still loads correctly
# ---------------------------------------------------------------------------

def test_existing_legacy_normalized_csv_loads_under_new_schema():
    df = pd.read_csv(_LEGACY_CSV)
    assert len(df) == 3

    rows = df.to_dict(orient="records")

    # Row 1: full 4 options.
    r1 = rows[0]
    assert build_option_map(r1) == {"A": "Green", "B": "Blue", "C": "Orange", "D": "Purple"}
    assert correct_option_for_row(r1) == "B"
    assert r1["correct_answer_text"] == "Blue"

    # Row 2: legacy 3-option row (empty choice_d) loads as a 3-option question.
    r2 = rows[1]
    assert build_option_map(r2) == {"A": "3", "B": "4", "C": "5"}
    assert list(build_option_map(r2).keys()) == ["A", "B", "C"]

    # Row 3: option map resolves the correct letter to its text.
    r3 = rows[2]
    assert build_option_map(r3)[correct_option_for_row(r3)] == "Tokyo"
