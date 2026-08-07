# tests/pipeline/test_options.py

import json

import pytest

from choicebench.benchmarks.base import make_normalized_row
from choicebench.pipeline.options import (
    build_choices,
    build_option_map,
    correct_option_for_row,
    serialize_choices,
)


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


def test_build_option_map_rejects_fewer_than_two_options():
    choices_json = json.dumps([
        {"text": "only", "source_index": 0},
        {"text": "", "source_index": 1},
    ])
    row = {"question_id": "x", "choices_json": choices_json, "correct_option": "A"}
    with pytest.raises(ValueError, match="fewer than 2 valid answer options"):
        build_option_map(row)


def test_build_option_map_rejects_correct_option_not_in_options():
    row = make_normalized_row("cat", "q", ["a", "b", "c"], correct_index=0)
    row["correct_option"] = "D"
    with pytest.raises(ValueError, match="correct_option"):
        build_option_map(row)
