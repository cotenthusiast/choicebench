# tests/scripts/test_prepare_toy_data.py

import importlib.util
from pathlib import Path

import pandas as pd

from choicebench.pipeline.options import build_option_map, correct_option_for_row


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_prepare_toy_data():
    spec = importlib.util.spec_from_file_location(
        "prepare_toy_data_under_test",
        _REPO_ROOT / "scripts" / "prepare_toy_data.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_row_includes_correct_answer_text():
    script = _load_prepare_toy_data()

    row = script._build_row(
        0,
        "What is 2 + 2?",
        "4",
        ["3", "5", "6"],
        script.random.Random(42),
    )

    assert row["correct_option"] == "A"
    assert build_option_map(row)["A"] == "4"
    assert row["correct_answer_text"] == "4"


def test_committed_toy_csv_matches_canonical_schema():
    df = pd.read_csv(_REPO_ROOT / "data" / "processed" / "toy_normalized.csv")

    assert "correct_answer_text" in df.columns
    for _, row in df.iterrows():
        options = build_option_map(row.to_dict())
        correct = correct_option_for_row(row.to_dict())
        assert row["correct_answer_text"] == options[correct]
