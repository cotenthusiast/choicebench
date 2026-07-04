# tests/metrics/test_recall_rstd.py

import math

import pandas as pd

from choicebench.metrics import BUILTIN_METRICS
from choicebench.metrics.recall_rstd import RecallRStd


def _df(rows: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"correct_option": c, "parsed_choice": p} for c, p in rows]
    )


def test_registered_under_recall_rstd():
    assert BUILTIN_METRICS["recall_rstd"] is RecallRStd
    assert RecallRStd().name == "recall_rstd"


def test_hand_computed_10_row_fixture():
    # A: 4 gold, 3 correct -> recall_A = 75
    # B: 4 gold, 2 correct -> recall_B = 50
    # C: 1 gold, 1 correct -> recall_C = 100
    # D: 1 gold, 0 correct -> recall_D = 0
    rows = [
        ("A", "A"), ("A", "A"), ("A", "A"), ("A", "B"),
        ("B", "B"), ("B", "B"), ("B", "A"), ("B", "C"),
        ("C", "C"),
        ("D", "A"),
    ]
    out = RecallRStd().compute(_df(rows))

    assert out["recall_A"] == 75.0
    assert out["recall_B"] == 50.0
    assert out["recall_C"] == 100.0
    assert out["recall_D"] == 0.0
    assert out["n_per_letter_A"] == 4.0
    assert out["n_per_letter_B"] == 4.0
    assert out["n_per_letter_C"] == 1.0
    assert out["n_per_letter_D"] == 1.0

    # mean=56.25; sum of squared deviations=5468.75; variance (ddof=0)=1367.1875
    expected_rstd = math.sqrt(1367.1875)
    assert out["rstd"] == expected_rstd


def test_5_letter_case():
    rows = [
        ("A", "A"), ("A", "A"),
        ("B", "B"), ("B", "C"),
        ("C", "A"), ("C", "B"),
        ("D", "D"),
        ("E", "E"),
    ]
    out = RecallRStd().compute(_df(rows))

    assert out["recall_A"] == 100.0
    assert out["recall_B"] == 50.0
    assert out["recall_C"] == 0.0
    assert out["recall_D"] == 100.0
    assert out["recall_E"] == 100.0
    assert out["rstd"] == 40.0


def test_all_one_letter_gold_degenerate_case():
    # Every gold answer is "A"; the model sometimes answers other letters, so
    # B/C/D appear in the label space (via parsed_choice) but never as gold.
    rows = [
        ("A", "A"), ("A", "A"), ("A", "B"), ("A", "C"), ("A", "A"),
    ]
    out = RecallRStd().compute(_df(rows))

    assert out["n_per_letter_A"] == 5.0
    assert out["n_per_letter_B"] == 0.0
    assert out["n_per_letter_C"] == 0.0
    assert out["recall_A"] == 60.0  # 3/5 correct
    assert math.isnan(out["recall_B"])
    assert math.isnan(out["recall_C"])
    # Only letter A has a defined recall -> std of a single value is 0.
    assert out["rstd"] == 0.0


def test_scored_subset_excludes_unparsed_rows():
    # Unparsed rows (parsed_choice is None) must not count toward either the
    # numerator or denominator of any letter's recall -- same semantics as MAD.
    rows = [
        ("A", "A"), ("A", None),
        ("B", "B"), ("B", None),
    ]
    out = RecallRStd().compute(_df(rows))

    assert out["n_per_letter_A"] == 1.0
    assert out["n_per_letter_B"] == 1.0
    assert out["recall_A"] == 100.0
    assert out["recall_B"] == 100.0
    assert out["rstd"] == 0.0


def test_empty_dataframe_returns_nan():
    out = RecallRStd().compute(_df([]))
    assert math.isnan(out["rstd"])
