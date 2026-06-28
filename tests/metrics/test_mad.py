# tests/metrics/test_mad.py

import math

import pandas as pd

from choicebench.metrics import BUILTIN_METRICS
from choicebench.metrics.mad import MAD


def _df(rows: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"correct_option": c, "parsed_choice": p} for c, p in rows]
    )


def test_registered_under_mad():
    assert BUILTIN_METRICS["mad"] is MAD
    assert MAD().name == "mad"


def test_zero_when_prediction_matches_ground_truth_distribution():
    # Predictions exactly mirror the gold distribution → no positional skew.
    rows = [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")]
    out = MAD().compute(_df(rows))
    assert out["mad"] == 0.0


def test_positive_when_predictions_skew_to_one_letter():
    # Gold is uniform across A–D, model always answers A → clear skew.
    rows = [("A", "A"), ("B", "A"), ("C", "A"), ("D", "A")]
    out = MAD().compute(_df(rows))
    assert out["mad"] > 0.0
    assert math.isfinite(out["mad"])


def test_all_unparsed_returns_nan():
    out = MAD().compute(_df([("A", None), ("B", None)]))
    assert math.isnan(out["mad"])
    assert math.isnan(out["mad_std"])


def test_deterministic_bootstrap_std():
    rows = [("A", "A"), ("B", "A"), ("C", "C"), ("D", "A")]
    first = MAD().compute(_df(rows))
    second = MAD().compute(_df(rows))
    assert first["mad"] == second["mad"]
    assert first["mad_std"] == second["mad_std"]  # seeded bootstrap → reproducible
