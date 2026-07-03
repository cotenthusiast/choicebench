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


def test_gold_and_predicted_percentages_share_the_scored_denominator():
    # 2 scored rows (gold A, B; both predicted A) + 2 unparsed rows (gold C, D).
    # gt_pct must be computed over the scored subset (denominator 2), not all
    # 4 rows — otherwise a benchmark's parse-failure rate would confound MAD.
    rows = [("A", "A"), ("B", "A"), ("C", None), ("D", None)]
    out = MAD().compute(_df(rows))
    # gt%% (scored-only): A=50, B=50, C=0, D=0. pred%%: A=100, B=0, C=0, D=0.
    # deviations: 50, 50, 0, 0 -> mean 25.0. (The pre-fix bug divided gt%% by
    # all 4 rows instead of the 2 scored ones, giving 37.5.)
    assert out["mad"] == 25.0
