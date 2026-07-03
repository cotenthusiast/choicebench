# tests/metrics/test_accuracy.py

import math

import pandas as pd

from choicebench.metrics import BUILTIN_METRICS
from choicebench.metrics.accuracy import Accuracy


def _df(rows: list[tuple[str, object]]) -> pd.DataFrame:
    """Build a results frame from (correct_option, parsed_choice) tuples."""
    return pd.DataFrame(
        [{"correct_option": c, "parsed_choice": p} for c, p in rows]
    )


def test_registered_under_accuracy():
    assert BUILTIN_METRICS["accuracy"] is Accuracy
    assert Accuracy().name == "accuracy"


def test_all_correct():
    out = Accuracy().compute(_df([("A", "A"), ("B", "B")]))
    assert out["accuracy"] == 1.0
    assert out["accuracy_conditional"] == 1.0
    # All-correct with n=2 still has a real (non-degenerate) upper bound < 1.
    assert out["accuracy_ci_low"] > 0.0
    assert out["accuracy_ci_high"] == 1.0


def test_mixed_correctness():
    out = Accuracy().compute(_df([("A", "A"), ("B", "C"), ("C", "C")]))
    assert out["accuracy"] == 2 / 3
    assert out["accuracy_conditional"] == 2 / 3
    assert 0.0 < out["accuracy_ci_low"] < out["accuracy"] < out["accuracy_ci_high"] < 1.0


def test_unparsed_rows_lower_end_to_end_but_not_conditional():
    # 1 correct, 1 wrong, 1 unparsed (None) → end-to-end 1/3, conditional 1/2
    out = Accuracy().compute(_df([("A", "A"), ("B", "C"), ("C", None)]))
    assert out["accuracy"] == 1 / 3
    assert out["accuracy_conditional"] == 1 / 2
    # accuracy's CI is over n=3 (all rows); accuracy_conditional's is over
    # n=2 (scored only) — different denominators, both must bracket their
    # own point estimate.
    assert 0.0 <= out["accuracy_ci_low"] < out["accuracy"] < out["accuracy_ci_high"] <= 1.0
    assert (
        0.0 <= out["accuracy_conditional_ci_low"]
        < out["accuracy_conditional"]
        < out["accuracy_conditional_ci_high"]
        <= 1.0
    )


def test_all_zero_correct_ci_low_is_zero():
    out = Accuracy().compute(_df([("A", "B"), ("A", "C")]))
    assert out["accuracy"] == 0.0
    assert out["accuracy_ci_low"] == 0.0
    assert out["accuracy_ci_high"] > 0.0


def test_all_unparsed_conditional_ci_is_nan():
    out = Accuracy().compute(_df([("A", None), ("B", None)]))
    assert out["accuracy_conditional"] == 0.0
    assert math.isnan(out["accuracy_conditional_ci_low"])
    assert math.isnan(out["accuracy_conditional_ci_high"])


def test_empty_frame():
    out = Accuracy().compute(_df([]))
    assert out["accuracy"] == 0.0
    assert out["accuracy_conditional"] == 0.0
    assert math.isnan(out["accuracy_ci_low"])
    assert math.isnan(out["accuracy_ci_high"])
    assert math.isnan(out["accuracy_conditional_ci_low"])
    assert math.isnan(out["accuracy_conditional_ci_high"])
