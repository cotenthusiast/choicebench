# tests/metrics/test_accuracy.py

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
    assert out == {"accuracy": 1.0, "accuracy_conditional": 1.0}


def test_mixed_correctness():
    out = Accuracy().compute(_df([("A", "A"), ("B", "C"), ("C", "C")]))
    assert out["accuracy"] == 2 / 3
    assert out["accuracy_conditional"] == 2 / 3


def test_unparsed_rows_lower_end_to_end_but_not_conditional():
    # 1 correct, 1 wrong, 1 unparsed (None) → end-to-end 1/3, conditional 1/2
    out = Accuracy().compute(_df([("A", "A"), ("B", "C"), ("C", None)]))
    assert out["accuracy"] == 1 / 3
    assert out["accuracy_conditional"] == 1 / 2


def test_empty_frame():
    out = Accuracy().compute(_df([]))
    assert out == {"accuracy": 0.0, "accuracy_conditional": 0.0}
