# tests/metrics/test_order_sensitivity.py

import json
import math

import numpy as np
import pandas as pd

from choicebench.metrics import BUILTIN_METRICS
from choicebench.metrics.order_sensitivity import OrderSensitivity


def _row(mat: list[list[float]], correct_index: int) -> dict:
    return {
        "option_distributions_json": json.dumps(mat),
        "correct_index": correct_index,
    }


def test_registered_under_order_sensitivity():
    assert BUILTIN_METRICS["order_sensitivity"] is OrderSensitivity
    assert OrderSensitivity().name == "order_sensitivity"


def test_no_rotation_data_returns_nan():
    # e.g. a direct_mcq slice: no option_distributions_json / correct_index at all.
    df = pd.DataFrame([{"correct_option": "A", "parsed_choice": "A"}])
    out = OrderSensitivity().compute(df)
    assert math.isnan(out["order_rstd"])
    assert math.isnan(out["order_flip_rate"])


def test_perfect_content_tracking_has_zero_rstd_and_zero_flip_rate():
    # Correct content is canonical index 0. Under rotation k it is printed at
    # position (0-k)%4; a model that always finds it (position-independent)
    # has a one-hot row at exactly that position for every k.
    n = 4
    correct_index = 0
    mat = []
    for k in range(n):
        row = [0.0] * n
        row[(correct_index - k) % n] = 1.0
        mat.append(row)
    df = pd.DataFrame([_row(mat, correct_index)])
    out = OrderSensitivity().compute(df)
    assert out["order_rstd"] == 0.0
    assert out["order_flip_rate"] == 0.0


def test_pure_positional_bias_has_positive_rstd_and_full_flip_rate():
    # Model always picks printed position 0 regardless of rotation -- it is
    # only correct on the one rotation where the correct content happens to
    # land there, and its canonical pick changes every rotation.
    n = 4
    correct_index = 0
    mat = [[1.0, 0.0, 0.0, 0.0] for _ in range(n)]
    df = pd.DataFrame([_row(mat, correct_index)])
    out = OrderSensitivity().compute(df)
    assert out["order_rstd"] > 0.0
    assert math.isclose(out["order_rstd"], float(np.std([1, 0, 0, 0])))
    assert out["order_flip_rate"] == 1.0


def _letters_row(letters: list, correct_index: int) -> dict:
    return {
        "per_rotation_choices_json": json.dumps(letters),
        "correct_index": correct_index,
    }


class TestPerRotationChoicesJson:
    """OrderSensitivity must also compute from per_rotation_choices_json
    (a plain list of canonical letters, one per rotation) -- the lighter
    trace cyclic_permutation, two_stage's rotation rerun, text_extraction's
    rotation rerun, and the other rotation-based flip-rate methods persist,
    as distinct from cyclic_logprob's full probability-matrix
    option_distributions_json. Same rotation convention: under rotation k,
    canonical content originally at position c is printed at position
    (c - k) % n."""

    def test_perfect_content_tracking_has_zero_rstd_and_zero_flip_rate(self):
        # Model always finds canonical content "A" (index 0) regardless of
        # rotation -- position-independent tracking.
        df = pd.DataFrame([_letters_row(["A", "A", "A", "A"], correct_index=0)])
        out = OrderSensitivity().compute(df)
        assert out["order_rstd"] == 0.0
        assert out["order_flip_rate"] == 0.0

    def test_pure_positional_bias_has_positive_rstd_and_full_flip_rate(self):
        # Model always picks whatever content is printed at display position
        # 0. Under rotation k, display position 0 shows canonical index k --
        # so its canonical pick is a different letter every rotation.
        df = pd.DataFrame([_letters_row(["A", "B", "C", "D"], correct_index=0)])
        out = OrderSensitivity().compute(df)
        assert out["order_rstd"] > 0.0
        assert math.isclose(out["order_rstd"], float(np.std([1, 0, 0, 0])))
        assert out["order_flip_rate"] == 1.0

    def test_a_failed_rotation_is_excluded_from_rstd_but_question_still_counted(self):
        # 3 unanimous "A" picks, one failed (None) rotation -- the failure
        # contributes no data point to RStd, but the question still has a
        # real (unanimous) trace, so it's not a flip and is still counted.
        df = pd.DataFrame([_letters_row(["A", "A", None, "A"], correct_index=0)])
        out = OrderSensitivity().compute(df)
        assert out["order_flip_rate"] == 0.0

    def test_every_rotation_failed_excludes_the_question_entirely(self):
        df = pd.DataFrame([
            _letters_row([None, None, None, None], correct_index=0),
            _letters_row(["A", "A", "A", "A"], correct_index=0),
        ])
        out = OrderSensitivity().compute(df)
        # Only the second (fully-successful) row should count.
        assert out["order_flip_rate"] == 0.0

    def test_option_distributions_and_per_rotation_choices_rows_combine(self):
        """A mixed slice (some cyclic_logprob rows, some cyclic_permutation
        rows) must aggregate both kinds into one combined result, not error
        or silently drop one kind."""
        n = 4
        correct_index = 0
        mat = []
        for k in range(n):
            row = [0.0] * n
            row[(correct_index - k) % n] = 1.0
            mat.append(row)
        df = pd.DataFrame([
            _row(mat, correct_index),  # perfect tracking, matrix form
            _letters_row(["A", "A", "A", "A"], correct_index=0),  # perfect tracking, letters form
        ])
        out = OrderSensitivity().compute(df)
        assert out["order_rstd"] == 0.0
        assert out["order_flip_rate"] == 0.0


def test_rows_without_rotation_data_are_skipped_not_averaged_in():
    n = 4
    correct_index = 0
    mat = []
    for k in range(n):
        row = [0.0] * n
        row[(correct_index - k) % n] = 1.0
        mat.append(row)
    df = pd.DataFrame(
        [
            _row(mat, correct_index),
            {"correct_option": "B", "parsed_choice": "B"},  # e.g. a direct_mcq row
        ]
    )
    out = OrderSensitivity().compute(df)
    assert out["order_rstd"] == 0.0
    assert out["order_flip_rate"] == 0.0
