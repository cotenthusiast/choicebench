"""Section P: metric oracles.

Every expected value below is hand-computed independently of the metric
code under test (per spec: "Do NOT generate expectations by calling
ChoiceBench metric code itself"), using tiny, fully-specified result tables.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, binomtest

from choicebench.analysis.agreement import compute_agreement_rate
from choicebench.analysis.paired_tests import mcnemar_exact_test
from choicebench.metrics.accuracy import Accuracy
from choicebench.metrics.order_sensitivity import OrderSensitivity
from choicebench.metrics.recall_rstd import RecallRStd


# --- P1/P2/P3 ---------------------------------------------------------------

def test_p1_p2_p3_accuracy_conditional_and_unscorable_rate():
    df = pd.DataFrame([
        {"parsed_choice": "A", "correct_option": "A"},  # correct
        {"parsed_choice": "B", "correct_option": "A"},  # wrong
        {"parsed_choice": "A", "correct_option": "A"},  # correct
        {"parsed_choice": None, "correct_option": "A"},  # unscorable
        {"parsed_choice": None, "correct_option": "B"},  # unscorable
    ])
    # P1: end-to-end accuracy -- unscorable counts as incorrect in the denominator.
    expected_accuracy = 2 / 5
    # P2: conditional accuracy -- denominator is scorable rows only.
    expected_conditional = 2 / 3
    # P3: unscorable rate, hand-computed directly from the table (Accuracy
    # itself exposes no field named this -- computed here, not asserted
    # against the metric under test).
    expected_unscorable_rate = 2 / 5

    result = Accuracy().compute(df)
    assert result["accuracy"] == pytest.approx(expected_accuracy)
    assert result["accuracy_conditional"] == pytest.approx(expected_conditional)
    assert df["parsed_choice"].isna().mean() == pytest.approx(expected_unscorable_rate)


# --- P4 ---------------------------------------------------------------

def test_p4_clopper_pearson_matches_scipy_beta_directly():
    k, n = 2, 5
    expected_low = float(beta.ppf(0.025, k, n - k + 1))
    expected_high = float(beta.ppf(0.975, k + 1, n - k))

    df = pd.DataFrame([
        {"parsed_choice": "A", "correct_option": "A"},
        {"parsed_choice": "A", "correct_option": "A"},
        {"parsed_choice": "B", "correct_option": "A"},
        {"parsed_choice": "B", "correct_option": "A"},
        {"parsed_choice": "B", "correct_option": "A"},
    ])
    result = Accuracy().compute(df)
    assert result["accuracy_ci_low"] == pytest.approx(expected_low)
    assert result["accuracy_ci_high"] == pytest.approx(expected_high)


# --- P5 ---------------------------------------------------------------

def test_p5_flip_rate_hand_computed():
    rows = [
        # No flip: all three rotations pick canonical A.
        {"correct_index": 0, "per_rotation_choices_json": json.dumps(["A", "A", "A"])},
        # Flip: rotations disagree (A then B).
        {"correct_index": 0, "per_rotation_choices_json": json.dumps(["A", "B", "A"])},
        # No flip: one rotation failed (null), remaining agree -> 1 unique value.
        {"correct_index": 0, "per_rotation_choices_json": json.dumps(["A", None, "A"])},
    ]
    df = pd.DataFrame(rows)
    result = OrderSensitivity().compute(df)
    assert result["order_flip_rate"] == pytest.approx(1 / 3)


# --- P6 -------------------------------------------------------------------

def test_p6_rstd_from_known_per_letter_recalls():
    # 4 letters, n=2 rows per letter's gold: recalls [1.0, 0.5, 0.0, 0.5] as
    # fractions -> [100, 50, 0, 50] as percentages (RecallRStd's own convention).
    rows = [
        {"correct_option": "A", "parsed_choice": "A"},
        {"correct_option": "A", "parsed_choice": "A"},
        {"correct_option": "B", "parsed_choice": "B"},
        {"correct_option": "B", "parsed_choice": "C"},
        {"correct_option": "C", "parsed_choice": "D"},
        {"correct_option": "C", "parsed_choice": "A"},
        {"correct_option": "D", "parsed_choice": "D"},
        {"correct_option": "D", "parsed_choice": "B"},
    ]
    df = pd.DataFrame(rows)
    expected_rstd = float(np.std([100.0, 50.0, 0.0, 50.0]))  # population SD

    result = RecallRStd().compute(df)
    assert result["recall_A"] == pytest.approx(100.0)
    assert result["recall_B"] == pytest.approx(50.0)
    assert result["recall_C"] == pytest.approx(0.0)
    assert result["recall_D"] == pytest.approx(50.0)
    assert result["rstd"] == pytest.approx(expected_rstd)


# --- P7 ---------------------------------------------------------------

def test_p7_arc_rstd_consumer_must_pre_filter_to_four_option_rows():
    """RecallRStd._label_set() derives its label space from the union of
    correct_option/parsed_choice VALUES present in the WHOLE input frame,
    with no awareness of n_choices (confirmed by reading metrics/base.py's
    _label_set). Feeding it a mixed-arity ARC frame directly would blend
    3-/4-/5-option rows' recall statistics over an incompatible label space.
    The frozen invariant ("only 4-option rows contribute") is therefore a
    CONSUMER responsibility: pre-filter to n_choices==4 before calling
    RecallRStd.compute(), exactly as a real ARC RStd computation must."""
    rows = [
        # 3-option rows (should NOT contribute once filtered out).
        {"n_choices": 3, "correct_option": "A", "parsed_choice": "A"},
        {"n_choices": 3, "correct_option": "B", "parsed_choice": "C"},
        # 5-option rows (should NOT contribute once filtered out).
        {"n_choices": 5, "correct_option": "E", "parsed_choice": "E"},
        {"n_choices": 5, "correct_option": "E", "parsed_choice": "A"},
        # 4-option rows (the only ones that should contribute).
        {"n_choices": 4, "correct_option": "A", "parsed_choice": "A"},
        {"n_choices": 4, "correct_option": "A", "parsed_choice": "A"},
        {"n_choices": 4, "correct_option": "B", "parsed_choice": "B"},
        {"n_choices": 4, "correct_option": "B", "parsed_choice": "A"},
        {"n_choices": 4, "correct_option": "C", "parsed_choice": "C"},
        {"n_choices": 4, "correct_option": "C", "parsed_choice": "C"},
        {"n_choices": 4, "correct_option": "D", "parsed_choice": "D"},
        {"n_choices": 4, "correct_option": "D", "parsed_choice": "A"},
    ]
    df = pd.DataFrame(rows)
    filtered = df[df["n_choices"] == 4]

    result = RecallRStd().compute(filtered)
    assert set(k[len("recall_"):] for k in result if k.startswith("recall_")) == {"A", "B", "C", "D"}
    assert result["recall_A"] == pytest.approx(100.0)
    assert result["recall_B"] == pytest.approx(50.0)
    assert result["recall_C"] == pytest.approx(100.0)
    assert result["recall_D"] == pytest.approx(50.0)

    # Feeding the UNFILTERED mixed-arity frame directly demonstrably pollutes
    # the label space with E (from the 5-option rows) -- proving the filter
    # is load-bearing, not redundant.
    unfiltered_result = RecallRStd().compute(df)
    assert "recall_E" in unfiltered_result


# --- P8 ---------------------------------------------------------------

def _tiny_scored_df(n: int = 20) -> pd.DataFrame:
    letters = ["A", "B", "C", "D"]
    rows = []
    for i in range(n):
        gold = letters[i % 4]
        pred = gold if i % 3 != 0 else letters[(i + 1) % 4]
        rows.append({"correct_option": gold, "parsed_choice": pred})
    return pd.DataFrame(rows)


def test_p8_bootstrap_determinism_and_resample_count():
    import choicebench.metrics.recall_rstd as recall_rstd_module

    assert recall_rstd_module._N_BOOTSTRAP == 10_000
    assert recall_rstd_module._BOOTSTRAP_SEED == 42

    scored = _tiny_scored_df()
    options = ["A", "B", "C", "D"]
    first = RecallRStd._bootstrap_std(scored, options)
    second = RecallRStd._bootstrap_std(scored, options)
    assert first == second  # deterministic under the hardcoded seed (42)


def test_p8_bootstrap_resamples_whole_rows_not_position_cells():
    """Row-level (question-level) resampling means a resampled (gold, pred)
    pair at any position can only ever be one that co-occurred in some real
    original row -- never an artificial combination synthesized by mixing
    one row's gold with another row's pred. Construct a 2-row frame where
    a cell-level (mix-and-match) resampler COULD produce (gold=A,pred=C) --
    a pair that appears in neither real row -- and confirm every resampled
    (gold, pred) pair is drawn from the 2 real pairs that actually exist."""
    scored = pd.DataFrame([
        {"correct_option": "A", "parsed_choice": "B"},
        {"correct_option": "C", "parsed_choice": "D"},
    ])
    options = ["A", "B", "C", "D"]
    opt_index = {opt: i for i, opt in enumerate(options)}
    rng = np.random.default_rng(42)
    n = len(scored)
    gold_enc = np.array([opt_index[v] for v in scored["correct_option"]])
    pred_enc = np.array([opt_index[v] for v in scored["parsed_choice"]])
    idx = rng.integers(0, n, size=(1000, n))
    gold_boot = gold_enc[idx]
    pred_boot = pred_enc[idx]
    real_pairs = {(0, 1), (2, 3)}  # (A,B) and (C,D) by index
    for i in range(1000):
        for j in range(n):
            assert (gold_boot[i, j], pred_boot[i, j]) in real_pairs


# --- P9 -------------------------------------------------------------------

def test_p9_mcnemar_contingency_counts():
    results_a = pd.DataFrame([
        {"question_id": "q1", "is_correct": True},   # both correct
        {"question_id": "q2", "is_correct": False},  # both wrong
        {"question_id": "q3", "is_correct": True},   # a-only correct
        {"question_id": "q4", "is_correct": False},  # b-only correct
    ])
    results_b = pd.DataFrame([
        {"question_id": "q1", "is_correct": True},
        {"question_id": "q2", "is_correct": False},
        {"question_id": "q3", "is_correct": False},
        {"question_id": "q4", "is_correct": True},
    ])
    result = mcnemar_exact_test(results_a, results_b)
    assert result["n_paired"] == 4
    assert result["n_a_only"] == 0
    assert result["n_b_only"] == 0
    assert result["b"] == 1
    assert result["c"] == 1
    assert result["n_discordant"] == 2
    expected_p = float(binomtest(min(1, 1), 2, 0.5).pvalue)
    assert result["p_value"] == pytest.approx(expected_p)
    assert result["p_value"] == pytest.approx(1.0)


# --- P10 ------------------------------------------------------------------

def test_p10_stochasticity_agreement_known_nontrivial_fraction():
    df = pd.DataFrame({
        "question_id": ["q1"] * 4 + ["q2"] * 4 + ["q3"] * 4,
        "parsed_choice": ["A", "A", "A", "A"] + ["B", "B", "B", "B"] + ["A", "A", "A", "B"],
    })
    result = compute_agreement_rate(df)
    assert result["agreement_rate"] == pytest.approx(2 / 3)
