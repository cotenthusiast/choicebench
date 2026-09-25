# src/choicebench/analysis/paired_tests.py

"""Paired significance testing between two methods'/conditions' per-question
correctness on the identical question set.

Confirmed absent from this codebase prior to this addition (repo-wide grep
for "mcnemar"/"McNemar" found nothing) -- needed for the paper's own paired
comparisons (e.g. cyclic majority vote vs. baseline on the same questions).
Deliberately a single small pure function, not a framework: the exact
(binomial) form of McNemar's test, which is simpler to hand-verify and more
correct than the chi-square approximation for the paper's own condition
sizes (in the low hundreds to ~1,200 questions).
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import binomtest


def mcnemar_exact_test(results_a: pd.DataFrame, results_b: pd.DataFrame) -> dict:
    """Exact (binomial) McNemar test comparing two methods' correctness on
    paired questions.

    Args:
        results_a, results_b: each a result DataFrame for ONE method/
            condition, with 'question_id' and 'is_correct' columns. Rows
            are paired by question_id (inner join) -- a question_id
            present in only one side is excluded from the test rather than
            silently dropped without a trace; its count is reported
            separately (n_a_only/n_b_only) so a denominator shrink from a
            mismatched question set is always visible, not hidden inside
            a single aggregate n_paired number.
            A null/NaN is_correct (unscorable row) counts as incorrect,
            matching metrics.accuracy's own "unparseable counts as
            incorrect, not excluded" convention.

    Returns:
        dict with: n_paired, n_a_only, n_b_only, b (a correct & b
        incorrect), c (a incorrect & b correct), n_discordant, p_value,
        statistic ("exact_binomial"). p_value is 1.0 when n_discordant==0
        (no evidence of a difference either way).

    Raises:
        ValueError: if either input has a duplicate question_id -- a
            many-to-many join would silently corrupt every count below.
    """
    for name, df in (("results_a", results_a), ("results_b", results_b)):
        if df["question_id"].duplicated().any():
            raise ValueError(
                f"{name} has duplicate question_id value(s) -- a paired test "
                "requires exactly one row per question_id."
            )

    a = results_a[["question_id", "is_correct"]].rename(columns={"is_correct": "is_correct_a"})
    b = results_b[["question_id", "is_correct"]].rename(columns={"is_correct": "is_correct_b"})
    ids_a = set(a["question_id"])
    ids_b = set(b["question_id"])

    paired = a.merge(b, on="question_id", how="inner")
    n_paired = len(paired)

    correct_a = paired["is_correct_a"].fillna(False).astype(bool)
    correct_b = paired["is_correct_b"].fillna(False).astype(bool)

    n_b = int((correct_a & ~correct_b).sum())  # a correct, b incorrect
    n_c = int((~correct_a & correct_b).sum())  # a incorrect, b correct
    n_discordant = n_b + n_c

    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(n_b, n_c), n_discordant, 0.5).pvalue)

    return {
        "n_paired": n_paired,
        "n_a_only": len(ids_a - ids_b),
        "n_b_only": len(ids_b - ids_a),
        "b": n_b, "c": n_c, "n_discordant": n_discordant,
        "p_value": p_value, "statistic": "exact_binomial",
    }
