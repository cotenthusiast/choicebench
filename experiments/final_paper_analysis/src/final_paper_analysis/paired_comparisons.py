"""Paired method comparisons (spec section 5).

For two methods (or a method vs. a baseline) evaluated on the same frozen
question set. Correctness here is always "strict" (a missing/parse-failed
prediction counts as incorrect) so both arms share one consistent, always-
defined per-question correctness label -- component 8 (sensitivity) is where
strict-vs-conditional accuracy policy is explicitly varied and compared;
mixing that variability into the paired-comparison primitives themselves
would make win/loss/tie and McNemar counts ambiguous. Never use an
independent-binomial interval for a paired difference -- only the
question-level bootstrap below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


class PairedComparisonError(ValueError):
    """Raised when a paired comparison is given structurally invalid input."""


def paired_correctness_table(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> pd.DataFrame:
    """Join two cells' canonical rows on question_id and derive strict
    correctness (missing/parse-failed -> False) for each arm. Fails closed
    if the two frames don't share exactly the same question_id set -- a
    paired comparison requires the same frozen questions on both sides."""
    ids_a, ids_b = set(frame_a["question_id"]), set(frame_b["question_id"])
    if ids_a != ids_b:
        raise PairedComparisonError(
            f"Paired comparison requires identical question sets; only in A: "
            f"{sorted(ids_a - ids_b)[:5]}, only in B: {sorted(ids_b - ids_a)[:5]}."
        )
    left = frame_a[["question_id", "is_correct"]].rename(columns={"is_correct": "correct_a"})
    right = frame_b[["question_id", "is_correct"]].rename(columns={"is_correct": "correct_b"})
    joined = left.merge(right, on="question_id", how="inner")
    joined["correct_a"] = joined["correct_a"].fillna(False).astype(bool)
    joined["correct_b"] = joined["correct_b"].fillna(False).astype(bool)
    return joined


def paired_accuracy_difference(joined: pd.DataFrame) -> float:
    """mean(correct_a) - mean(correct_b) over the shared question set."""
    if joined.empty:
        raise PairedComparisonError("Cannot compute a paired difference over zero questions.")
    return float(joined["correct_a"].mean() - joined["correct_b"].mean())


@dataclass(frozen=True)
class WinLossTie:
    wins: int
    losses: int
    ties: int
    n_questions: int


def win_loss_tie_counts(joined: pd.DataFrame) -> WinLossTie:
    """wins: A correct, B incorrect. losses: A incorrect, B correct. ties:
    both correct or both incorrect."""
    wins = int(((joined["correct_a"]) & (~joined["correct_b"])).sum())
    losses = int(((~joined["correct_a"]) & (joined["correct_b"])).sum())
    ties = len(joined) - wins - losses
    return WinLossTie(wins=wins, losses=losses, ties=ties, n_questions=len(joined))


@dataclass(frozen=True)
class McNemarResult:
    statistic_b: int
    statistic_c: int
    p_value: float


def mcnemar_test(joined: pd.DataFrame) -> McNemarResult:
    """Exact McNemar test via the sign-test formulation: b/c are the two
    discordant-pair counts (win_loss_tie_counts' wins/losses), and the
    two-sided exact p-value is a binomial test of min(b, c) successes out of
    b+c trials at p=0.5. Zero discordant pairs (b=c=0) has no evidence of a
    difference and is defined here as p_value=1.0, not NaN or an error."""
    counts = win_loss_tie_counts(joined)
    b, c = counts.wins, counts.losses
    if b + c == 0:
        return McNemarResult(statistic_b=b, statistic_c=c, p_value=1.0)
    p_value = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    return McNemarResult(statistic_b=b, statistic_c=c, p_value=float(p_value))


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    seed: int
    n_resamples: int
    replicate_diffs: tuple[float, ...]


def paired_bootstrap_ci(
    joined: pd.DataFrame, *, seed: int, n_resamples: int = 10_000, confidence_level: float = 0.95
) -> BootstrapResult:
    """Question-level (not independent-binomial) bootstrap CI for the paired
    accuracy difference: each replicate resamples question indices with
    replacement (keeping each question's (correct_a, correct_b) pair
    together, preserving the pairing), recomputes the difference, and the CI
    is the percentile interval across replicates. Deterministic from
    ``seed`` -- same seed and inputs always produce bit-identical replicate
    diffs, recorded in full for auditability."""
    if joined.empty:
        raise PairedComparisonError("Cannot bootstrap over zero questions.")
    correct_a = joined["correct_a"].to_numpy()
    correct_b = joined["correct_b"].to_numpy()
    n = len(joined)
    rng = np.random.default_rng(seed)
    replicate_diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        replicate_diffs[i] = correct_a[indices].mean() - correct_b[indices].mean()
    alpha = 1.0 - confidence_level
    lower = float(np.percentile(replicate_diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(replicate_diffs, 100 * (1 - alpha / 2)))
    point_estimate = float(correct_a.mean() - correct_b.mean())
    return BootstrapResult(
        point_estimate=point_estimate, lower=lower, upper=upper,
        confidence_level=confidence_level, seed=seed, n_resamples=n_resamples,
        replicate_diffs=tuple(replicate_diffs.tolist()),
    )


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the same order as the
    input. Standard algorithm: sort ascending, adjusted_(i) =
    max_{j<=i}((m-j+1) * p_(j)) capped at 1.0, enforcing monotonicity before
    mapping back to the original order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted = [0.0] * m
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = (m - rank) * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = min(running_max, 1.0)
    result = [0.0] * m
    for rank, original_index in enumerate(order):
        result[original_index] = adjusted_sorted[rank]
    return result
