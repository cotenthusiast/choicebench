"""Core cell metrics (spec section 4).

All metrics are recomputed from row-level canonical-table data (see
canonical_table.py), never trusted from any prior aggregate. Every function
documents its exact denominator and what happens when that denominator is
zero (always a NaN with an explicit reason, never a silently invented 0.0 or
1.0). MAD is the primary positional-bias metric; entropy/Jensen-Shannon are
explicitly secondary and never substitute for it as a headline number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


class MetricsError(ValueError):
    """Raised when a metric is computed over structurally invalid input."""


@dataclass(frozen=True)
class ClopperPearsonInterval:
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_correct: int
    n_total: int


def accuracy(n_correct: int, n_total: int) -> float:
    """Plain accuracy = n_correct / n_total. NaN (denominator zero) only
    when n_total == 0 -- a cell with zero rows, which should never occur for
    a real condition; this is a defensive definition, not expected to fire."""
    if n_total == 0:
        return float("nan")
    if n_correct < 0 or n_correct > n_total:
        raise MetricsError(f"n_correct={n_correct} out of range for n_total={n_total}.")
    return n_correct / n_total


def clopper_pearson_interval(
    n_correct: int, n_total: int, *, confidence_level: float = 0.95
) -> ClopperPearsonInterval:
    """Exact Clopper-Pearson binomial confidence interval, correct at the
    0/n and n/n boundaries (no continuity-corrected or normal-approximation
    shortcuts)."""
    if n_total <= 0:
        raise MetricsError("n_total must be positive for a Clopper-Pearson interval.")
    if n_correct < 0 or n_correct > n_total:
        raise MetricsError(f"n_correct={n_correct} out of range for n_total={n_total}.")
    alpha = 1.0 - confidence_level
    lower = 0.0 if n_correct == 0 else beta_dist.ppf(alpha / 2, n_correct, n_total - n_correct + 1)
    upper = 1.0 if n_correct == n_total else beta_dist.ppf(1 - alpha / 2, n_correct + 1, n_total - n_correct)
    return ClopperPearsonInterval(
        point_estimate=n_correct / n_total, lower=float(lower), upper=float(upper),
        confidence_level=confidence_level, n_correct=n_correct, n_total=n_total,
    )


def strict_accuracy(frame: pd.DataFrame) -> float:
    """Denominator = every row in the cell (len(frame)); numerator = rows
    with is_correct == True. A missing/parse-failed prediction counts as
    incorrect (never dropped from the denominator) -- this is the metric
    that answers 'how often did this condition get the right answer, out of
    everything it was asked.'"""
    total = len(frame)
    if total == 0:
        return float("nan")
    correct = int((frame["is_correct"] == True).sum())  # noqa: E712 (is_correct may be None)
    return correct / total


def conditional_accuracy(frame: pd.DataFrame) -> float:
    """Denominator = rows with has_prediction and not parse_failed (i.e.
    is_correct is not None); numerator = is_correct == True among those.
    NaN (with the cell having zero valid predictions) if the denominator is
    zero -- this must never be silently reported as 0.0, since 0/0 is 'no
    information', not 'always wrong'."""
    valid = frame[frame["is_correct"].notna()]
    if len(valid) == 0:
        return float("nan")
    return float((valid["is_correct"] == True).sum()) / len(valid)  # noqa: E712


def coverage(frame: pd.DataFrame) -> float:
    """Fraction of rows with a non-missing prediction (has_prediction),
    regardless of correctness. Denominator = len(frame); NaN only for an
    empty cell."""
    total = len(frame)
    if total == 0:
        return float("nan")
    return float(frame["has_prediction"].sum()) / total


def parse_failure_rate(frame: pd.DataFrame) -> float:
    """Fraction of rows with parse_failed == True. Denominator = len(frame);
    NaN only for an empty cell."""
    total = len(frame)
    if total == 0:
        return float("nan")
    return float(frame["parse_failed"].sum()) / total


def fallback_rate(frame: pd.DataFrame) -> float:
    """Fraction of rows with fallback_used == True, among rows where
    fallback_used is recorded at all (not None). NaN if the method never
    records fallback_used (denominator zero) -- distinct from 0.0, which
    would wrongly claim 'fallback never triggered' for a method that simply
    doesn't track the concept."""
    tracked = frame[frame["fallback_used"].notna()]
    if len(tracked) == 0:
        return float("nan")
    return float((tracked["fallback_used"] == True).sum()) / len(tracked)  # noqa: E712


def selection_frequency_by_position(frame: pd.DataFrame) -> dict[str, float]:
    """{position_letter: fraction of predicted rows selecting that letter}.
    Denominator = rows with has_prediction (predicted_option not null);
    positions never predicted are still reported at 0.0 (an observed zero
    frequency, not a missing value) if any other position was predicted;
    if there are zero predicted rows the whole result is an empty dict."""
    predicted = frame[frame["has_prediction"]]
    if len(predicted) == 0:
        return {}
    counts = predicted["predicted_option"].value_counts()
    total = len(predicted)
    all_positions = sorted(set(predicted["predicted_option"]) | set(frame["gold_answer"]))
    return {position: float(counts.get(position, 0)) / total for position in all_positions}


def accuracy_by_gold_position(frame: pd.DataFrame) -> dict[str, float]:
    """{gold_position_letter: conditional_accuracy over rows whose gold
    answer was displayed at that letter}. A position with zero rows in this
    cell is simply absent from the returned mapping (not present with a
    fabricated value) -- callers must treat a missing key as 'no data for
    this position in this cell', not as 0.0."""
    result: dict[str, float] = {}
    for position, group in frame.groupby("gold_answer"):
        acc = conditional_accuracy(group)
        if not math.isnan(acc):
            result[position] = acc
    return result


def positional_mad(accuracy_by_position: Mapping[str, float]) -> float:
    """ChoiceBench's primary positional-bias metric: mean absolute deviation
    of per-position accuracy from the mean across positions actually present
    (>=1 position required). NaN if fewer than 2 positions have data (MAD is
    undefined/degenerately zero-denominator-adjacent with a single position;
    reported as NaN rather than a misleading 0.0)."""
    values = list(accuracy_by_position.values())
    if len(values) < 2:
        return float("nan")
    mean_accuracy = sum(values) / len(values)
    return sum(abs(value - mean_accuracy) for value in values) / len(values)


def positional_max_min_gap(accuracy_by_position: Mapping[str, float]) -> float:
    """max(per-position accuracy) - min(per-position accuracy). NaN if fewer
    than 2 positions have data, same rationale as positional_mad."""
    values = list(accuracy_by_position.values())
    if len(values) < 2:
        return float("nan")
    return max(values) - min(values)


def selection_entropy(selection_frequency: Mapping[str, float], *, base: float = 2.0) -> float:
    """Secondary metric: Shannon entropy (log base 2 by default) of the
    selection-frequency distribution. NaN if there is no selection data at
    all. This is explicitly secondary and must never be reported as the
    positional-bias headline in place of MAD."""
    probabilities = [p for p in selection_frequency.values() if p > 0]
    if not probabilities:
        return float("nan")
    return -sum(p * math.log(p, base) for p in probabilities)


def selection_jensen_shannon_divergence(
    selection_frequency: Mapping[str, float], *, option_count: int
) -> float:
    """Secondary metric: Jensen-Shannon divergence (base-2, bounded [0, 1])
    between the observed selection-frequency distribution and a uniform
    reference distribution over ``option_count`` positions (the natural
    'no positional bias' null for a cell with that many options per
    question). NaN if there is no selection data. Explicitly secondary --
    must never replace MAD as the paper headline."""
    if option_count < 2:
        raise MetricsError(f"option_count must be >= 2, got {option_count}.")
    if not selection_frequency:
        return float("nan")
    positions = sorted(selection_frequency)
    if len(positions) > option_count:
        raise MetricsError(
            f"selection_frequency has {len(positions)} position(s), more than "
            f"the declared option_count={option_count}."
        )
    observed = np.array([selection_frequency[position] for position in positions], dtype=float)
    observed_sum = observed.sum()
    if observed_sum <= 0:
        return float("nan")
    observed = observed / observed_sum
    reference = np.full(option_count, 1.0 / option_count)
    # Pad the observed distribution to option_count entries (positions with
    # literally zero observed selections among option_count>len(positions)
    # contribute 0 mass, which is a real observation, not a missing value).
    padded = np.zeros(option_count)
    padded[: len(observed)] = observed
    midpoint = (padded + reference) / 2.0

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * _kl(padded, midpoint) + 0.5 * _kl(reference, midpoint)


@dataclass(frozen=True)
class CellMetrics:
    n_total: int
    n_correct: int
    strict_accuracy: float
    conditional_accuracy: float
    clopper_pearson: ClopperPearsonInterval
    coverage: float
    parse_failure_rate: float
    fallback_rate: float
    selection_frequency_by_position: Mapping[str, float]
    accuracy_by_gold_position: Mapping[str, float]
    positional_mad: float
    positional_max_min_gap: float
    selection_entropy: float
    selection_jensen_shannon_divergence: float


def compute_cell_metrics(frame: pd.DataFrame, *, option_count: int) -> CellMetrics:
    """Compute every core cell metric from one cell's canonical rows.
    ``option_count`` is passed explicitly (not inferred from the frame)
    because a cell's option_count is a property of the question set for
    that benchmark, not something to silently re-derive per call."""
    if frame.empty:
        raise MetricsError("Cannot compute cell metrics for an empty frame.")
    n_total = len(frame)
    n_correct = int((frame["is_correct"] == True).sum())  # noqa: E712
    selection_frequency = selection_frequency_by_position(frame)
    accuracy_by_position = accuracy_by_gold_position(frame)
    return CellMetrics(
        n_total=n_total,
        n_correct=n_correct,
        strict_accuracy=strict_accuracy(frame),
        conditional_accuracy=conditional_accuracy(frame),
        clopper_pearson=clopper_pearson_interval(n_correct, n_total),
        coverage=coverage(frame),
        parse_failure_rate=parse_failure_rate(frame),
        fallback_rate=fallback_rate(frame),
        selection_frequency_by_position=selection_frequency,
        accuracy_by_gold_position=accuracy_by_position,
        positional_mad=positional_mad(accuracy_by_position),
        positional_max_min_gap=positional_max_min_gap(accuracy_by_position),
        selection_entropy=selection_entropy(selection_frequency),
        selection_jensen_shannon_divergence=selection_jensen_shannon_divergence(
            selection_frequency, option_count=option_count
        ),
    )
