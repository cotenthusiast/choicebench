"""Flip-rate analysis module (spec section 9).

Per the methodology handoff (METHODOLOGY_LOCK.md #12) and this session's own
research memory: no historical cyclic-permutation run, cloud or local, in
any repo or era, retained per-permutation trace data -- only aggregate/final
predictions survive. A real flip-rate analysis requires NEW inference with
trace retention that has not yet been executed as of this session. This
module therefore accepts a typed ``PermutationTrace`` schema (one row per
question x permutation) and fails closed -- via
``validate_has_permutation_level_data`` -- against any input that is only
aggregate majority output, rather than assuming traces already exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class FlipRateError(ValueError):
    """Raised when flip-rate input lacks genuine per-permutation granularity
    or is otherwise structurally invalid."""


@dataclass(frozen=True)
class PermutationTrace:
    """One question's response under one specific option-order permutation.
    ``semantic_choice_id`` is the position-invariant identity of the option
    actually selected (i.e. already mapped back through this permutation's
    letter assignment) -- distinct from ``predicted_displayed_letter``,
    which is just which letter was picked in this particular rendering."""

    question_id: str
    model_key: str
    backend: str
    benchmark_name: str
    permutation_index: int
    predicted_displayed_letter: str | None
    semantic_choice_id: str | None
    is_correct: bool | None


def traces_to_frame(traces: Sequence[PermutationTrace]) -> pd.DataFrame:
    return pd.DataFrame([vars(trace) for trace in traces])


def validate_has_permutation_level_data(frame: pd.DataFrame) -> None:
    """Fail closed if ``frame`` is only aggregate/majority output rather
    than genuine per-permutation traces: requires a ``permutation_index``
    column, and requires every question to have more than one distinct
    permutation_index (a single row per question is an aggregate, not a
    trace, no matter what it's labeled)."""
    required = {"question_id", "permutation_index", "semantic_choice_id", "predicted_displayed_letter"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise FlipRateError(
            f"Input lacks per-permutation trace columns {sorted(missing_columns)}; "
            "this looks like aggregate majority output, not a real trace."
        )
    permutation_counts = frame.groupby("question_id")["permutation_index"].nunique()
    aggregate_only = permutation_counts[permutation_counts <= 1]
    if not aggregate_only.empty:
        raise FlipRateError(
            f"{len(aggregate_only)} question(s) have only one distinct "
            f"permutation_index (e.g. {list(aggregate_only.index[:5])}), which is "
            "aggregate majority output, not a genuine per-permutation trace."
        )


@dataclass(frozen=True)
class QuestionFlipStats:
    question_id: str
    n_permutations: int
    n_distinct_semantic_predictions: int
    majority_semantic_choice: str
    majority_vote_stability: float
    semantic_flip_rate: float
    displayed_letter_only_flip_rate: float
    majority_is_correct: bool | None


def _mode(values: Sequence[str]) -> str:
    """Most frequent value; ties broken deterministically by the
    lexicographically greatest value (a real tie has no principled winner,
    so this only needs to be fixed and documented, not "correct")."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda key: (counts[key], key))


def per_question_flip_stats(frame: pd.DataFrame) -> list[QuestionFlipStats]:
    """Compute per-question flip statistics. Requires ``frame`` to have
    already passed ``validate_has_permutation_level_data``. Rows with a
    null ``semantic_choice_id`` (unparseable response for that permutation)
    are excluded from the semantic-flip computation for that question but
    counted in ``n_permutations``."""
    validate_has_permutation_level_data(frame)
    results: list[QuestionFlipStats] = []
    for question_id, group in frame.groupby("question_id"):
        n_permutations = len(group)
        scorable = group[group["semantic_choice_id"].notna()]
        if scorable.empty:
            results.append(QuestionFlipStats(
                question_id=question_id, n_permutations=n_permutations,
                n_distinct_semantic_predictions=0, majority_semantic_choice="",
                majority_vote_stability=float("nan"), semantic_flip_rate=float("nan"),
                displayed_letter_only_flip_rate=float("nan"), majority_is_correct=None,
            ))
            continue
        semantic_choices = list(scorable["semantic_choice_id"])
        majority_choice = _mode(semantic_choices)
        majority_count = semantic_choices.count(majority_choice)
        stability = majority_count / len(semantic_choices)
        flip_rate = 1.0 - stability

        # Displayed-letter-only changes: among rows that agree with the
        # majority SEMANTIC choice, how often does the displayed letter
        # still differ from the majority-agreeing rows' own most common
        # letter (cosmetic-only instability, the model consistently picked
        # the same real answer, just rendered at a different letter each
        # time). Uses the same "1 - majority share" convention as
        # semantic_flip_rate above, not a raw distinct-count ratio: a
        # distinct-count ratio cannot tell a 50/50 letter split from a
        # 75/25 split with the same number of distinct letters (both would
        # wrongly report identical instability), while majority share
        # correctly reports the 50/50 case as more unstable.
        majority_rows = scorable[scorable["semantic_choice_id"] == majority_choice]
        letter_only_flip_rate = 0.0
        if len(majority_rows) > 1:
            letter_counts = majority_rows["predicted_displayed_letter"].value_counts()
            max_letter_count = int(letter_counts.max())
            letter_only_flip_rate = 1.0 - (max_letter_count / len(majority_rows))

        majority_is_correct = None
        correctness_rows = scorable[scorable["semantic_choice_id"] == majority_choice]
        if correctness_rows["is_correct"].notna().any():
            majority_is_correct = bool(correctness_rows["is_correct"].dropna().mode().iloc[0])

        results.append(QuestionFlipStats(
            question_id=question_id, n_permutations=n_permutations,
            n_distinct_semantic_predictions=len(set(semantic_choices)),
            majority_semantic_choice=majority_choice,
            majority_vote_stability=stability, semantic_flip_rate=flip_rate,
            displayed_letter_only_flip_rate=letter_only_flip_rate,
            majority_is_correct=majority_is_correct,
        ))
    return results


def permutation_level_correctness(frame: pd.DataFrame) -> float:
    """Fraction of ALL permutation rows (not per-question majorities) that
    were individually correct. NaN if no row records correctness."""
    scored = frame[frame["is_correct"].notna()]
    if scored.empty:
        return float("nan")
    return float((scored["is_correct"] == True).mean())  # noqa: E712


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient between two paired sequences (e.g.
    per-question flip rate vs. majority correctness, or per-cell mean flip
    rate vs. per-cell MAD). NaN if fewer than 2 pairs or either series has
    zero variance."""
    if len(x) != len(y):
        raise FlipRateError("x and y must have the same length for a paired correlation.")
    if len(x) < 2:
        return float("nan")
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


@dataclass(frozen=True)
class FlipRateBootstrapResult:
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    seed: int
    n_resamples: int


def question_level_bootstrap_ci(
    per_question_values: Sequence[float], *, seed: int, n_resamples: int = 10_000,
    confidence_level: float = 0.95,
) -> FlipRateBootstrapResult:
    """Deterministic question-level bootstrap CI for the mean of any
    per-question flip-rate-derived statistic (e.g. mean semantic flip
    rate)."""
    values = np.asarray([v for v in per_question_values if v == v], dtype=float)  # drop NaN
    if len(values) == 0:
        raise FlipRateError("Cannot bootstrap over zero valid (non-NaN) question values.")
    n = len(values)
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        replicates[i] = values[indices].mean()
    alpha = 1.0 - confidence_level
    return FlipRateBootstrapResult(
        point_estimate=float(values.mean()),
        lower=float(np.percentile(replicates, 100 * (alpha / 2))),
        upper=float(np.percentile(replicates, 100 * (1 - alpha / 2))),
        confidence_level=confidence_level, seed=seed, n_resamples=n_resamples,
    )


def compare_by_group(
    stats: Sequence[QuestionFlipStats], *, group_of: Mapping[str, str]
) -> dict[str, float]:
    """Mean semantic flip rate grouped by an arbitrary label per question
    (e.g. {question_id: 'api'|'local'} or {question_id: 'arc_challenge'|
    'mmlu'}) -- the mechanism behind the required API/local and ARC/MMLU
    flip-rate comparisons. Groups with zero valid (non-NaN) flip rates are
    omitted rather than reported as a fabricated 0.0."""
    grouped: dict[str, list[float]] = {}
    for stat in stats:
        label = group_of.get(stat.question_id)
        if label is None or stat.semantic_flip_rate != stat.semantic_flip_rate:
            continue
        grouped.setdefault(label, []).append(stat.semantic_flip_rate)
    return {label: sum(values) / len(values) for label, values in grouped.items() if values}
