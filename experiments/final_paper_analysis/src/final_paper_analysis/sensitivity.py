"""Sensitivity analyses (spec section 8).

Every function here reports two (or more) named variants side by side --
never silently picks the more favorable one. This is the explicit point of
the missing-prediction-policy sensitivity function in particular: all
policies are computed and returned together, with no default "winner".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import pandas as pd

from final_paper_analysis.metrics import conditional_accuracy, strict_accuracy


class SensitivityError(ValueError):
    """Raised when a sensitivity comparison is given structurally invalid input."""


@dataclass(frozen=True)
class SensitivityComparison:
    label_a: str
    value_a: float
    label_b: str
    value_b: float

    @property
    def difference(self) -> float:
        return self.value_a - self.value_b


def with_and_without_repaired_questions(
    frame: pd.DataFrame, *, repaired_question_ids: Sequence[str]
) -> SensitivityComparison:
    """Strict accuracy with all rows vs. with the repaired question_ids
    excluded -- reveals how much of the reported result depends on the 28
    inference-repair patches' 3 rows per cell."""
    without = frame[~frame["question_id"].isin(set(repaired_question_ids))]
    return SensitivityComparison(
        label_a="with_repaired_questions", value_a=strict_accuracy(frame),
        label_b="without_repaired_questions", value_b=strict_accuracy(without),
    )


def strict_vs_conditional(frame: pd.DataFrame) -> SensitivityComparison:
    return SensitivityComparison(
        label_a="strict_accuracy", value_a=strict_accuracy(frame),
        label_b="conditional_accuracy", value_b=conditional_accuracy(frame),
    )


def included_only_vs_included_plus_qualified(frame: pd.DataFrame) -> SensitivityComparison:
    """``frame`` must carry a ``status`` column (canonical_table.py's
    included/qualified/incomplete/excluded). Compares strict accuracy over
    included-only rows vs. included+qualified rows."""
    if "status" not in frame.columns:
        raise SensitivityError("frame must have a 'status' column.")
    included_only = frame[frame["status"] == "included"]
    included_plus_qualified = frame[frame["status"].isin({"included", "qualified"})]
    return SensitivityComparison(
        label_a="included_only", value_a=strict_accuracy(included_only),
        label_b="included_plus_qualified", value_b=strict_accuracy(included_plus_qualified),
    )


def mmlu_duplicate_sensitivity(
    frame: pd.DataFrame, *, duplicate_question_ids: Sequence[str]
) -> SensitivityComparison:
    """Strict accuracy with vs. without MMLU's known duplicate-question_id
    rows (methodology-relevant: frozen selected MMLU questions can contain
    duplicates, e.g. the same underlying question sampled twice)."""
    without = frame[~frame["question_id"].isin(set(duplicate_question_ids))]
    return SensitivityComparison(
        label_a="with_mmlu_duplicates", value_a=strict_accuracy(frame),
        label_b="without_mmlu_duplicates", value_b=strict_accuracy(without),
    )


def api_vs_local_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Strict accuracy per backend ('api'/'local'), reported separately --
    never pooled into one number, per methodology lock #8 (local and API
    model identities stay separate)."""
    if "backend" not in frame.columns:
        raise SensitivityError("frame must have a 'backend' column.")
    result = {}
    for backend, group in frame.groupby("backend"):
        result[backend] = strict_accuracy(group)
    return result


_MISSING_PREDICTION_POLICIES: dict[str, Callable[[pd.DataFrame], float]] = {
    "strict_missing_as_incorrect": strict_accuracy,
    "conditional_exclude_missing": conditional_accuracy,
}


def missing_prediction_policy_sensitivity(frame: pd.DataFrame) -> dict[str, float]:
    """Compute accuracy under every named missing-prediction policy and
    return them all together -- callers must not select only the most
    favorable one for a headline number."""
    return {name: policy(frame) for name, policy in _MISSING_PREDICTION_POLICIES.items()}


def per_group_aggregate(
    frame: pd.DataFrame, *, group_column: str, metric: Callable[[pd.DataFrame], float]
) -> dict[str, float]:
    """Generic per-group aggregate (e.g. group_column='model_key' or
    'benchmark_name'), applying ``metric`` to each group's rows separately."""
    if group_column not in frame.columns:
        raise SensitivityError(f"frame must have a {group_column!r} column.")
    return {str(value): metric(group) for value, group in frame.groupby(group_column)}


def macro_average(per_group_values: Mapping[str, float]) -> float:
    """Unweighted mean across groups -- one vote per group regardless of how
    many rows each group has, so no single model/benchmark dominates the
    aggregate through unequal row counts."""
    values = [value for value in per_group_values.values() if value == value]  # drop NaN
    if not values:
        return float("nan")
    return sum(values) / len(values)
