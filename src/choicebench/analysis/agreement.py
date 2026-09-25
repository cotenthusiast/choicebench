# src/choicebench/analysis/agreement.py

"""Repeated-sample agreement: given several independent observations of
the same question (e.g. N repeated calls at temperature 0, probing
provider-side non-determinism rather than any positional/order effect),
what fraction of questions produce the identical answer every time?

Deliberately generic and not specific to any one experiment's repetition
protocol -- the caller supplies which column identifies "the same
question" and which column holds "the observed answer for one
repetition"; this module only does the grouping/comparison arithmetic.
"""

from __future__ import annotations

import math

import pandas as pd


def compute_agreement_rate(
        results_df: pd.DataFrame,
        *,
        group_by: str = "question_id",
        answer_col: str = "parsed_choice",
) -> dict[str, float | int]:
    """Fraction of multi-observation groups whose answer_col is identical
    across every observation in the group.

    A group with fewer than 2 observations is excluded entirely -- no
    agreement or disagreement is measurable from a single observation. A
    missing/unparseable answer (None/NaN) is treated as its own distinct
    value for comparison purposes (it never silently "agrees" with a real
    answer), but a group where EVERY observation is missing is still
    counted as agreeing (nunique == 1) -- distinguished via
    n_fully_missing_groups so callers can tell genuine answer agreement
    apart from "the model never answered" agreement.

    If results_df has a 'repetition_index' column, each counted group's own
    repetition_index values must form a complete 0..N-1 range with no gaps
    or duplicates (generic -- not hardcoded to any one experiment's N, but
    naturally requires exactly {0,1,2,3} for this paper's own 4-observation
    stochasticity protocol) -- a group missing an index, or with a
    duplicate, is never silently treated as a valid N-observation group.
    Absent this column entirely (any caller not using repetition indices),
    behavior is unchanged from before this check existed.

    Args:
        results_df: One row per observation (e.g. one row per
            question x repetition).
        group_by: Column identifying which rows are repeated observations
            of the same question.
        answer_col: Column holding each observation's answer.

    Returns:
        {"agreement_rate": float (NaN if no comparable group exists),
         "n_groups": number of groups with >=2 observations,
         "n_fully_missing_groups": of those, how many have every
         observation's answer_col missing}

    Raises:
        ValueError: if results_df has a repetition_index column and any
            counted group's repetition_index values aren't exactly a
            complete, duplicate-free 0..N-1 range.
    """
    if results_df.empty:
        return {"agreement_rate": float("nan"), "n_groups": 0, "n_fully_missing_groups": 0}

    has_repetition_index = "repetition_index" in results_df.columns
    n_groups = 0
    n_agree = 0
    n_fully_missing = 0

    for group_key, group in results_df.groupby(group_by, dropna=False):
        if len(group) < 2:
            continue
        if has_repetition_index:
            actual = sorted(group["repetition_index"].tolist())
            expected = list(range(len(group)))
            if actual != expected:
                raise ValueError(
                    f"compute_agreement_rate: group {group_key!r} has "
                    f"repetition_index values {actual}, expected exactly "
                    f"{expected} (a complete, duplicate-free 0..N-1 range) -- "
                    "refusing to silently treat a missing or duplicated "
                    "observation as a valid group."
                )
        n_groups += 1
        values = group[answer_col]
        unanimous = values.nunique(dropna=False) == 1
        if unanimous:
            n_agree += 1
            if values.isna().all():
                n_fully_missing += 1

    agreement_rate = (n_agree / n_groups) if n_groups > 0 else float("nan")
    return {
        "agreement_rate": agreement_rate,
        "n_groups": n_groups,
        "n_fully_missing_groups": n_fully_missing,
    }
