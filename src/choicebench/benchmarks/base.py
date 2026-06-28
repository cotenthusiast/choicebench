# src/choicebench/benchmarks/base.py

from typing import Any, Callable

import pandas as pd


def build_normalized_dataframe(
    df: pd.DataFrame,
    row_normalizer_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> pd.DataFrame:
    """Build a normalized DataFrame by applying a row normalizer to each row.

    Shared by every benchmark module (e.g. choicebench.benchmarks.mmlu,
    choicebench.benchmarks.arc) so each only needs to provide its own
    normalize_row function; the row-iteration and DataFrame assembly are
    identical across benchmarks.

    Args:
        df: Raw benchmark DataFrame.
        row_normalizer_fn: Function that converts one raw row dict into the
            project's normalized schema.

    Returns:
        DataFrame where each row follows the normalized schema produced by
        row_normalizer_fn.
    """
    rows = [row_normalizer_fn(row.to_dict()) for _, row in df.iterrows()]
    return pd.DataFrame(rows)
