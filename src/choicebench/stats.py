# src/choicebench/stats.py

"""Dataset-level statistics for normalized benchmarks.

This module is deliberately method-independent: the modal choice count (k) and
its coverage are useful to any part of the codebase (not just PriDe), and the
sidecar it writes can be read back with read_stats() without importing any
method code.

Stats are persisted as a general JSON structure (not a modal-k-only file) so
more derived statistics can be added later without changing the file contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from choicebench.pipeline.options import build_choices

_STATS_SUFFIX = "_stats.json"
_NORMALIZED_SUFFIX = "_normalized.csv"


def n_choices_for_row(row: Mapping[str, Any]) -> int:
    """Number of real options for a normalized row.

    Prefers a persisted ``n_choices`` column; otherwise counts the built
    choices.
    """
    n = row.get("n_choices")
    if n is not None:
        try:
            if not pd.isna(n):
                return int(n)
        except (TypeError, ValueError):
            pass
    return len(build_choices(row))


def choice_count_distribution(df: pd.DataFrame) -> dict[int, int]:
    """Map of choice-count -> number of questions with that many choices."""
    counts: dict[int, int] = {}
    for _, row in df.iterrows():
        k = n_choices_for_row(row.to_dict())
        counts[k] = counts.get(k, 0) + 1
    return counts


def modal_k(counts: Mapping[int, int]) -> int:
    """Most common choice count. Ties are broken deterministically by lowest k.

    Args:
        counts: choice-count -> frequency (from choice_count_distribution).

    Returns:
        The k with the highest frequency; if several k share the top
        frequency, the smallest such k.
    """
    if not counts:
        raise ValueError("Cannot compute modal_k over an empty distribution.")
    # Sort by (frequency desc, k asc); the first element is the winner. Using
    # the (count, -k) key means a frequency tie is resolved by the lowest k.
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def compute_benchmark_stats(df: pd.DataFrame, benchmark: str | None = None) -> dict[str, Any]:
    """Compute the persisted stats structure for a normalized benchmark.

    Returns a general dict (extensible) with the modal choice count, its
    coverage proportion, and the full distribution.
    """
    n_questions = int(len(df))
    dist = choice_count_distribution(df)
    if not dist:
        return {
            "benchmark": benchmark,
            "n_questions": 0,
            "modal_k": None,
            "modal_k_count": 0,
            "modal_k_proportion": None,
            "choice_count_distribution": {},
        }
    k = modal_k(dist)
    k_count = dist[k]
    return {
        "benchmark": benchmark,
        "n_questions": n_questions,
        "modal_k": int(k),
        "modal_k_count": int(k_count),
        "modal_k_proportion": k_count / n_questions,
        # JSON object keys must be strings.
        "choice_count_distribution": {str(kk): int(v) for kk, v in sorted(dist.items())},
    }


def stats_path_for(normalized_csv: Path) -> Path:
    """Sidecar stats path for a normalized CSV (X_normalized.csv -> X_stats.json)."""
    normalized_csv = Path(normalized_csv)
    name = normalized_csv.name
    if name.endswith(_NORMALIZED_SUFFIX):
        stem = name[: -len(_NORMALIZED_SUFFIX)]
    else:
        stem = normalized_csv.stem
    return normalized_csv.with_name(f"{stem}{_STATS_SUFFIX}")


def write_stats(stats: dict[str, Any], path: Path) -> Path:
    """Write a stats dict to a JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))
    return path


def read_stats(path: Path) -> dict[str, Any]:
    """Read a stats sidecar JSON. Raises FileNotFoundError if missing."""
    return json.loads(Path(path).read_text())
