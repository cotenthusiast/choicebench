# src/choicebench/benchmarks/registry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from choicebench.benchmarks.base import build_normalized_dataframe


@dataclass
class BenchmarkEntry:
    name: str
    hf_path: str
    hf_subset: str | None
    default_split: str
    normalizer: Callable[[pd.DataFrame], pd.DataFrame]


BENCHMARK_REGISTRY: dict[str, BenchmarkEntry] = {}


def benchmark(
    name: str,
    hf_path: str,
    hf_subset: str | None = None,
    default_split: str = "test",
) -> Callable:
    """Decorator that registers a row-level normalize_row(row) -> dict function.

    Wraps it with base.build_normalized_dataframe so the registry always
    stores a DataFrame-level normalizer, while each benchmark module only
    needs to define and decorate its row-level function.
    """
    def decorator(normalize_row_fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable:
        def _build_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            return build_normalized_dataframe(df, normalize_row_fn)

        BENCHMARK_REGISTRY[name] = BenchmarkEntry(
            name=name,
            hf_path=hf_path,
            hf_subset=hf_subset,
            default_split=default_split,
            normalizer=_build_normalized_dataframe,
        )
        return normalize_row_fn
    return decorator


def get_by_hf_path(
    hf_path: str,
    hf_subset: str | None = None,
) -> BenchmarkEntry | None:
    """Look up a registry entry by hf_path + hf_subset."""
    for entry in BENCHMARK_REGISTRY.values():
        if entry.hf_path == hf_path and entry.hf_subset == hf_subset:
            return entry
    return None
