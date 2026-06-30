# src/choicebench/benchmarks/registry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


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
    """Decorator that registers a build_normalized_dataframe function."""
    def decorator(fn: Callable) -> Callable:
        BENCHMARK_REGISTRY[name] = BenchmarkEntry(
            name=name,
            hf_path=hf_path,
            hf_subset=hf_subset,
            default_split=default_split,
            normalizer=fn,
        )
        return fn
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
