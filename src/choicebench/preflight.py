# src/choicebench/preflight.py

"""Load preflight questions for methods that require pre-run data (e.g. calibration)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from choicebench.benchmarks.registry import BENCHMARK_REGISTRY
from choicebench.config.paths import (
    PROCESSED_DIR,
    TOY_BENCHMARK_PATH,
    get_benchmark_path,
)
from choicebench.config.schema import (
    BENCHMARK_HUGGINGFACE,
    BENCHMARK_TOY,
    BenchmarkConfig,
    MethodConfig,
    benchmark_normalized_stem,
)
from choicebench.io.readers import read_benchmark

logger = logging.getLogger(__name__)


def _load_benchmark_df(benchmark_cfg: BenchmarkConfig) -> pd.DataFrame:
    name = benchmark_cfg.name
    if name == BENCHMARK_TOY:
        return read_benchmark(TOY_BENCHMARK_PATH)
    if name == BENCHMARK_HUGGINGFACE:
        stem = benchmark_normalized_stem(benchmark_cfg)
        csv_path = PROCESSED_DIR / f"{stem}_normalized.csv"
        return read_benchmark(csv_path)
    if name in BENCHMARK_REGISTRY:
        return read_benchmark(get_benchmark_path(name))
    raise ValueError(f"Unknown benchmark for preflight loading: {name!r}")


def load_preflight(
    method_config: MethodConfig,
    benchmark_cfg: BenchmarkConfig,
    run_seed: int,
    eval_question_ids: set[str] | None = None,
) -> list[dict] | None:
    """Load preflight questions for a method config, or return None if not configured.

    Args:
        method_config: The method's config, which may contain a preflight block.
        benchmark_cfg: The benchmark being evaluated (used when source is "benchmark").
        run_seed: Seed for reproducible sampling.
        eval_question_ids: question_ids already selected for the evaluation sample.
            When source is "benchmark", these are excluded from the calibration
            pool before sampling — the split-label check alone does not
            guarantee disjoint rows, since "benchmark" reloads the same
            normalized CSV the eval sample was drawn from.

    Returns:
        A list of question dicts in the standard record schema, or None if the
        method has no preflight block.

    Raises:
        ValueError: If source is "benchmark" and preflight split matches eval split
            (disjointness violation), or if the source file type is unsupported.
    """
    if method_config.preflight is None:
        return None

    cfg = method_config.preflight

    if cfg.source == "benchmark":
        if cfg.split == benchmark_cfg.split:
            raise ValueError(
                f"Preflight split {cfg.split!r} matches the evaluation split "
                f"{benchmark_cfg.split!r}. Preflight questions must be disjoint "
                f"from the evaluation set — use a different split (e.g. 'validation')."
            )
        df = _load_benchmark_df(benchmark_cfg)
        if eval_question_ids:
            df = df[~df["question_id"].isin(eval_question_ids)]
        n = max(0, min(cfg.n, len(df)))
        if n == 0:
            logger.warning(
                "load_preflight: n=%d yields 0 questions (benchmark has %d rows).",
                cfg.n,
                len(df),
            )
            return []
        sample = df.sample(n=n, random_state=run_seed)
        return sample.to_dict(orient="records")

    # source is a file path
    source_path = Path(cfg.source)
    suffix = source_path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict] = []
        with open(source_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if suffix == ".csv":
        return pd.read_csv(source_path).to_dict(orient="records")
    raise ValueError(
        f"Unsupported preflight source file type {suffix!r} in {cfg.source!r}. "
        f"Use a .jsonl or .csv file, or set source: benchmark."
    )
