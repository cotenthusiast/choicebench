# src/choicebench/preflight.py

"""Load preflight questions for methods that require pre-run data (e.g. calibration)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from dataclasses import dataclass

from choicebench.benchmarks.registry import BENCHMARK_REGISTRY
from choicebench.config.paths import (
    PROCESSED_DIR,
    TOY_BENCHMARK_PATH,
)
from choicebench.config.schema import (
    BENCHMARK_HUGGINGFACE,
    BENCHMARK_TOY,
    BenchmarkConfig,
    MethodConfig,
    benchmark_normalized_stem,
)
from choicebench.datasets import (
    PreparedDataset, dataset_content_digest, dataset_sample_identities,
    load_prepared_dataset, spec_for_benchmark,
)
from choicebench.identity import short_id

logger = logging.getLogger(__name__)


def _load_benchmark_artifact(benchmark_cfg: BenchmarkConfig, split: str) -> PreparedDataset:
    return load_prepared_dataset(PROCESSED_DIR, spec_for_benchmark(benchmark_cfg, split=split))


def _load_benchmark_df(benchmark_cfg: BenchmarkConfig, split: str | None = None) -> pd.DataFrame:
    """Compatibility seam; production provenance uses _load_benchmark_artifact."""
    return _load_benchmark_artifact(benchmark_cfg, split or benchmark_cfg.split).dataframe


@dataclass(frozen=True)
class PreflightSelection:
    records: list[dict]
    source: str
    split: str
    artifact_id: str
    content_digest: str
    selection_id: str
    sample_identities: tuple[str, ...]
    artifact_metadata: dict
    artifact_path: str | None
    question_ids: tuple[str, ...]


def load_preflight(
    method_config: MethodConfig,
    benchmark_cfg: BenchmarkConfig,
    run_seed: int,
    eval_question_ids: set[str] | None = None,
    eval_artifact_id: str | None = None,
    eval_sample_identities: set[str] | None = None,
    return_selection: bool = False,
) -> list[dict] | PreflightSelection | None:
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
        if not return_selection:
            if cfg.split == benchmark_cfg.split:
                raise ValueError("Preflight and evaluation splits must be disjoint.")
            df = _load_benchmark_df(benchmark_cfg, cfg.split)
            if eval_question_ids:
                df = df[~df["question_id"].isin(eval_question_ids)]
            n = max(0, min(cfg.n, len(df)))
            return df.sample(n=n, random_state=run_seed).to_dict(orient="records") if n else []
        artifact = _load_benchmark_artifact(benchmark_cfg, cfg.split)
        if eval_artifact_id is not None and artifact.artifact_id == eval_artifact_id:
            raise ValueError("Preflight and evaluation resolve to the same prepared artifact.")
        df = artifact.dataframe
        if eval_question_ids:
            df = df[~df["question_id"].isin(eval_question_ids)]
        n = max(0, min(cfg.n, len(df)))
        if n == 0:
            logger.warning(
                "load_preflight: n=%d yields 0 questions (benchmark has %d rows).",
                cfg.n,
                len(df),
            )
            records: list[dict] = []
            selection = PreflightSelection(
                records, "benchmark", cfg.split, artifact.artifact_id,
                artifact.content_digest, short_id("sel", []), (), artifact.metadata,
                str(artifact.path.relative_to(PROCESSED_DIR)), (),
            )
            return selection if return_selection else records
        sample = df.sample(n=n, random_state=run_seed)
        records = sample.to_dict(orient="records")
        identities = tuple(dataset_sample_identities(sample))
        overlap = set(identities) & set(eval_sample_identities or set())
        if overlap:
            raise ValueError(
                f"Preflight/evaluation content overlap detected across real artifacts: {len(overlap)} sample(s)."
            )
        selection = PreflightSelection(
            records, "benchmark", cfg.split, artifact.artifact_id,
            artifact.content_digest, short_id("sel", {"artifact_id": artifact.artifact_id, "rows": identities}), identities,
            artifact.metadata, str(artifact.path.relative_to(PROCESSED_DIR)),
            tuple(str(record.get("question_id")) for record in records),
        )
        return selection if return_selection else records

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
        df = pd.DataFrame(records)
        records = df.head(cfg.n).to_dict(orient="records")
        identities = tuple(dataset_sample_identities(pd.DataFrame(records)))
        overlap = set(identities) & set(eval_sample_identities or set())
        if overlap:
            raise ValueError(f"Preflight/evaluation content overlap detected: {len(overlap)} sample(s).")
        digest = dataset_content_digest(pd.DataFrame(records))
        selection = PreflightSelection(
            records, Path(cfg.source).name, cfg.split, short_id("file", digest), digest,
            short_id("sel", identities), identities,
            {"provenance_status": "verified-file", "source_name": Path(cfg.source).name},
            Path(cfg.source).name, tuple(str(record.get("question_id")) for record in records),
        )
        return selection if return_selection else records
    if suffix == ".csv":
        df = pd.read_csv(source_path).head(cfg.n)
        records = df.to_dict(orient="records")
        identities = tuple(dataset_sample_identities(df))
        overlap = set(identities) & set(eval_sample_identities or set())
        if overlap:
            raise ValueError(f"Preflight/evaluation content overlap detected: {len(overlap)} sample(s).")
        digest = dataset_content_digest(df)
        selection = PreflightSelection(
            records, Path(cfg.source).name, cfg.split, short_id("file", digest), digest,
            short_id("sel", identities), identities,
            {"provenance_status": "verified-file", "source_name": Path(cfg.source).name},
            Path(cfg.source).name, tuple(str(record.get("question_id")) for record in records),
        )
        return selection if return_selection else records
    raise ValueError(
        f"Unsupported preflight source file type {suffix!r} in {cfg.source!r}. "
        f"Use a .jsonl or .csv file, or set source: benchmark."
    )
