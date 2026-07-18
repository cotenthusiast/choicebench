"""Generic HuggingFace benchmark downloader and normalizer.

Usage:
    python scripts/prepare_data.py --hf-path cais/mmlu --hf-subset all
    python scripts/prepare_data.py --hf-path allenai/ai2_arc --hf-subset ARC-Challenge
    python scripts/prepare_data.py --hf-path cais/mmlu --output-name my_mmlu
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from choicebench.benchmarks.registry import get_by_hf_path
from choicebench.config.paths import PROCESSED_DIR, ensure_dirs
from choicebench.datasets import (
    DatasetSpec, artifact_csv_path, artifact_stats_path, load_prepared_dataset,
    write_prepared_dataset,
)
from choicebench.permutation_filter import (
    filter_permutation_unsafe,
    permutation_filter_path_for,
    write_permutation_filter_report,
)
from choicebench.stats import compute_benchmark_stats, stats_path_for, write_stats
from choicebench.provenance import resolve_hf_dataset_revision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_from_huggingface(hf_path: str, hf_subset: str | None, split: str, revision: str | None = None) -> pd.DataFrame:
    """Load a dataset split from HuggingFace and return it as a raw DataFrame."""
    from datasets import load_dataset

    resolved_revision = resolve_hf_dataset_revision(hf_path, revision)
    logger.info(
        "Downloading %s (subset=%s, split=%s, commit=%s)...",
        hf_path, hf_subset or "default", split, resolved_revision,
    )
    if hf_subset:
        dataset = load_dataset(hf_path, hf_subset, split=split, revision=resolved_revision)
    else:
        dataset = load_dataset(hf_path, split=split, revision=resolved_revision)
    frame = dataset.to_pandas()
    frame.attrs["hf_fingerprint"] = getattr(dataset, "_fingerprint", None)
    frame.attrs["resolved_revision"] = resolved_revision
    info = getattr(dataset, "info", None)
    frame.attrs["hf_dataset_info"] = {
        "builder_name": getattr(info, "builder_name", None),
        "config_name": getattr(info, "config_name", None),
        "version": str(getattr(info, "version", "")) or None,
    }
    return frame


def normalize_to_schema(
    df: pd.DataFrame, hf_path: str, hf_subset: str | None = None
) -> pd.DataFrame:
    """Dispatch to the right normalizer based on hf_path (and hf_subset).

    To add support for a new dataset:
      1. Create src/choicebench/benchmarks/<name>.py
      2. Implement build_normalized_dataframe(df) decorated with
         @benchmark(name=..., hf_path=..., hf_subset=...)
      3. Import it in src/choicebench/benchmarks/__init__.py
    """
    entry = get_by_hf_path(hf_path, hf_subset)
    if entry is None:
        raise NotImplementedError(
            f"No normalizer registered for dataset {hf_path!r} "
            f"(subset={hf_subset!r}).\n"
            f"To add one:\n"
            f"  1. Create src/choicebench/benchmarks/<name>.py\n"
            f"  2. Implement build_normalized_dataframe(df) decorated "
            f"with @benchmark(name=..., hf_path={hf_path!r}, "
            f"hf_subset={hf_subset!r})\n"
            f"  3. Import it in src/choicebench/benchmarks/__init__.py"
        )
    return entry.normalizer(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize a HuggingFace MCQ dataset."
    )
    parser.add_argument(
        "--hf-path", required=True,
        help="HuggingFace dataset path, e.g. cais/mmlu",
    )
    parser.add_argument(
        "--hf-subset", default=None,
        help="Dataset subset/config name, e.g. all or ARC-Challenge",
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--output-name", default=None,
        help="CSV filename stem; derived from --hf-path if omitted",
    )
    parser.add_argument("--revision", default=None, help="HuggingFace dataset revision/commit.")
    parser.add_argument("--force", action="store_true", help="Replace the exact verified artifact.")
    parser.add_argument(
        "--filter-permutation-unsafe", action="store_true",
        help=(
            "Exclude questions whose option text is meta-referential (e.g. "
            "'A and B', 'none of the above') — see choicebench.permutation_filter. "
            "Opt-in; never applied unless this flag is passed. Unless "
            "--output-name is also given, '_filtered' is appended to the "
            "resolved stem so this never overwrites the canonical unfiltered "
            "normalized CSV that other experiments read."
        ),
    )
    parser.add_argument(
        "--exclude-duplicate-question-ids", action="store_true",
        help=(
            "Remove every row belonging to a duplicate question_id group. The explicit "
            "unique_question_ids_v1 transform becomes part of artifact identity."
        ),
    )
    return parser.parse_args()


def _resolve_output_stem(
    output_name: str | None, hf_path: str, hf_subset: str | None
) -> str:
    """Resolve the CSV filename stem for a prepared dataset.

    Precedence:
      1. An explicit --output-name always wins.
      2. If the (hf_path, hf_subset) pair matches a registered benchmark, use
         that registry entry's name — so e.g. allenai/ai2_arc + ARC-Challenge
         uses the arc_challenge logical artifact name, which is exactly what
         the runtime spec resolves. Without this, the name would be derived
         from the hf_path ("ai2_arc") and the run would never find the file.
      3. Otherwise derive from the hf_path tail (lowercased, hyphens→underscores).
    """
    if output_name:
        return output_name
    entry = get_by_hf_path(hf_path, hf_subset)
    if entry is not None:
        return entry.name
    return hf_path.rsplit("/", 1)[-1].lower().replace("-", "_")


def _write_stats(df: pd.DataFrame, output_path, stem: str, dataset_digest: str | None = None) -> None:
    """Compute and persist dataset-level stats (modal k, distribution)."""
    stats = compute_benchmark_stats(df, benchmark=stem)
    stats["dataset_content_digest"] = dataset_digest
    path = write_stats(stats, artifact_stats_path(output_path))
    logger.info(
        "Stats: modal k=%s over %d questions (%s%% share) → %s",
        stats["modal_k"],
        stats["n_questions"],
        f"{stats['modal_k_proportion'] * 100:.1f}" if stats["modal_k_proportion"] is not None else "n/a",
        path,
    )


def main() -> None:
    ensure_dirs()
    args = parse_args()

    stem = _resolve_output_stem(args.output_name, args.hf_path, args.hf_subset)
    if args.filter_permutation_unsafe and not args.output_name:
        stem = f"{stem}_filtered"
    revision = getattr(args, "revision", None)
    force = getattr(args, "force", False)
    transforms = tuple(
        name for enabled, name in (
            (args.filter_permutation_unsafe, "permutation_safe_v1"),
            (getattr(args, "exclude_duplicate_question_ids", False), "unique_question_ids_v1"),
        ) if enabled
    )
    entry = get_by_hf_path(args.hf_path, args.hf_subset)
    benchmark_name = entry.name if entry is not None else "huggingface"
    spec = DatasetSpec(
        benchmark=benchmark_name, split=args.split, hf_path=args.hf_path,
        hf_subset=args.hf_subset, source_revision=revision,
        transforms=transforms, output_name=stem,
    )
    output_path = artifact_csv_path(PROCESSED_DIR, spec)

    if output_path.exists() and not force:
        artifact = load_prepared_dataset(PROCESSED_DIR, spec)
        logger.info("Verified artifact already exists: %s — skipping download.", output_path)
        if not artifact_stats_path(output_path).exists():
            _write_stats(artifact.dataframe, output_path, stem, artifact.content_digest)
        return

    df_raw = download_from_huggingface(args.hf_path, args.hf_subset, args.split, revision)
    logger.info("Downloaded %d rows.", len(df_raw))

    df_normalized = normalize_to_schema(df_raw, args.hf_path, args.hf_subset)
    logger.info("Normalized to %d rows.", len(df_normalized))

    if args.filter_permutation_unsafe:
        n_before = len(df_normalized)
        df_normalized, exclusions = filter_permutation_unsafe(df_normalized)
        logger.info(
            "Permutation-safety filter: excluded %d/%d question(s) (%.1f%%) with "
            "meta-referential options.",
            len(exclusions), n_before,
            (len(exclusions) / n_before * 100) if n_before else 0.0,
        )
        write_permutation_filter_report(
            exclusions, n_before, permutation_filter_path_for(output_path)
        )

    if getattr(args, "exclude_duplicate_question_ids", False):
        duplicated = df_normalized["question_id"].astype(str).duplicated(keep=False)
        removed = int(duplicated.sum())
        df_normalized = df_normalized.loc[~duplicated].reset_index(drop=True)
        logger.info("Duplicate-ID transform: removed %d rows across all duplicate groups.", removed)

    artifact = write_prepared_dataset(
        df_normalized, PROCESSED_DIR, spec,
        source_metadata={
            "hf_path": args.hf_path, "hf_subset": args.hf_subset, "split": args.split,
            "requested_revision": revision,
            "resolved_revision": df_raw.attrs.get("resolved_revision"),
            "hf_fingerprint": df_raw.attrs.get("hf_fingerprint"),
            "hf_dataset_info": df_raw.attrs.get("hf_dataset_info", {}),
        },
    )
    logger.info("Saved → %s", output_path)
    _write_stats(df_normalized, output_path, stem, artifact.content_digest)


if __name__ == "__main__":
    main()
