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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_from_huggingface(hf_path: str, hf_subset: str | None, split: str) -> pd.DataFrame:
    """Load a dataset split from HuggingFace and return it as a raw DataFrame."""
    from datasets import load_dataset

    logger.info("Downloading %s (subset=%s, split=%s)...", hf_path, hf_subset or "default", split)
    if hf_subset:
        dataset = load_dataset(hf_path, hf_subset, split=split)
    else:
        dataset = load_dataset(hf_path, split=split)
    return dataset.to_pandas()


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
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()

    stem = args.output_name or args.hf_path.rsplit("/", 1)[-1].lower().replace("-", "_")
    output_path = PROCESSED_DIR / f"{stem}_normalized.csv"

    if output_path.exists():
        logger.info("Normalized file already exists: %s — skipping.", output_path)
        return

    df_raw = download_from_huggingface(args.hf_path, args.hf_subset, args.split)
    logger.info("Downloaded %d rows.", len(df_raw))

    df_normalized = normalize_to_schema(df_raw, args.hf_path, args.hf_subset)
    logger.info("Normalized to %d rows.", len(df_normalized))

    df_normalized.to_csv(output_path, index=False)
    logger.info("Saved → %s", output_path)


if __name__ == "__main__":
    main()
