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

from mcq_eval.benchmarks.arc import build_normalized_dataframe as normalize_arc
from mcq_eval.benchmarks.mmlu import build_normalized_dataframe as normalize_mmlu
from mcq_eval.config.paths import PROCESSED_DIR, ensure_dirs

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


def normalize_to_schema(df: pd.DataFrame, hf_path: str) -> pd.DataFrame:
    """Dispatch to the right normalizer based on hf_path.

    HuggingFace delivers data in its native format; each normalizer branch
    converts it to the project's canonical schema before calling the shared
    build_normalized_dataframe() from the benchmark module.

    To add support for a new dataset:
      1. Implement normalize_row() in src/mcq_eval/benchmarks/<name>.py
      2. Add a branch here that pre-processes the raw DataFrame if needed
         and calls build_normalized_dataframe() from that module.
    """
    if hf_path == "cais/mmlu":
        # HuggingFace delivers choices as a Python list; the MMLU normalizer
        # expects the string-serialized form (it calls ast.literal_eval).
        df = df.copy()
        df["choices"] = df["choices"].apply(str)
        return normalize_mmlu(df)

    if hf_path == "allenai/ai2_arc":
        # HuggingFace delivers choices as a dict {"text": [...], "label": [...]},
        # which the ARC normalizer consumes directly.
        return normalize_arc(df)

    raise NotImplementedError(
        f"No normalizer registered for dataset {hf_path!r}. "
        f"To add one: implement normalize_row() in "
        f"src/mcq_eval/benchmarks/<your_name>.py and add a branch in "
        f"normalize_to_schema() in scripts/prepare_data.py."
    )


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

    df_normalized = normalize_to_schema(df_raw, args.hf_path)
    logger.info("Normalized to %d rows.", len(df_normalized))

    df_normalized.to_csv(output_path, index=False)
    logger.info("Saved → %s", output_path)


if __name__ == "__main__":
    main()
