# src/choicebench/io/readers.py

import json
from pathlib import Path
from typing import Any

import pandas as pd

from choicebench.config.paths import RAW_DIR, PROCESSED_DIR, SPLITS_DIR


def read_raw_questions(filename: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Reads a csv file containing the raw MMLU questions.

    Args:
        filename: filename of raw questions
        raw_dir: directory containing the file

    Returns:
        pandas dataframe containing the raw_questions
    """
    raw_location = raw_dir.joinpath(filename)
    df = pd.read_csv(raw_location)
    return df


def read_normalized_questions(filename: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """
    Reads a csv file containing the normalized MMLU questions.

    Args:
        filename: filename of normalized questions
        processed_dir: directory containing the file

    Returns:
        pandas dataframe containing the normalized questions
    """
    processed_location = processed_dir.joinpath(filename)
    return pd.read_csv(processed_location)


def read_split_ids(
    split_name: str,
    input_dir: Path = SPLITS_DIR,
    artifact_group: str = "",
) -> list[str]:
    """
    Read one split's question IDs from disk.

    Args:
        split_name: Logical split name, such as "robustness" or "review".
        input_dir: Root directory from which split artifacts should be read.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Returns:
        Ordered list of question IDs for the requested split.
    """
    filename = f"{split_name}_ids.json"
    split_ids_path = input_dir / artifact_group / filename

    with split_ids_path.open("r", encoding="utf-8") as f:
        split_ids = json.load(f)

    return split_ids


def read_split_metadata(
    split_name: str,
    input_dir: Path = SPLITS_DIR,
    artifact_group: str = "",
) -> dict[str, Any]:
    """
    Read one split's metadata dictionary from disk.

    Args:
        split_name: Logical split name, such as "robustness" or "review".
        input_dir: Root directory from which split artifacts should be read.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Returns:
        Metadata dictionary describing the requested split.
    """
    filename = f"{split_name}_metadata.json"
    split_metadata_path = input_dir / artifact_group / filename

    with split_metadata_path.open("r", encoding="utf-8") as f:
        split_metadata = json.load(f)

    return split_metadata


def read_group_splits(
    artifact_group: str,
    input_dir: Path = SPLITS_DIR,
) -> dict[str, dict[str, Any]]:
    """
    Read the full set of split artifacts for one artifact group from disk.

    Args:
        input_dir: Root directory from which split artifacts should be read.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Returns:
        Mapping from split name to its reconstructed artifact payload.
    """
    group_to_splits = {
        "benchmark": ["robustness"],
        "faithfulness": ["review"],
        "stronger_model": ["review"],
    }

    split_names = group_to_splits[artifact_group]

    data: dict[str, dict[str, Any]] = {}

    for split_name in split_names:
        data[split_name] = {
            "ids": read_split_ids(split_name, input_dir, artifact_group),
            "metadata": read_split_metadata(split_name, input_dir, artifact_group),
        }

    return data


def read_benchmark(path: Path) -> pd.DataFrame:
    """Read a normalized benchmark CSV at a known full path.

    Args:
        path: Full path to the benchmark CSV file.

    Returns:
        DataFrame containing the benchmark questions.
    """
    return pd.read_csv(path)


def read_run_results(input_path: Path) -> pd.DataFrame:
    """Read experiment results from a CSV file.

    Args:
        input_path: Path to the CSV file to read.

    Returns:
        DataFrame containing the experiment results.
    """
    return pd.read_csv(input_path)


def read_all_run_results(
    input_dir: Path,
    run_id: str | None = None,
    method_name: str | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Read and combine results from multiple CSV files.

    Optionally filters by run ID, method, or model using filename
    matching. All matching files are concatenated into a single DataFrame.

    Args:
        input_dir: Directory containing result CSV files.
        run_id: If provided, only include files matching this run ID.
        method_name: If provided, only include files matching this method.
        model_name: If provided, only include files matching this model.

    Returns:
        Combined DataFrame from all matching files.
    """
    frames: list[pd.DataFrame] = []

    for csv_path in sorted(input_dir.glob("*.csv")):
        filename = csv_path.stem

        if run_id and run_id not in filename:
            continue
        if method_name and method_name not in filename:
            continue
        if model_name and model_name not in filename:
            continue

        frames.append(pd.read_csv(csv_path))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
