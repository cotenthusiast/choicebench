# src/choicebench/io/writers.py

import json
from pathlib import Path
from typing import Any

import pandas as pd

from choicebench.benchmarks.mmlu import build_normalized_dataframe
from choicebench.config.paths import MMLU_NORMALIZED_PATH, MMLU_RAW_PATH


def write_raw_questions(raw_questions_path: Path = MMLU_RAW_PATH) -> None:
    from datasets import load_dataset
    """
    Downloads the raw MMLU test split and saves it as a CSV file.

    Args:
        raw_questions_path: Full file path where the raw questions CSV
            should be written.
    """
    dataset = load_dataset("cais/mmlu", "all", split="test")
    df = dataset.to_pandas()
    df["choices"] = df["choices"].apply(list)
    df.to_csv(raw_questions_path, index=False)


def write_normalized_questions(
    raw_questions_path: Path = MMLU_RAW_PATH,
    normalized_questions_path: Path = MMLU_NORMALIZED_PATH,
) -> None:
    """
    Reads the raw MMLU CSV, converts it into the project's normalized
    schema, and saves the normalized result as a CSV file.

    Args:
        raw_questions_path: Full file path of the raw questions CSV.
        normalized_questions_path: Full file path where the normalized
            questions CSV should be written.
    """
    df = pd.read_csv(raw_questions_path)
    df_normalized = build_normalized_dataframe(df)
    df_normalized.to_csv(normalized_questions_path, index=False)


def write_split_ids(
    split_ids: list[str],
    split_name: str,
    output_dir: Path,
    artifact_group: str,
) -> None:
    """
    Write one split's question IDs to disk.

    Args:
        split_ids: Ordered list of question IDs belonging to a single split.
        split_name: Logical split name, such as "robustness" or "review".
        output_dir: Root directory under which split artifacts should be written.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Notes:
        This function is responsible only for persistence of the ID list for one
        split. It should not compute metadata or validate split correctness.
    """
    filename = split_name + "_ids.json"
    output_location = output_dir / artifact_group / filename
    output_location.parent.mkdir(parents=True, exist_ok=True)

    with open(output_location, "w", encoding="utf-8") as file:
        json.dump(split_ids, file, indent=2)


def write_split_metadata(
    split_metadata: dict[str, Any],
    split_name: str,
    output_dir: Path,
    artifact_group: str,
) -> None:
    """
    Write one split's metadata dictionary to disk.

    Args:
        split_metadata: Metadata describing a single split artifact.
        split_name: Logical split name, such as "robustness" or "review".
        output_dir: Root directory under which split artifacts should be written.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Notes:
        The metadata is expected to already be built before this function is called.
        This function only serializes and saves it.
    """
    filename = split_name + "_metadata.json"
    output_location = output_dir / artifact_group / filename
    output_location.parent.mkdir(parents=True, exist_ok=True)

    with open(output_location, "w", encoding="utf-8") as file:
        json.dump(split_metadata, file, indent=2)


def write_group_splits(
    split_artifacts: dict[str, dict[str, Any]],
    output_dir: Path,
    artifact_group: str,
) -> None:
    """
    Write the full set of split artifacts to disk.

    Args:
        split_artifacts: Mapping from split name to its full artifact payload.
            Each payload must contain:
                - "ids": list of question IDs
                - "metadata": metadata dictionary for that split
        output_dir: Root directory under which split artifacts should be written.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Notes:
        This is the top-level split writer for the phase. It delegates to the
        single-split writer functions rather than duplicating file logic.
    """
    for split_name, split_artifact in split_artifacts.items():
        write_split_ids(
            split_artifact["ids"],
            split_name,
            output_dir,
            artifact_group,
        )
        write_split_metadata(
            split_artifact["metadata"],
            split_name,
            output_dir,
            artifact_group,
        )


def write_run_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    run_id: str,
    method_name: str,
    model_name: str,
    benchmark: str = "",
) -> Path:
    """Write experiment results to a CSV file.

    The filename encodes the run ID, benchmark, method, and model so that
    results from different conditions and benchmarks never overwrite each other.

    Args:
        results: List of flat result dictionaries produced by a runner.
        output_dir: Directory where the CSV file should be written.
        run_id: Unique identifier for this experimental run.
        method_name: Experimental condition name.
        model_name: Model that produced the results.
        benchmark: Benchmark name (e.g. "mmlu", "arc_challenge").

    Returns:
        Path to the written CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "_")
    parts = [run_id, method_name, safe_model]
    if benchmark:
        parts.append(benchmark)
    filename = "_".join(parts) + ".csv"
    output_path = output_dir / filename

    pd.DataFrame(results).to_csv(output_path, index=False)
    return output_path
