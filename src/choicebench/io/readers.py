# src/choicebench/io/readers.py

from pathlib import Path

import pandas as pd


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
