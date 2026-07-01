# src/choicebench/io/readers.py

from pathlib import Path

import pandas as pd

# Tried longest-first so arc_challenge is matched before a hypothetical
# benchmark whose name is a suffix of it.
_KNOWN_BENCHMARKS = sorted(
    [
        "arc_challenge",
        "truthful_qa",
        "hellaswag",
        "mmlu_pro",
        "mmlu",
        "huggingface",
        "toy",
    ],
    key=len,
    reverse=True,
)


def _infer_benchmark_from_stem(stem: str) -> str:
    """Parse the benchmark name from a result CSV filename stem.

    Filename format: {run_id}_{method_name}_{safe_model}_{benchmark}.
    The benchmark is the suffix after the last _ that matches a known name.
    """
    for name in _KNOWN_BENCHMARKS:
        if stem.endswith("_" + name):
            return name
    return "unknown"


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

        df = pd.read_csv(csv_path)
        if "benchmark_name" not in df.columns:
            df["benchmark_name"] = _infer_benchmark_from_stem(csv_path.stem)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
