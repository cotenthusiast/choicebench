# src/choicebench/io/writers.py

from pathlib import Path
from typing import Any

import pandas as pd

from choicebench.benchmarks.mmlu import build_normalized_dataframe
from choicebench.config.paths import MMLU_NORMALIZED_PATH, MMLU_RAW_PATH


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
