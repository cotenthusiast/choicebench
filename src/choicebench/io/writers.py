# src/choicebench/io/writers.py

from pathlib import Path
from typing import Any

import pandas as pd


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
