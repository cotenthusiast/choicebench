# src/choicebench/io/writers.py

from pathlib import Path
from typing import Any

import pandas as pd
import json

from choicebench.identity import file_digest, integrity_digest
from choicebench.infra.atomic_io import atomic_write_json, atomic_write_text

RESULT_ARTIFACT_SCHEMA_VERSION = "choicebench.result-artifact.v1"


def result_artifact_path(result_path: Path) -> Path:
    return Path(result_path).with_suffix(".artifact.json")


def validate_result_artifact(result_path: Path) -> dict[str, Any]:
    metadata_path = result_artifact_path(result_path)
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Result metadata is missing or unreadable: {metadata_path}: {exc}") from exc
    payload = {key: value for key, value in metadata.items() if key != "metadata_digest"}
    if metadata.get("schema_version") != RESULT_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported result metadata schema: {metadata_path}")
    if metadata.get("metadata_digest") != integrity_digest(payload):
        raise RuntimeError(f"Result metadata integrity check failed: {metadata_path}")
    actual = file_digest(result_path)
    if metadata.get("file_sha256") != actual:
        raise RuntimeError(
            f"Result content integrity check failed: {result_path}; expected "
            f"{metadata.get('file_sha256')}, found {actual}."
        )
    return metadata


def write_run_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    run_id: str,
    method_name: str,
    model_name: str,
    benchmark: str = "",
    condition_id: str | None = None,
) -> Path:
    """Write experiment results to a CSV file.

    Config-driven results use the canonical condition ID plus an integrity
    sidecar. The legacy programmatic path retains human-readable filenames.

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
    if condition_id is not None:
        output_dir = output_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{condition_id}.csv"
        if not results:
            raise RuntimeError(f"Refusing to write an empty completed result for {condition_id}.")
        if output_path.exists():
            metadata = validate_result_artifact(output_path)
            if metadata.get("condition_id") != condition_id:
                raise RuntimeError(f"Refusing to overwrite conflicting result artifact {output_path}.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model_name.replace("/", "_")
        parts = [run_id, method_name, safe_model]
        if benchmark:
            parts.append(benchmark)
        filename = "_".join(parts) + ".csv"
        output_path = output_dir / filename

    df = pd.DataFrame(results)
    df["benchmark_name"] = benchmark
    if condition_id is not None:
        identity_columns = (
            "experiment_id", "condition_id", "dataset_artifact_id", "dataset_selection_id",
            "model_id", "method_id", "prompt_id", "benchmark_split",
        )
        identities = {}
        for column in identity_columns:
            # Fail closed: a row with a missing/null identity value is corrupt
            # and must abort the write, never be silently dropped from the check.
            if column not in df or df[column].isna().any():
                raise RuntimeError(
                    f"Result rows for {condition_id} have missing {column} identity values."
                )
            values = sorted(set(df[column].astype(str)))
            if len(values) != 1:
                raise RuntimeError(f"Result rows for {condition_id} have invalid {column} identity.")
            identities[column] = values[0]
    atomic_write_text(output_path, df.to_csv(index=False))
    if condition_id is not None:
        metadata = {
            "schema_version": RESULT_ARTIFACT_SCHEMA_VERSION,
            "condition_id": condition_id,
            "row_count": len(df),
            "columns": list(df.columns),
            "columns_digest": integrity_digest(list(df.columns)),
            "file_sha256": file_digest(output_path),
            "identities": identities,
        }
        metadata["metadata_digest"] = integrity_digest(metadata)
        atomic_write_json(result_artifact_path(output_path), metadata)
    return output_path
