# src/choicebench/io/writers.py

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import json

from choicebench.identity import canonicalize, file_digest, integrity_digest, short_id
from choicebench.infra.artifacts import atomic_write_json, atomic_write_text

RESULT_ARTIFACT_SCHEMA_VERSION = "choicebench.result-artifact.v1"
RESULT_ARTIFACT_V2_SCHEMA_VERSION = "choicebench.result-artifact.v2"


def result_artifact_path(result_path: Path) -> Path:
    return Path(result_path).with_suffix(".artifact.json")


def validate_result_artifact(
    result_path: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    realization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_path = result_artifact_path(result_path)
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Result metadata is missing or unreadable: {metadata_path}: {exc}") from exc
    schema_version = metadata.get("schema_version")
    if schema_version == RESULT_ARTIFACT_SCHEMA_VERSION:
        payload = {key: value for key, value in metadata.items() if key != "metadata_digest"}
        if metadata.get("metadata_digest") != integrity_digest(payload):
            raise RuntimeError(f"Result metadata integrity check failed: {metadata_path}")
        actual = file_digest(result_path)
        if metadata.get("file_sha256") != actual:
            raise RuntimeError(
                f"Result content integrity check failed: {result_path}; expected "
                f"{metadata.get('file_sha256')}, found {actual}."
            )
        return metadata
    if schema_version != RESULT_ARTIFACT_V2_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported result metadata schema: {metadata_path}")
    payload = {
        key: value
        for key, value in metadata.items()
        if key not in {"result_artifact_id", "result_artifact_digest"}
    }
    if metadata.get("result_artifact_digest") != integrity_digest(payload):
        raise RuntimeError(f"Result metadata integrity check failed: {metadata_path}")
    if metadata.get("result_artifact_id") != short_id("result", payload):
        raise RuntimeError(f"Result metadata ID does not match its own digest: {metadata_path}")
    actual = file_digest(result_path)
    if metadata.get("file_sha256") != actual:
        raise RuntimeError(
            f"Result content integrity check failed: {result_path}; expected "
            f"{metadata.get('file_sha256')}, found {actual}."
        )
    if manifest is not None:
        realization_id = metadata.get("realization_id")
        record = (
            realization
            if realization is not None
            else manifest.get("payload", {}).get("realizations", {}).get(realization_id)
        )
        if record is None or record.get("realization_digest") != metadata.get("realization_digest"):
            raise RuntimeError(f"Result metadata realization binding is inconsistent: {metadata_path}")
        if (
            metadata.get("experiment_id") != manifest.get("experiment_id")
            or metadata.get("experiment_digest") != manifest.get("experiment_digest")
        ):
            raise RuntimeError(f"Result metadata experiment binding is inconsistent: {metadata_path}")
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


@dataclass(frozen=True)
class PreparedResultArtifact:
    result_path: str
    metadata_path: str
    csv_bytes: bytes
    metadata: Mapping[str, Any]


def prepare_manifest_result(
    results: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    realization_id: str,
) -> PreparedResultArtifact:
    """Render one realization's rows and compute result identity after the
    manifest is fixed. Never writes anything; identity here can never depend
    on its own child (the manifest/realization are already immutable inputs).
    """
    payload = manifest["payload"]
    realization = payload["realizations"].get(realization_id)
    if realization is None:
        raise RuntimeError(f"Unknown realization {realization_id!r} in manifest.")
    condition_id = realization["condition_id"]
    condition = payload["semantic_conditions"].get(condition_id)
    if condition is None or condition["condition_digest"] != realization["condition_digest"]:
        raise RuntimeError(f"Realization {realization_id!r} condition binding is inconsistent.")
    condition_identity = condition["identity"]
    result_origin = realization["identity"]["realization"]["result_origin"]

    expected_assignments: dict[str, tuple[str, str]] = {}
    for row in result_origin["row_assignments"]:
        qid = str(row["question_id"])
        if qid in expected_assignments:
            raise RuntimeError(
                f"Realization {realization_id!r} result origin has duplicate question_id {qid!r}."
            )
        expected_assignments[qid] = (row["prediction_origin"], row["prediction_lineage_id"])

    rows_by_qid: dict[str, dict[str, Any]] = {}
    for row in results:
        qid = row.get("question_id")
        qid = None if qid is None else str(qid)
        if qid is None or qid not in expected_assignments:
            raise RuntimeError(
                f"Result row for realization {realization_id!r} references an "
                f"unexpected question_id {row.get('question_id')!r}."
            )
        if qid in rows_by_qid:
            raise RuntimeError(
                f"Result rows for realization {realization_id!r} declare "
                f"question_id {qid!r} more than once."
            )
        expected_origin, expected_lineage_id = expected_assignments[qid]
        if (
            row.get("prediction_origin") != expected_origin
            or row.get("prediction_lineage_id") != expected_lineage_id
        ):
            raise RuntimeError(
                f"Result row {qid!r} prediction_origin/prediction_lineage_id does "
                f"not match its declared realization lineage for {realization_id!r}."
            )
        rows_by_qid[qid] = dict(row)

    missing = sorted(set(expected_assignments) - set(rows_by_qid))
    if missing:
        raise RuntimeError(
            f"Realization {realization_id!r} is missing result rows for {missing}."
        )

    ordered_question_ids = [str(row["question_id"]) for row in result_origin["row_assignments"]]
    ordered_rows = [rows_by_qid[qid] for qid in ordered_question_ids]

    benchmark = condition_identity["benchmark"]
    identity_columns = {
        "condition_id": condition_id,
        "realization_id": realization_id,
        "experiment_id": manifest["experiment_id"],
        "dataset_artifact_id": benchmark["artifact_id"],
        "dataset_selection_id": benchmark["selection_id"],
        "model_id": condition_identity["model_id"],
        "method_id": condition_identity["method_id"],
        "prompt_id": condition_identity["prompt_id"],
        "benchmark_name": benchmark["name"],
        "benchmark_split": benchmark["split"],
    }
    for row in ordered_rows:
        for column, expected_value in identity_columns.items():
            if row.get(column) != expected_value:
                raise RuntimeError(
                    f"Result row {row.get('question_id')!r} for realization "
                    f"{realization_id!r} has missing or incorrect {column!r}."
                )

    df = pd.DataFrame(ordered_rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    result_path = f"results/{realization_id}.csv"
    metadata_path = f"results/{realization_id}.artifact.json"

    prediction_origin_counts: dict[str, int] = {}
    for row in ordered_rows:
        origin = row["prediction_origin"]
        prediction_origin_counts[origin] = prediction_origin_counts.get(origin, 0) + 1

    metadata_core = {
        "schema_version": RESULT_ARTIFACT_V2_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "experiment_digest": manifest["experiment_digest"],
        "condition_id": condition_id,
        "condition_digest": condition["condition_digest"],
        "realization_id": realization_id,
        "realization_digest": realization["realization_digest"],
        "row_count": len(ordered_rows),
        "columns": list(df.columns),
        "question_ids": ordered_question_ids,
        "rows_digest": integrity_digest(ordered_rows),
        "file_sha256": sha256(csv_bytes).hexdigest(),
        "prediction_origins": sorted(prediction_origin_counts),
        "prediction_origin_counts": {
            key: prediction_origin_counts[key] for key in sorted(prediction_origin_counts)
        },
        "result_path": result_path,
        "metadata_path": metadata_path,
    }
    result_artifact_digest = integrity_digest(metadata_core)
    metadata = {
        **metadata_core,
        "result_artifact_id": short_id("result", metadata_core),
        "result_artifact_digest": result_artifact_digest,
    }
    return PreparedResultArtifact(
        result_path=result_path,
        metadata_path=metadata_path,
        csv_bytes=csv_bytes,
        metadata=canonicalize(metadata),
    )


def publish_manifest_result(
    prepared: PreparedResultArtifact, *, run_dir: Path
) -> tuple[Path, Mapping[str, Any], bool]:
    """Write a prepared result artifact, or verify-only if it already exists.

    Returns (result_path, metadata, reused). `reused=True` means an identical
    artifact already existed; any divergence is refused rather than overwritten.
    """
    run_dir = Path(run_dir)
    result_path = run_dir / prepared.result_path
    metadata_path = run_dir / prepared.metadata_path
    if result_path.exists() != metadata_path.exists():
        raise RuntimeError(f"Realization result/sidecar pair is incomplete: {result_path}")
    if result_path.exists():
        existing_bytes = result_path.read_bytes()
        try:
            existing_metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Existing result metadata is unreadable: {metadata_path}: {exc}"
            ) from exc
        if existing_bytes == prepared.csv_bytes and existing_metadata == dict(prepared.metadata):
            return result_path, existing_metadata, True
        raise RuntimeError(
            f"Refusing to overwrite a divergent existing result artifact: {result_path}"
        )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(result_path, prepared.csv_bytes.decode("utf-8"))
    atomic_write_json(metadata_path, dict(prepared.metadata))
    return result_path, prepared.metadata, False
