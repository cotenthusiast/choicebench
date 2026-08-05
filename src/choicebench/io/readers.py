# src/choicebench/io/readers.py

from pathlib import Path
import json

import pandas as pd
from choicebench.manifest import ManifestCompatibilityError, validate_manifest, validate_run_state
from choicebench.io.writers import result_artifact_path, validate_result_artifact
from choicebench.identity import integrity_digest
from choicebench.datasets import dataset_content_digest


class ResultSetError(RuntimeError):
    pass


def read_manifest_results(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Read exactly the completed result artifacts declared by a run manifest.

    Completed conditions are validated and concatenated; gated and failed
    conditions contribute no rows but are legal states (the evaluator accounts
    for them). Any other status means the run has not finished and is refused.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    state_path = run_dir / "run_state.json"
    if not manifest_path.exists():
        raise ResultSetError(
            f"Run {run_dir} has no publication-grade manifest; legacy aggregation is refused."
        )
    manifest = json.loads(manifest_path.read_text())
    try:
        validate_manifest(manifest)
    except ManifestCompatibilityError as exc:
        raise ResultSetError(str(exc)) from exc
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state is None:
        raise ResultSetError("Missing or incompatible run_state.json.")
    try:
        validate_run_state(state, manifest)
    except ManifestCompatibilityError as exc:
        raise ResultSetError(str(exc)) from exc
    condition_items = manifest["payload"]["conditions"]
    conditions = {c["condition_id"]: c for c in condition_items}
    if len(conditions) != len(condition_items):
        raise ResultSetError("Manifest declares duplicate condition IDs.")
    snapshot_problems: list[str] = []
    for section in ("datasets", "calibrations"):
        for record in manifest["payload"].get(section, []):
            path = run_dir / record["run_snapshot_path"]
            if not path.is_file():
                snapshot_problems.append(f"{record['selection_id']}: missing snapshot {record['run_snapshot_path']}")
                continue
            try:
                frame = (
                    pd.read_csv(path, dtype={"question_id": "string"})
                    if int(record["row_count"]) else pd.DataFrame()
                )
            except (OSError, pd.errors.EmptyDataError) as exc:
                snapshot_problems.append(f"{record['selection_id']}: unreadable snapshot: {exc}")
                continue
            expected_digest = record["selected_content_digest"]
            actual_ids = [str(value) for value in frame.get("question_id", [])]
            if (
                len(frame) != int(record["row_count"])
                or dataset_content_digest(frame) != expected_digest
                or actual_ids != list(record.get("selected_question_ids", []))
            ):
                snapshot_problems.append(f"{record['selection_id']}: snapshot content/identity mismatch")
    if snapshot_problems:
        raise ResultSetError("Run input snapshot validation failed:\n- " + "\n- ".join(snapshot_problems))
    prompt = manifest["payload"].get("prompts", {})
    prompt_root = run_dir / prompt.get("run_snapshot_path", "") / prompt.get("version", "")
    declared_prompts = {f"{name}.txt" for name in prompt.get("files", {})}
    actual_prompts = {path.name for path in prompt_root.glob("*.txt")} if prompt_root.is_dir() else set()
    if actual_prompts != declared_prompts:
        raise ResultSetError("Run prompt snapshot is missing files or contains unexpected templates.")
    for name, item in prompt.get("files", {}).items():
        path = prompt_root / f"{name}.txt"
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ResultSetError(f"Run prompt snapshot is unreadable: {path}: {exc}") from exc
        if content != item.get("content") or integrity_digest(content) != item.get("sha256"):
            raise ResultSetError(f"Run prompt snapshot content mismatch: {path}")
    declared = {Path(c["result_path"]) for c in conditions.values()}
    results_dir = run_dir / "results"
    actual = {p.relative_to(run_dir) for p in results_dir.glob("*.csv")} if results_dir.exists() else set()
    unexpected = actual - declared
    if unexpected:
        raise ResultSetError(f"Unexpected result artifacts: {sorted(map(str, unexpected))}")

    frames: list[pd.DataFrame] = []
    problems: list[str] = []
    selections = {d["selection_id"]: d for d in manifest["payload"].get("datasets", [])}
    models = {item["model_id"]: item for item in manifest["payload"].get("models", [])}
    methods = {item["method_id"]: item for item in manifest["payload"].get("methods", [])}
    for condition_id in sorted(conditions):
        condition = conditions[condition_id]
        status = state["conditions"][condition_id].get("status")
        path = run_dir / condition["result_path"]
        if "gate_path" in condition:
            gate_path = run_dir / condition["gate_path"]
            try:
                gate = json.loads(gate_path.read_text())
                gate_digest = gate.pop("artifact_digest")
                if gate_digest != integrity_digest(gate):
                    raise ValueError("digest mismatch")
                if gate.get("condition_id") != condition_id or gate.get("experiment_id") != manifest["experiment_id"]:
                    raise ValueError("identity mismatch")
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                problems.append(f"{condition_id}: invalid modal-k gate artifact: {exc}")
        if status == "gated":
            if path.exists():
                problems.append(f"{condition_id}: gated condition has a result artifact")
            continue
        if status == "failed":
            # Failed conditions are evaluable as accounting entries (no metrics);
            # a failed condition must not leave a result artifact behind.
            if path.exists():
                problems.append(
                    f"{condition_id}: failed condition has a result artifact; "
                    "resume the run to re-verify it or use --reset-run"
                )
            continue
        if status != "completed":
            problems.append(
                f"{condition_id}: run has not finished (status={status!r}); "
                "complete, resume, or reset the run before evaluating"
            )
            continue
        if not path.exists():
            problems.append(f"{condition_id}: missing {condition['result_path']}")
            continue
        try:
            artifact = validate_result_artifact(path)
            df = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError, RuntimeError) as exc:
            problems.append(f"{condition_id}: invalid result artifact: {exc}")
            continue
        if artifact.get("condition_id") != condition_id or artifact.get("row_count") != len(df):
            problems.append(f"{condition_id}: result metadata identity/row count mismatch")
        if state["conditions"][condition_id].get("result_sha256") != artifact.get("file_sha256"):
            problems.append(f"{condition_id}: run-state result digest mismatch")
        for column, expected in (
            ("condition_id", condition_id),
            ("experiment_id", manifest["experiment_id"]),
            ("dataset_selection_id", condition["selection_id"]),
            ("model_id", condition["model_id"]),
            ("method_id", condition["method_id"]),
            ("prompt_id", condition["prompt_id"]),
            ("dataset_artifact_id", condition["artifact_id"]),
            ("benchmark_split", condition["split"]),
            ("benchmark_name", condition["benchmark_name"]),
            ("model_name", models.get(condition["model_id"], {}).get("config", {}).get("model_name_or_path")),
            ("method_name", methods.get(condition["method_id"], {}).get("config", {}).get("name")),
        ):
            # Fail closed on missing/null identity values: a row whose ownership
            # metadata is absent must be an error, never silently excluded.
            if (
                expected is None
                or column not in df
                or df[column].isna().any()
                or set(df[column].astype(str)) != {str(expected)}
            ):
                problems.append(f"{condition_id}: invalid {column}")
        expected_qids = selections[condition["selection_id"]]["selected_question_ids"]
        # Modal-k methods may intentionally evaluate a manifest-recorded subset;
        # gate counts remain in rows, so require uniqueness/subset rather than equality.
        actual_qids = [str(v) for v in df.get("question_id", [])]
        if len(actual_qids) != len(set(actual_qids)) or not set(actual_qids) <= set(expected_qids):
            problems.append(f"{condition_id}: question IDs conflict with declared selection")
        elif "gate_n_evaluated" in df and df["gate_n_evaluated"].notna().any():
            counts = set(df["gate_n_evaluated"].dropna().astype(int))
            if len(counts) != 1 or len(actual_qids) != next(iter(counts)):
                problems.append(f"{condition_id}: modal-k result row count is incomplete")
        elif set(actual_qids) != set(expected_qids):
            problems.append(f"{condition_id}: result is missing declared question rows")
        frames.append(df)
    if problems:
        raise ResultSetError("Result-set validation failed:\n- " + "\n- ".join(problems))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), manifest
