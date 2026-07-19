# src/choicebench/io/readers.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
import json

import pandas as pd
from choicebench.manifest import (
    RUN_STATE_FILENAME,
    ManifestCompatibilityError,
    validate_manifest,
    validate_run_state,
)
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


@dataclass(frozen=True)
class RealizationSelection:
    policy: Literal["single_evaluable_per_condition", "explicit"]
    realization_ids: tuple[str, ...]


@dataclass(frozen=True)
class ManifestResultSet:
    rows: pd.DataFrame
    manifest: Mapping[str, Any]
    state: Mapping[str, Any]
    selection: RealizationSelection
    verified_realizations: Mapping[str, Any]


def read_manifest_result_set(
    run_dir: Path, *, realization_ids: Sequence[str] | None = None
) -> ManifestResultSet:
    """Read a v3 imported/repaired run's result rows through the shared
    full-graph verifier (never a bare CSV read): unselected/ineligible
    realizations remain visible via verified_realizations for accounting,
    but never concatenated into rows. Auto-selection only applies when a
    condition has at most one eligible realization; ambiguity is a refusal.
    """
    from choicebench.importing.engine import verify_import_run

    run_dir = Path(run_dir)
    verified = verify_import_run(run_dir)
    manifest = verified.manifest
    state = json.loads((run_dir / RUN_STATE_FILENAME).read_text())

    by_condition: dict[str, list[str]] = {}
    for realization_id, record in manifest["payload"]["realizations"].items():
        by_condition.setdefault(record["condition_id"], []).append(realization_id)

    def _eligible(realization_id: str) -> bool:
        return verified.realizations[realization_id].result_sha256 is not None

    if realization_ids is None:
        selected: list[str] = []
        for condition_id, ids in by_condition.items():
            eligible_ids = sorted(rid for rid in ids if _eligible(rid))
            if len(eligible_ids) > 1:
                raise ResultSetError(
                    f"Condition {condition_id!r} has multiple eligible realizations "
                    f"{eligible_ids}; pass an explicit realization_ids selection."
                )
            selected.extend(eligible_ids)
        policy: Literal["single_evaluable_per_condition", "explicit"] = (
            "single_evaluable_per_condition"
        )
    else:
        unknown = sorted(set(realization_ids) - set(manifest["payload"]["realizations"]))
        if unknown:
            raise ResultSetError(f"Unknown realization ID(s) {unknown}.")
        ineligible = sorted(rid for rid in realization_ids if not _eligible(rid))
        if ineligible:
            raise ResultSetError(f"Realization(s) {ineligible} have no result to evaluate.")
        selected = sorted(set(realization_ids))
        policy = "explicit"

    frames = [
        pd.read_csv(run_dir / f"results/{realization_id}.csv", dtype={"question_id": "string"})
        for realization_id in selected
    ]
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return ManifestResultSet(
        rows=rows,
        manifest=manifest,
        state=state,
        selection=RealizationSelection(policy=policy, realization_ids=tuple(selected)),
        verified_realizations=verified.realizations,
    )


def build_evaluation_report_for_result_set(
    run_id: str, result_set: ManifestResultSet, *, reparse: bool
) -> dict[str, Any]:
    """Reduced-scope evaluation-v2 report: per-condition realization lists,
    per-realization selection/status/scope accounting, and accuracy for
    selected evaluable realizations. Every realization -- selected or not --
    is accounted for; only selected ones contribute metrics or rows."""
    manifest = result_set.manifest
    conditions: dict[str, dict[str, Any]] = {}
    realizations: dict[str, dict[str, Any]] = {}
    selected_ids = set(result_set.selection.realization_ids)

    for realization_id, realization in manifest["payload"]["realizations"].items():
        condition_id = realization["condition_id"]
        conditions.setdefault(condition_id, {"realization_ids": []})
        conditions[condition_id]["realization_ids"].append(realization_id)

        evidence = realization["identity"]["realization"]["evidence"]
        result_origin = realization["identity"]["realization"]["result_origin"]
        selected = realization_id in selected_ids
        entry: dict[str, Any] = {
            "selected": selected,
            "evidence_status": evidence["evidence_status"],
            "scope_disposition": evidence["scope_disposition"],
            "prediction_origins": result_origin["prediction_origins"],
            "metrics": {},
        }
        if selected:
            rows = result_set.rows[result_set.rows["realization_id"] == realization_id]
            if len(rows):
                predicted = rows["predicted_option"].astype(str).str.upper()
                correct = rows["correct_option"].astype(str).str.upper()
                entry["metrics"] = {
                    "accuracy": float((predicted == correct).mean()),
                    "n": int(len(rows)),
                }
        realizations[realization_id] = entry

    for condition in conditions.values():
        condition["realization_ids"] = sorted(condition["realization_ids"])

    return {
        "schema_version": "choicebench.evaluation.v2",
        "run_id": run_id,
        "selection_policy": result_set.selection.policy,
        "conditions": conditions,
        "realizations": realizations,
    }
