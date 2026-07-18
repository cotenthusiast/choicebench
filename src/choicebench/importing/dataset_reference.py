"""Trust-qualified expected-dataset snapshots for external result imports."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

import pandas as pd

from choicebench.datasets import (
    NORMALIZATION_VERSION,
    dataset_content_digest,
    dataset_sample_identities,
    validate_normalized_dataset,
)
from choicebench.identity import canonicalize, integrity_digest, short_id
from choicebench.importing.csv_adapter import OpenedSource
from choicebench.importing.schema import DatasetReferenceSpec
from choicebench.infra.artifacts import atomic_write_json, atomic_write_text
from choicebench.pipeline.options import build_option_map, correct_option_for_row


class DatasetReferenceError(ValueError):
    """Raised when an expected-dataset reference is incomplete or inconsistent."""


@dataclass(frozen=True)
class ExpectedDataset:
    dataset_id: str
    benchmark_name: str
    split: str
    artifact_id: str
    artifact_digest: str
    selection_id: str
    selection_digest: str
    selection_semantics: Mapping[str, Any]
    frame: pd.DataFrame
    selected_question_ids: tuple[str, ...]
    reference_kind: Literal[
        "independent_input_snapshot", "profile_derived_reference_snapshot"
    ]
    trust_label: str
    question_set_digest: str
    snapshot_digest: str
    derivation: Mapping[str, Any]
    derivation_digest: str
    limitations: tuple[str, ...]


_SNAPSHOT_SCHEMA = "choicebench.expected-dataset.v1"
_REFERENCE_SCHEMA = "choicebench.dataset-reference.v1"
_REQUIRED_FIELDS = ("question_id", "question_text", "correct_option")


def _source_frame(source_id: str, source: OpenedSource) -> pd.DataFrame:
    if source.source_id != source_id:
        raise DatasetReferenceError(
            f"Expected dataset source key {source_id!r} does not match opened source_id "
            f"{source.source_id!r}."
        )
    actual = sha256(source.data).hexdigest()
    if source.sha256 != actual:
        raise DatasetReferenceError(
            f"Expected dataset source {source_id!r} checksum does not match its bytes."
        )
    try:
        return pd.read_csv(
            BytesIO(source.data), dtype=str, keep_default_na=False, na_filter=False
        )
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetReferenceError(
            f"Expected dataset source {source_id!r} is not a valid UTF-8 CSV."
        ) from exc


def _choice_keys(columns: Mapping[str, str]) -> tuple[str, ...]:
    keys = tuple(
        sorted(
            (key for key in columns if key.startswith("choice_") and len(key) == 8),
            key=lambda key: key.removeprefix("choice_"),
        )
    )
    return keys


def _normalize_source(
    declaration: DatasetReferenceSpec, source_id: str, source: OpenedSource
) -> pd.DataFrame:
    raw = _source_frame(source_id, source)
    mapping = dict(declaration.columns)
    missing_common = sorted(set(_REQUIRED_FIELDS) - set(mapping))
    if missing_common:
        raise DatasetReferenceError(
            f"Expected dataset mapping is missing required fields {missing_common}."
        )
    missing_source = sorted(set(mapping.values()) - set(raw.columns))
    if missing_source:
        raise DatasetReferenceError(
            f"Expected dataset source {source_id!r} is missing mapped columns {missing_source}."
        )
    if len(mapping.values()) != len(set(mapping.values())):
        raise DatasetReferenceError("Expected dataset mapping reuses a source column.")

    records: list[dict[str, Any]] = []
    choice_keys = _choice_keys(mapping)
    has_structured_choices = "choices_json" in mapping
    if not choice_keys and not has_structured_choices:
        raise DatasetReferenceError("Expected dataset mapping declares no ordered options.")

    ordinary_keys = sorted(
        key for key in mapping if key not in choice_keys and key != "choices_json"
    )
    for row_number, raw_row in raw.iterrows():
        record = {key: raw_row[mapping[key]] for key in ordinary_keys}
        question_id = str(record["question_id"]).strip()
        if not question_id:
            raise DatasetReferenceError(
                f"Expected dataset source {source_id!r} has an empty question_id at row "
                f"{row_number + 2}."
            )
        record["question_id"] = question_id
        record["correct_option"] = str(record["correct_option"]).strip().upper()

        if has_structured_choices:
            raw_choices = raw_row[mapping["choices_json"]]
            try:
                choices = json.loads(raw_choices)
            except (json.JSONDecodeError, TypeError) as exc:
                raise DatasetReferenceError(
                    f"Expected dataset source {source_id!r} has malformed ordered options "
                    f"for question_id {question_id!r}."
                ) from exc
            if not isinstance(choices, list):
                raise DatasetReferenceError(
                    f"Expected dataset source {source_id!r} has malformed ordered options "
                    f"for question_id {question_id!r}."
                )
            normalized_choices = []
            for index, item in enumerate(choices):
                if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
                    raise DatasetReferenceError(
                        f"Expected dataset source {source_id!r} has malformed ordered options "
                        f"for question_id {question_id!r}."
                    )
                normalized_choices.append(
                    {
                        "text": item["text"],
                        "source_index": int(item.get("source_index", index)),
                    }
                )
        else:
            normalized_choices = [
                {"text": raw_row[mapping[key]], "source_index": index}
                for index, key in enumerate(choice_keys)
                if str(raw_row[mapping[key]]).strip()
            ]
        record["choices_json"] = json.dumps(
            normalized_choices, ensure_ascii=True, separators=(",", ":")
        )
        try:
            options = build_option_map(record)
            record["correct_option"] = correct_option_for_row(record, options)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DatasetReferenceError(
                f"Expected dataset source {source_id!r} has invalid ordered options or "
                f"correct_option for question_id {question_id!r}: {exc}"
            ) from exc
        records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise DatasetReferenceError(f"Expected dataset source {source_id!r} has no rows.")
    duplicated = frame["question_id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(frame.loc[duplicated, "question_id"].astype(str).unique())
        raise DatasetReferenceError(
            f"Expected dataset source {source_id!r} has duplicate question_id values "
            f"{duplicate_ids}."
        )
    return frame


def _selected_frame(
    frame: pd.DataFrame, expected_question_ids: tuple[str, ...]
) -> pd.DataFrame:
    if len(expected_question_ids) != len(set(expected_question_ids)):
        duplicates = sorted(
            question_id
            for question_id in set(expected_question_ids)
            if expected_question_ids.count(question_id) > 1
        )
        raise DatasetReferenceError(
            f"Expected dataset declaration has duplicate selected question ID(s) {duplicates}."
        )
    indexed = frame.set_index(frame["question_id"].astype(str), drop=False)
    missing = [question_id for question_id in expected_question_ids if question_id not in indexed.index]
    if missing:
        raise DatasetReferenceError(
            f"Expected dataset is missing selected question ID(s) {missing}."
        )
    selected = indexed.loc[list(expected_question_ids)].reset_index(drop=True)
    return selected


def _semantic_disagreement(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    for left_row, right_row in zip(
        left.to_dict("records"), right.to_dict("records"), strict=True
    ):
        question_id = str(left_row["question_id"])
        if left_row["question_text"] != right_row["question_text"]:
            return f"question_text disagreement for question_id {question_id!r}"
        if left_row["correct_option"] != right_row["correct_option"]:
            return f"correct_option disagreement for question_id {question_id!r}"
        if build_option_map(left_row) != build_option_map(right_row):
            return f"ordered options disagreement for question_id {question_id!r}"
    return None


def _profile_groups(declaration: DatasetReferenceSpec) -> tuple[tuple[str, ...], ...]:
    raw = declaration.derivation.get("independent_source_groups")
    if not isinstance(raw, (list, tuple)):
        raise DatasetReferenceError(
            "Profile-derived reference requires at least two independent source groups."
        )
    groups: list[tuple[str, ...]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or not item:
            raise DatasetReferenceError(
                "Profile-derived reference requires at least two independent source groups."
            )
        groups.append(tuple(str(source_id) for source_id in item))
    if len(groups) < 2:
        raise DatasetReferenceError(
            "Profile-derived reference requires at least two independent source groups."
        )
    flattened = [source_id for group in groups for source_id in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(declaration.source_ids):
        raise DatasetReferenceError(
            "Profile-derived reference independent source groups must be disjoint and "
            "cover every declared source."
        )
    return tuple(groups)


def _artifact_payload(dataset: ExpectedDataset | pd.DataFrame, declaration: DatasetReferenceSpec) -> dict[str, Any]:
    frame = dataset.frame if isinstance(dataset, ExpectedDataset) else dataset
    return {
        "spec": {
            "benchmark": declaration.benchmark_name,
            "split": declaration.split,
            "hf_path": None,
            "hf_subset": None,
            "source_revision": None,
            "normalization_version": NORMALIZATION_VERSION,
            "transforms": [],
            "output_name": declaration.benchmark_name,
        },
        "content_digest": dataset_content_digest(frame),
        "source": {},
    }


def _selection_payload(
    frame: pd.DataFrame, declaration: DatasetReferenceSpec, artifact_id: str
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "content_digest": dataset_content_digest(frame),
        "sample_identities": dataset_sample_identities(frame),
        "seed": declaration.selection_seed,
        "n_samples": declaration.selection_n_samples,
        "subject_filter": sorted(declaration.subject_filter),
    }


def build_expected_dataset(
    declaration: DatasetReferenceSpec,
    opened_sources: Mapping[str, OpenedSource],
) -> ExpectedDataset:
    """Build one semantic snapshot plus its trust-qualified reference record."""
    missing_sources = sorted(set(declaration.source_ids) - set(opened_sources))
    if missing_sources:
        raise DatasetReferenceError(
            f"Expected dataset declaration is missing opened source(s) {missing_sources}."
        )
    if declaration.selection_source_id not in declaration.source_ids:
        raise DatasetReferenceError("selection_source_id is not a declared dataset source.")

    normalized = {
        source_id: _normalize_source(declaration, source_id, opened_sources[source_id])
        for source_id in declaration.source_ids
    }
    selected = {
        source_id: _selected_frame(frame, declaration.expected_question_ids)
        for source_id, frame in normalized.items()
    }
    if declaration.reference_kind == "profile_derived_reference_snapshot":
        _profile_groups(declaration)
        baseline = selected[declaration.selection_source_id]
        for source_id in declaration.source_ids:
            disagreement = _semantic_disagreement(baseline, selected[source_id])
            if disagreement is not None:
                raise DatasetReferenceError(
                    f"Profile-derived reference source {source_id!r} has {disagreement}."
                )

    frame = selected[declaration.selection_source_id].copy()
    try:
        validate_normalized_dataset(frame, source="expected dataset reference")
    except Exception as exc:
        raise DatasetReferenceError(f"Expected dataset reference is invalid: {exc}") from exc

    artifact_payload = _artifact_payload(frame, declaration)
    artifact_digest = integrity_digest(artifact_payload)
    artifact_id = short_id("ds", artifact_payload)
    selection_payload = _selection_payload(frame, declaration, artifact_id)
    selection_digest = integrity_digest(selection_payload)
    selection_id = short_id("sel", selection_payload)
    selection_semantics = canonicalize(
        {
            "seed": declaration.selection_seed,
            "n_samples": declaration.selection_n_samples,
            "subject_filter": sorted(declaration.subject_filter),
        }
    )

    limitations = list(declaration.limitations)
    if declaration.revision is None and "publisher revision was not recorded" not in limitations:
        limitations.append("publisher revision was not recorded")
    if (
        declaration.reference_kind == "profile_derived_reference_snapshot"
        and "not independently authenticated" not in limitations
    ):
        limitations.append("not independently authenticated")

    source_chain = [
        {
            "source_id": source_id,
            "logical_path": opened_sources[source_id].logical_path,
            "sha256": opened_sources[source_id].sha256,
        }
        for source_id in declaration.source_ids
    ]
    derivation = canonicalize(
        {
            "schema_version": _REFERENCE_SCHEMA,
            "dataset_id": declaration.dataset_id,
            "reference_kind": declaration.reference_kind,
            "trust_label": declaration.trust_label,
            "source_chain": source_chain,
            "selection_source_id": declaration.selection_source_id,
            "columns": dict(declaration.columns),
            "revision": declaration.revision,
            "fingerprint": declaration.fingerprint,
            "declared_derivation": dict(declaration.derivation),
            "selected_question_ids": list(declaration.expected_question_ids),
            "selection_semantics": selection_semantics,
            "limitations": limitations,
        }
    )
    derivation_digest = integrity_digest(derivation)
    question_set_digest = integrity_digest(list(declaration.expected_question_ids))
    snapshot_digest = integrity_digest(
        {
            "artifact_digest": artifact_digest,
            "selection_digest": selection_digest,
            "question_set_digest": question_set_digest,
            "derivation_digest": derivation_digest,
            "reference_kind": declaration.reference_kind,
            "trust_label": declaration.trust_label,
        }
    )
    return ExpectedDataset(
        dataset_id=declaration.dataset_id,
        benchmark_name=declaration.benchmark_name,
        split=declaration.split,
        artifact_id=artifact_id,
        artifact_digest=artifact_digest,
        selection_id=selection_id,
        selection_digest=selection_digest,
        selection_semantics=selection_semantics,
        frame=frame,
        selected_question_ids=tuple(declaration.expected_question_ids),
        reference_kind=declaration.reference_kind,
        trust_label=declaration.trust_label,
        question_set_digest=question_set_digest,
        snapshot_digest=snapshot_digest,
        derivation=derivation,
        derivation_digest=derivation_digest,
        limitations=tuple(limitations),
    )


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str):
        raise DatasetReferenceError(f"Expected dataset {field} must be a relative path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise DatasetReferenceError(f"Expected dataset {field} is unsafe.")
    return Path(*pure.parts)


def _snapshot_record(dataset: ExpectedDataset) -> dict[str, Any]:
    snapshot_path = f"artifacts/datasets/{dataset.selection_id}.csv"
    metadata_path = f"artifacts/datasets/{dataset.selection_id}.reference.json"
    csv_bytes = dataset.frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    record: dict[str, Any] = {
        "schema_version": _SNAPSHOT_SCHEMA,
        "dataset_id": dataset.dataset_id,
        "benchmark_name": dataset.benchmark_name,
        "split": dataset.split,
        "artifact_id": dataset.artifact_id,
        "artifact_digest": dataset.artifact_digest,
        "selection_id": dataset.selection_id,
        "selection_digest": dataset.selection_digest,
        "selection_semantics": dataset.selection_semantics,
        "selected_question_ids": list(dataset.selected_question_ids),
        "question_set_digest": dataset.question_set_digest,
        "snapshot_digest": dataset.snapshot_digest,
        "reference_kind": dataset.reference_kind,
        "trust_label": dataset.trust_label,
        "derivation": dataset.derivation,
        "derivation_digest": dataset.derivation_digest,
        "limitations": list(dataset.limitations),
        "run_snapshot_path": snapshot_path,
        "reference_metadata_path": metadata_path,
        "snapshot_sha256": sha256(csv_bytes).hexdigest(),
        "row_count": len(dataset.frame),
    }
    record["record_digest"] = integrity_digest(record)
    return record


def write_expected_snapshot(staged_run: Path, dataset: ExpectedDataset) -> dict[str, Any]:
    """Atomically write the selected semantic CSV and its self-digesting record."""
    root = Path(staged_run)
    record = _snapshot_record(dataset)
    snapshot_path = root / _safe_relative_path(record["run_snapshot_path"], "run_snapshot_path")
    metadata_path = root / _safe_relative_path(
        record["reference_metadata_path"], "reference_metadata_path"
    )
    atomic_write_text(snapshot_path, dataset.frame.to_csv(index=False, lineterminator="\n"))
    atomic_write_json(metadata_path, record)
    return record


def validate_expected_snapshot(run_dir: Path, record: Mapping[str, Any]) -> None:
    """Recompute every snapshot, semantic identity, and reference digest."""
    root = Path(run_dir)
    raw_record = dict(record)
    claimed_digest = raw_record.pop("record_digest", None)
    if claimed_digest != integrity_digest(raw_record):
        raise DatasetReferenceError("Expected dataset reference record integrity failed.")
    snapshot_path = root / _safe_relative_path(record.get("run_snapshot_path"), "run_snapshot_path")
    metadata_path = root / _safe_relative_path(
        record.get("reference_metadata_path"), "reference_metadata_path"
    )
    try:
        persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise DatasetReferenceError("Expected dataset reference metadata is unreadable.") from exc
    if canonicalize(persisted) != canonicalize(dict(record)):
        raise DatasetReferenceError("Expected dataset reference metadata integrity failed.")
    try:
        snapshot_bytes = snapshot_path.read_bytes()
    except OSError as exc:
        raise DatasetReferenceError("Expected dataset snapshot is unreadable.") from exc
    if sha256(snapshot_bytes).hexdigest() != record.get("snapshot_sha256"):
        raise DatasetReferenceError("Expected dataset snapshot integrity failed.")
    try:
        frame = pd.read_csv(BytesIO(snapshot_bytes), dtype=str, keep_default_na=False, na_filter=False)
        validate_normalized_dataset(frame, source="expected dataset snapshot")
    except Exception as exc:
        raise DatasetReferenceError("Expected dataset snapshot integrity failed.") from exc
    selected_ids = tuple(frame["question_id"].astype(str))
    if selected_ids != tuple(record.get("selected_question_ids", ())):
        raise DatasetReferenceError("Expected dataset snapshot question ownership is invalid.")
    if len(frame) != record.get("row_count"):
        raise DatasetReferenceError("Expected dataset snapshot row count is invalid.")
    if integrity_digest(list(selected_ids)) != record.get("question_set_digest"):
        raise DatasetReferenceError("Expected dataset snapshot question-set identity is invalid.")
    derivation = record.get("derivation")
    if integrity_digest(derivation) != record.get("derivation_digest"):
        raise DatasetReferenceError("Expected dataset derivation integrity failed.")

    artifact_payload = {
        "spec": {
            "benchmark": record.get("benchmark_name"),
            "split": record.get("split"),
            "hf_path": None,
            "hf_subset": None,
            "source_revision": None,
            "normalization_version": NORMALIZATION_VERSION,
            "transforms": [],
            "output_name": record.get("benchmark_name"),
        },
        "content_digest": dataset_content_digest(frame),
        "source": {},
    }
    artifact_digest = integrity_digest(artifact_payload)
    artifact_id = short_id("ds", artifact_payload)
    if artifact_digest != record.get("artifact_digest") or artifact_id != record.get("artifact_id"):
        raise DatasetReferenceError("Expected dataset artifact identity is invalid.")
    selection_semantics = record.get("selection_semantics")
    if (
        not isinstance(selection_semantics, Mapping)
        or set(selection_semantics) != {"seed", "n_samples", "subject_filter"}
        or (
            selection_semantics["seed"] is not None
            and not isinstance(selection_semantics["seed"], int)
        )
        or (
            selection_semantics["n_samples"] is not None
            and not isinstance(selection_semantics["n_samples"], int)
        )
        or not isinstance(selection_semantics["subject_filter"], list)
        or not all(
            isinstance(subject, str)
            for subject in selection_semantics["subject_filter"]
        )
    ):
        raise DatasetReferenceError("Expected dataset selection semantics are invalid.")
    selection_payload = {
        "artifact_id": artifact_id,
        "content_digest": dataset_content_digest(frame),
        "sample_identities": dataset_sample_identities(frame),
        "seed": selection_semantics["seed"],
        "n_samples": selection_semantics["n_samples"],
        "subject_filter": selection_semantics["subject_filter"],
    }
    if (
        integrity_digest(selection_payload) != record.get("selection_digest")
        or short_id("sel", selection_payload) != record.get("selection_id")
    ):
        raise DatasetReferenceError("Expected dataset selection identity is invalid.")
    expected_snapshot_digest = integrity_digest(
        {
            "artifact_digest": record.get("artifact_digest"),
            "selection_digest": record.get("selection_digest"),
            "question_set_digest": record.get("question_set_digest"),
            "derivation_digest": record.get("derivation_digest"),
            "reference_kind": record.get("reference_kind"),
            "trust_label": record.get("trust_label"),
        }
    )
    if expected_snapshot_digest != record.get("snapshot_digest"):
        raise DatasetReferenceError("Expected dataset reference snapshot identity is invalid.")
