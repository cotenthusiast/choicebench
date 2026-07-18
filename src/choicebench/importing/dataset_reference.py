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
    artifact_payload: Mapping[str, Any]
    selection_id: str
    selection_digest: str
    selection_payload: Mapping[str, Any]
    selection_semantics: Mapping[str, Any]
    selection_unknown_reasons: Mapping[str, str]
    identity_mode: Literal["imported_semantic_fallback", "native_compatibility"]
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
_NATIVE_DATASET_SPEC_KEYS = {
    "benchmark",
    "split",
    "hf_path",
    "hf_subset",
    "source_revision",
    "normalization_version",
    "transforms",
    "output_name",
}
_SNAPSHOT_RECORD_KEYS = {
    "schema_version",
    "dataset_id",
    "benchmark_name",
    "split",
    "artifact_id",
    "artifact_digest",
    "artifact_payload",
    "selection_id",
    "selection_digest",
    "selection_payload",
    "selection_semantics",
    "selection_unknown_reasons",
    "identity_mode",
    "selected_question_ids",
    "question_set_digest",
    "snapshot_digest",
    "reference_kind",
    "trust_label",
    "derivation",
    "derivation_digest",
    "limitations",
    "run_snapshot_path",
    "reference_metadata_path",
    "snapshot_sha256",
    "row_count",
    "record_digest",
}


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
            source_indices: set[int] = set()
            for index, item in enumerate(choices):
                if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
                    raise DatasetReferenceError(
                        f"Expected dataset source {source_id!r} has malformed ordered options "
                        f"for question_id {question_id!r}."
                    )
                raw_source_index = item.get("source_index", index)
                try:
                    if isinstance(raw_source_index, bool):
                        raise ValueError
                    source_index = int(raw_source_index)
                except (TypeError, ValueError) as exc:
                    raise DatasetReferenceError(
                        f"Expected dataset source {source_id!r} has an invalid source_index "
                        f"for question_id {question_id!r}."
                    ) from exc
                if source_index < 0 or source_index in source_indices:
                    raise DatasetReferenceError(
                        f"Expected dataset source {source_id!r} has an invalid source_index "
                        f"for question_id {question_id!r}."
                    )
                if not item["text"].strip():
                    raise DatasetReferenceError(
                        f"Expected dataset source {source_id!r} has an empty structured "
                        f"option for question_id {question_id!r}."
                    )
                source_indices.add(source_index)
                normalized_choices.append(
                    {
                        "text": item["text"],
                        "source_index": source_index,
                    }
                )
        else:
            choice_values = [str(raw_row[mapping[key]]) for key in choice_keys]
            nonempty_indices = [
                index for index, value in enumerate(choice_values) if value.strip()
            ]
            if nonempty_indices:
                last_nonempty = nonempty_indices[-1]
                if any(not value.strip() for value in choice_values[:last_nonempty]):
                    raise DatasetReferenceError(
                        f"Expected dataset source {source_id!r} has an empty option followed "
                        f"by a populated option for question_id {question_id!r}."
                    )
                choice_values = choice_values[: last_nonempty + 1]
            normalized_choices = [
                {"text": value, "source_index": index}
                for index, value in enumerate(choice_values)
                if value.strip()
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


def _fallback_artifact_payload(
    frame: pd.DataFrame, declaration: DatasetReferenceSpec
) -> dict[str, Any]:
    return {
        "schema_version": "choicebench.semantic-dataset.v1",
        "benchmark": declaration.benchmark_name,
        "split": declaration.split,
        "content_digest": dataset_content_digest(frame),
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


def _validated_identity_records(
    frame: pd.DataFrame, declaration: DatasetReferenceSpec
) -> tuple[
    Mapping[str, Any],
    str,
    str,
    Mapping[str, Any],
    str,
    str,
    Literal["imported_semantic_fallback", "native_compatibility"],
]:
    native = declaration.native_compatibility_identity
    if native is None:
        artifact_payload = canonicalize(_fallback_artifact_payload(frame, declaration))
        artifact_digest = integrity_digest(artifact_payload)
        artifact_id = short_id("ds", artifact_payload)
        selection_payload = canonicalize(
            _selection_payload(frame, declaration, artifact_id)
        )
        return (
            artifact_payload,
            artifact_digest,
            artifact_id,
            selection_payload,
            integrity_digest(selection_payload),
            short_id("sel", selection_payload),
            "imported_semantic_fallback",
        )

    required = {
        "artifact_payload",
        "artifact_digest",
        "artifact_id",
        "selection_payload",
        "selection_digest",
        "selection_id",
    }
    if not isinstance(native, Mapping) or set(native) != required:
        raise DatasetReferenceError(
            "Expected dataset native compatibility identity has invalid fields."
        )
    artifact_payload = canonicalize(native["artifact_payload"])
    selection_payload = canonicalize(native["selection_payload"])
    artifact_digest = integrity_digest(artifact_payload)
    artifact_id = short_id("ds", artifact_payload)
    selection_digest = integrity_digest(selection_payload)
    selection_id = short_id("sel", selection_payload)
    if (
        native["artifact_digest"] != artifact_digest
        or native["artifact_id"] != artifact_id
        or native["selection_digest"] != selection_digest
        or native["selection_id"] != selection_id
    ):
        raise DatasetReferenceError(
            "Expected dataset native compatibility identity claims are invalid."
        )
    if not isinstance(artifact_payload, Mapping) or set(artifact_payload) != {
        "spec",
        "content_digest",
        "source",
    }:
        raise DatasetReferenceError(
            "Expected dataset native compatibility artifact payload is invalid."
        )
    spec = artifact_payload.get("spec")
    if (
        not isinstance(spec, Mapping)
        or set(spec) != _NATIVE_DATASET_SPEC_KEYS
        or not isinstance(artifact_payload.get("source"), Mapping)
        or spec.get("benchmark") != declaration.benchmark_name
        or spec.get("split") != declaration.split
        or artifact_payload.get("content_digest") != dataset_content_digest(frame)
    ):
        raise DatasetReferenceError(
            "Expected dataset native compatibility content does not own the selected rows."
        )
    expected_selection = _selection_payload(frame, declaration, artifact_id)
    if selection_payload != canonicalize(expected_selection):
        raise DatasetReferenceError(
            "Expected dataset native compatibility selection does not own the selected rows."
        )
    return (
        artifact_payload,
        artifact_digest,
        artifact_id,
        selection_payload,
        selection_digest,
        selection_id,
        "native_compatibility",
    )


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

    unknown_reasons = canonicalize(dict(declaration.selection_unknown_reasons))
    expected_unknown_keys = {
        field
        for field, value in (
            ("selection_seed", declaration.selection_seed),
            ("selection_n_samples", declaration.selection_n_samples),
        )
        if value is None
    }
    if set(unknown_reasons) != expected_unknown_keys or not all(
        isinstance(reason, str) and reason.strip()
        for reason in unknown_reasons.values()
    ):
        raise DatasetReferenceError(
            "Expected dataset selection unknown reasons do not match null semantics."
        )
    (
        artifact_payload,
        artifact_digest,
        artifact_id,
        selection_payload,
        selection_digest,
        selection_id,
        identity_mode,
    ) = _validated_identity_records(frame, declaration)
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
            "selection_unknown_reasons": unknown_reasons,
            "identity_mode": identity_mode,
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
        artifact_payload=artifact_payload,
        selection_id=selection_id,
        selection_digest=selection_digest,
        selection_payload=selection_payload,
        selection_semantics=selection_semantics,
        selection_unknown_reasons=unknown_reasons,
        identity_mode=identity_mode,
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


def _contained_path(root: Path, value: Any, field: str) -> Path:
    relative = _safe_relative_path(value, field)
    try:
        canonical_root = root.resolve(strict=True)
    except OSError as exc:
        raise DatasetReferenceError("Expected dataset output root is unreadable.") from exc
    candidate = canonical_root / relative
    current = canonical_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise DatasetReferenceError(
                f"Expected dataset {field} contains an unsafe symlink."
            )
    try:
        candidate.resolve(strict=False).relative_to(canonical_root)
    except (OSError, ValueError) as exc:
        raise DatasetReferenceError(f"Expected dataset {field} is unsafe.") from exc
    return candidate


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
        "artifact_payload": dataset.artifact_payload,
        "selection_id": dataset.selection_id,
        "selection_digest": dataset.selection_digest,
        "selection_payload": dataset.selection_payload,
        "selection_semantics": dataset.selection_semantics,
        "selection_unknown_reasons": dataset.selection_unknown_reasons,
        "identity_mode": dataset.identity_mode,
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
    snapshot_path = _contained_path(root, record["run_snapshot_path"], "run_snapshot_path")
    metadata_path = _contained_path(
        root,
        record["reference_metadata_path"], "reference_metadata_path"
    )
    atomic_write_text(snapshot_path, dataset.frame.to_csv(index=False, lineterminator="\n"))
    atomic_write_json(metadata_path, record)
    return record


def _validate_snapshot_record_shape(record: Mapping[str, Any]) -> None:
    if set(record) != _SNAPSHOT_RECORD_KEYS:
        raise DatasetReferenceError("Expected dataset reference record fields are invalid.")
    if record["schema_version"] != _SNAPSHOT_SCHEMA:
        raise DatasetReferenceError("Expected dataset reference schema is unsupported.")
    nonempty_strings = {
        "dataset_id",
        "benchmark_name",
        "split",
        "artifact_id",
        "artifact_digest",
        "selection_id",
        "selection_digest",
        "question_set_digest",
        "snapshot_digest",
        "trust_label",
        "derivation_digest",
        "run_snapshot_path",
        "reference_metadata_path",
        "snapshot_sha256",
        "record_digest",
    }
    if any(
        not isinstance(record[field], str) or not record[field]
        for field in nonempty_strings
    ):
        raise DatasetReferenceError("Expected dataset reference record fields are invalid.")
    if record["reference_kind"] not in {
        "independent_input_snapshot",
        "profile_derived_reference_snapshot",
    } or record["identity_mode"] not in {
        "imported_semantic_fallback",
        "native_compatibility",
    }:
        raise DatasetReferenceError("Expected dataset reference record fields are invalid.")
    if not all(
        isinstance(record[field], Mapping)
        for field in (
            "artifact_payload",
            "selection_payload",
            "selection_semantics",
            "selection_unknown_reasons",
            "derivation",
        )
    ):
        raise DatasetReferenceError("Expected dataset reference record fields are invalid.")
    selected = record["selected_question_ids"]
    limitations = record["limitations"]
    if (
        not isinstance(selected, list)
        or not selected
        or not all(isinstance(value, str) and value for value in selected)
        or len(selected) != len(set(selected))
        or not isinstance(limitations, list)
        or not all(isinstance(value, str) for value in limitations)
        or isinstance(record["row_count"], bool)
        or not isinstance(record["row_count"], int)
        or record["row_count"] < 1
    ):
        raise DatasetReferenceError("Expected dataset reference record fields are invalid.")


def validate_expected_snapshot(run_dir: Path, record: Mapping[str, Any]) -> None:
    """Recompute every snapshot, semantic identity, and reference digest."""
    _validate_snapshot_record_shape(record)
    root = Path(run_dir)
    raw_record = dict(record)
    claimed_digest = raw_record.pop("record_digest", None)
    if claimed_digest != integrity_digest(raw_record):
        raise DatasetReferenceError("Expected dataset reference record integrity failed.")
    snapshot_path = _contained_path(root, record["run_snapshot_path"], "run_snapshot_path")
    metadata_path = _contained_path(
        root, record["reference_metadata_path"], "reference_metadata_path"
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

    artifact_payload = canonicalize(record["artifact_payload"])
    if record["identity_mode"] == "imported_semantic_fallback":
        expected_artifact_payload = {
            "schema_version": "choicebench.semantic-dataset.v1",
            "benchmark": record["benchmark_name"],
            "split": record["split"],
            "content_digest": dataset_content_digest(frame),
        }
        if artifact_payload != expected_artifact_payload:
            raise DatasetReferenceError("Expected dataset artifact identity is invalid.")
    else:
        if not isinstance(artifact_payload, Mapping) or set(artifact_payload) != {
            "spec",
            "content_digest",
            "source",
        }:
            raise DatasetReferenceError("Expected dataset artifact identity is invalid.")
        native_spec = artifact_payload["spec"]
        if (
            not isinstance(native_spec, Mapping)
            or set(native_spec) != _NATIVE_DATASET_SPEC_KEYS
            or not isinstance(artifact_payload["source"], Mapping)
            or native_spec.get("benchmark") != record["benchmark_name"]
            or native_spec.get("split") != record["split"]
            or artifact_payload["content_digest"] != dataset_content_digest(frame)
        ):
            raise DatasetReferenceError("Expected dataset artifact identity is invalid.")
    artifact_digest = integrity_digest(artifact_payload)
    artifact_id = short_id("ds", artifact_payload)
    if artifact_digest != record.get("artifact_digest") or artifact_id != record.get("artifact_id"):
        raise DatasetReferenceError("Expected dataset artifact identity is invalid.")
    selection_semantics = record["selection_semantics"]
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
    if canonicalize(record["selection_payload"]) != selection_payload:
        raise DatasetReferenceError("Expected dataset selection identity is invalid.")
    unknown_reasons = record["selection_unknown_reasons"]
    expected_unknown_keys = {
        field
        for field, value in (
            ("selection_seed", selection_semantics["seed"]),
            ("selection_n_samples", selection_semantics["n_samples"]),
        )
        if value is None
    }
    if set(unknown_reasons) != expected_unknown_keys or not all(
        isinstance(reason, str) and reason.strip()
        for reason in unknown_reasons.values()
    ):
        raise DatasetReferenceError("Expected dataset selection semantics are invalid.")
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
