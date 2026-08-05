"""Split-addressed, content-verified prepared dataset artifacts."""

from __future__ import annotations

import json
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from choicebench.identity import canonicalize, integrity_digest, short_id
from choicebench.infra.atomic_io import atomic_write_json, atomic_write_text
from choicebench.infra.file_lock import FileLock

DATASET_ARTIFACT_SCHEMA_VERSION = "choicebench.dataset.v2"
NORMALIZATION_VERSION = "2"


class DatasetArtifactError(RuntimeError):
    """Raised when prepared data cannot be verified against its metadata."""


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return value or "unnamed"


@dataclass(frozen=True)
class DatasetSpec:
    benchmark: str
    split: str
    hf_path: str | None = None
    hf_subset: str | None = None
    source_revision: str | None = None
    normalization_version: str = NORMALIZATION_VERSION
    transforms: tuple[str, ...] = ()
    output_name: str | None = None

    def identity_payload(self) -> dict[str, Any]:
        return canonicalize({
            "benchmark": self.benchmark,
            "split": self.split,
            "hf_path": self.hf_path,
            "hf_subset": self.hf_subset,
            "source_revision": self.source_revision,
            "normalization_version": self.normalization_version,
            "transforms": list(self.transforms),
            "output_name": self.output_name,
        })


@dataclass(frozen=True)
class PreparedDataset:
    path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    dataframe: pd.DataFrame

    @property
    def artifact_id(self) -> str:
        return str(self.metadata["artifact_id"])

    @property
    def content_digest(self) -> str:
        return str(self.metadata["content_digest"])

    @property
    def sample_identities(self) -> tuple[str, ...]:
        return tuple(dataset_sample_identities(self.dataframe))


def spec_for_benchmark(config: Any, *, split: str | None = None) -> DatasetSpec:
    """Build the exact prepared-artifact spec for a benchmark config."""
    from choicebench.benchmarks.registry import BENCHMARK_REGISTRY, get_by_hf_path
    name = config.name
    resolved_split = split or config.split
    if name == "toy":
        return DatasetSpec("toy", resolved_split, output_name="toy")
    if name == "huggingface":
        entry = get_by_hf_path(config.hf_path, config.hf_subset)
        output_name = (
            config.output_name
            or (entry.name if entry is not None else config.hf_path.rsplit("/", 1)[-1].lower().replace("-", "_"))
        )
        return DatasetSpec(
            entry.name if entry is not None else "huggingface",
            resolved_split, config.hf_path, config.hf_subset,
            getattr(config, "source_revision", None),
            transforms=tuple(getattr(config, "transforms", [])), output_name=output_name,
        )
    entry = BENCHMARK_REGISTRY[name]
    return DatasetSpec(
        name, resolved_split, entry.hf_path, entry.hf_subset,
        getattr(config, "source_revision", None),
        transforms=tuple(getattr(config, "transforms", [])), output_name=name,
    )


def artifact_dir(processed_dir: Path, spec: DatasetSpec) -> Path:
    logical_name = spec.output_name or spec.benchmark
    source_id = short_id("src", spec.identity_payload(), 12)
    return (
        Path(processed_dir) / _slug(logical_name) / _slug(spec.split)
        / f"norm-v{_slug(spec.normalization_version)}" / source_id
    )


def artifact_csv_path(processed_dir: Path, spec: DatasetSpec) -> Path:
    return artifact_dir(processed_dir, spec) / "normalized.csv"


def artifact_metadata_path(csv_path: Path) -> Path:
    return Path(csv_path).with_name("artifact.json")


def artifact_stats_path(csv_path: Path) -> Path:
    return Path(csv_path).with_name("stats.json")


def _cell(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    # Runtime consumes ordinary text fields verbatim. Structured parsing is
    # deliberately handled by canonical_dataset_rows only for *_json columns.
    if isinstance(value, str):
        return value
    return canonicalize(value, redact_secrets=False)


def canonical_dataset_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Canonical semantic rows, preserving dataset order and every column."""
    columns = sorted(str(c) for c in df.columns)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        item: dict[str, Any] = {}
        for column in columns:
            value = _cell(row[column])
            if column.endswith("_json") and isinstance(value, str):
                try:
                    value = canonicalize(json.loads(value), redact_secrets=False)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            item[column] = value
        rows.append(item)
    return rows


def dataset_content_digest(df: pd.DataFrame) -> str:
    # Do not apply generic identity normalization or secret-key redaction here:
    # this digest binds the exact scientific text and values runtime consumes.
    encoded = json.dumps(
        canonical_dataset_rows(df), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _semantic_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    return " ".join(text.split()).casefold()


def _semantic_choices(row: dict[str, Any]) -> list[str]:
    raw = row.get("choices_json")
    choices: list[Any] = []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    choices.append(item.get("text") if isinstance(item, dict) else item)
        except json.JSONDecodeError:
            pass
    if not choices:
        choices = [row[key] for key in sorted(row) if re.fullmatch(r"choice_[a-z]", key)]
    return sorted(_semantic_text(choice) for choice in choices)


def dataset_sample_identities(df: pd.DataFrame) -> list[str]:
    """Leakage identities over normalized stem and option content, label-free."""
    identities: list[str] = []
    for row in canonical_dataset_rows(df):
        payload = {
            "question": _semantic_text(row.get("question_text")),
            # Choice order and labels are presentation transforms; content is not.
            "choices": _semantic_choices(row),
        }
        identities.append(integrity_digest(payload))
    return identities


def validate_normalized_dataset(df: pd.DataFrame, *, source: str = "prepared dataset") -> None:
    required = {"question_id", "question_text", "correct_option"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise DatasetArtifactError(f"{source} is missing required normalized columns: {missing}.")
    ids = df["question_id"]
    if ids.isna().any() or any(not str(value).strip() for value in ids):
        raise DatasetArtifactError(f"{source} contains an empty question_id.")
    duplicated = ids.astype(str).duplicated(keep=False)
    if duplicated.any():
        examples = sorted(ids.astype(str)[duplicated].unique())[:3]
        raise DatasetArtifactError(
            f"{source} contains duplicate question_id values {examples}; resume and row-level "
            "provenance require unique IDs. Apply an explicit deduplication transform and re-prepare."
        )
    has_choices = "choices_json" in df.columns or any(re.fullmatch(r"choice_[a-z]", str(c)) for c in df.columns)
    if not has_choices:
        raise DatasetArtifactError(f"{source} has no choices_json or choice_a-style option columns.")


def write_prepared_dataset(
    df: pd.DataFrame,
    processed_dir: Path,
    spec: DatasetSpec,
    *,
    source_metadata: dict[str, Any] | None = None,
) -> PreparedDataset:
    lock_path = artifact_dir(processed_dir, spec) / ".prepare.lock"
    with FileLock(lock_path, f"prepare dataset {spec.benchmark}/{spec.split}"):
        return _write_prepared_dataset_unlocked(
            df, processed_dir, spec, source_metadata=source_metadata,
        )


def _build_prepared_dataset_metadata(
    spec_payload: dict[str, Any], digest: str, source_payload: dict[str, Any], row_count: int,
) -> dict[str, Any]:
    metadata = {
        "schema_version": DATASET_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": short_id("ds", {"spec": spec_payload, "content_digest": digest, "source": source_payload}),
        "spec": spec_payload,
        "content_digest": digest,
        "row_count": row_count,
        "source": source_payload,
        "provenance_status": "verified",
    }
    metadata["metadata_digest"] = integrity_digest(metadata)
    return metadata


def _write_prepared_dataset_unlocked(
    df: pd.DataFrame,
    processed_dir: Path,
    spec: DatasetSpec,
    *,
    source_metadata: dict[str, Any] | None = None,
) -> PreparedDataset:
    path = artifact_csv_path(processed_dir, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec_payload = spec.identity_payload()
    source_payload = canonicalize(source_metadata or {})
    validate_normalized_dataset(df, source=f"dataset for {spec.benchmark}/{spec.split}")
    atomic_write_text(path, df.to_csv(index=False))
    # Identity is defined over the exact representation runtime loading sees,
    # not the pre-serialization in-memory dtypes.
    persisted_df = pd.read_csv(path, dtype={"question_id": "string"})
    digest = dataset_content_digest(persisted_df)
    metadata = _build_prepared_dataset_metadata(spec_payload, digest, source_payload, len(persisted_df))
    metadata_path = artifact_metadata_path(path)
    atomic_write_json(metadata_path, metadata)
    return PreparedDataset(path, metadata_path, metadata, persisted_df)


def load_prepared_dataset(processed_dir: Path, spec: DatasetSpec) -> PreparedDataset:
    path = artifact_csv_path(processed_dir, spec)
    metadata_path = artifact_metadata_path(path)
    if not path.exists() or not metadata_path.exists():
        legacy = Path(processed_dir) / f"{spec.output_name or spec.benchmark}_normalized.csv"
        legacy_note = (
            f" A legacy unverified file exists at {legacy}; it cannot establish split "
            "or source provenance and will not be used. Re-run preparation."
            if legacy.exists() else ""
        )
        raise FileNotFoundError(
            f"Verified prepared artifact not found for benchmark={spec.benchmark!r}, "
            f"split={spec.split!r}: {path}.{legacy_note}"
        )
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetArtifactError(f"Unreadable dataset metadata {metadata_path}: {exc}") from exc
    if metadata.get("schema_version") != DATASET_ARTIFACT_SCHEMA_VERSION:
        raise DatasetArtifactError(f"Unsupported dataset artifact schema in {metadata_path}.")
    integrity_payload = {k: v for k, v in metadata.items() if k != "metadata_digest"}
    if metadata.get("metadata_digest") != integrity_digest(integrity_payload):
        raise DatasetArtifactError(f"Dataset metadata integrity check failed: {metadata_path}")
    if canonicalize(metadata.get("spec")) != spec.identity_payload():
        raise DatasetArtifactError(f"Dataset metadata spec does not match requested artifact: {path}")
    df = pd.read_csv(path, dtype={"question_id": "string"})
    validate_normalized_dataset(df, source=str(path))
    actual_digest = dataset_content_digest(df)
    if metadata.get("content_digest") != actual_digest or metadata.get("row_count") != len(df):
        raise DatasetArtifactError(
            f"Prepared dataset content no longer matches {metadata_path}; expected "
            f"{metadata.get('content_digest')}, found {actual_digest}. Re-prepare it."
        )
    expected_id = short_id("ds", {
        "spec": spec.identity_payload(), "content_digest": actual_digest,
        "source": canonicalize(metadata.get("source", {})),
    })
    if metadata.get("artifact_id") != expected_id:
        raise DatasetArtifactError(f"Dataset artifact identity is invalid in {metadata_path}.")
    return PreparedDataset(path, metadata_path, metadata, df)
