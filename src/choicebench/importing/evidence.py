"""Open declared sources with a verified checksum and store referenced
row-level evidence bytes as run-local content-addressed blobs.

Reduced scope: covers the concrete safety guarantees the repair-import
workflow needs -- read-once-hash-forward source opening, path containment,
symlink refusal, and a self-digested evidence index -- without the full
historical suffix-derivation/dedup-across-runs richness of the original plan.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from choicebench.identity import canonicalize, integrity_digest
from choicebench.importing.csv_adapter import OpenedSource
from choicebench.importing.schema import SourceArtifactSpec
from choicebench.infra.artifacts import atomic_write_bytes, atomic_write_json

EVIDENCE_INDEX_SCHEMA_VERSION = "choicebench.evidence-index.v1"
_SAFE_SUFFIXES = {"csv": "csv"}


class EvidenceError(ValueError):
    """Raised when a source or stored evidence fails a safety or integrity check."""


def open_verified_source(
    declaration: SourceArtifactSpec, *, containment_root: Path | None = None
) -> OpenedSource:
    """Open exactly the declared regular file, read its bytes once, and hash
    those same bytes -- never re-read or re-derive the digest from a
    separately reopened handle."""
    path = Path(declaration.path)
    if containment_root is not None:
        root = Path(containment_root).resolve()
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise EvidenceError(
                f"Source {declaration.source_id!r} path {path} escapes containment "
                f"root {root}."
            ) from exc
    if path.is_symlink():
        raise EvidenceError(f"Source {declaration.source_id!r} path {path} is a symlink.")
    if not path.is_file():
        raise EvidenceError(f"Source {declaration.source_id!r} path {path} is not a regular file.")
    data = path.read_bytes()
    actual_sha256 = sha256(data).hexdigest()
    if actual_sha256 != declaration.expected_sha256:
        raise EvidenceError(
            f"Source {declaration.source_id!r} checksum mismatch: expected "
            f"{declaration.expected_sha256}, found {actual_sha256}."
        )
    return OpenedSource(
        source_id=declaration.source_id,
        audit_path=path,
        logical_path=declaration.logical_path,
        data=data,
        sha256=actual_sha256,
    )


def evidence_blob_path(staged_run: Path, sha256_digest: str, format_name: str) -> Path:
    suffix = _SAFE_SUFFIXES.get(format_name)
    if suffix is None:
        raise EvidenceError(f"Unsupported evidence blob format {format_name!r}.")
    if len(sha256_digest) != 64 or any(c not in "0123456789abcdef" for c in sha256_digest):
        raise EvidenceError(f"Invalid evidence blob digest {sha256_digest!r}.")
    return (
        Path(staged_run)
        / "artifacts" / "imports" / "evidence" / "sha256"
        / sha256_digest[:2] / f"{sha256_digest}.{suffix}"
    )


def write_evidence_blob(
    staged_run: Path, source: OpenedSource, references: Sequence[str]
) -> dict[str, Any]:
    """Store one referenced source's bytes as a run-local content-addressed
    blob. Identical bytes already staged (e.g. two references to the same
    source) are reused rather than rewritten; a divergent existing blob at
    the same digest path is refused."""
    blob_path = evidence_blob_path(staged_run, source.sha256, "csv")
    if blob_path.exists():
        if blob_path.read_bytes() != source.data:
            raise EvidenceError(
                f"Refusing to overwrite a divergent existing evidence blob: {blob_path}"
            )
    else:
        atomic_write_bytes(blob_path, source.data)
    record = canonicalize(
        {
            "source_id": source.source_id,
            "logical_path": source.logical_path,
            "sha256": source.sha256,
            "size": len(source.data),
            "format": "csv",
            "references": sorted(set(references)),
        }
    )
    sidecar_path = blob_path.with_suffix(blob_path.suffix + ".json")
    if sidecar_path.exists():
        existing = json.loads(sidecar_path.read_text())
        if existing != record:
            raise EvidenceError(
                f"Refusing to overwrite a divergent evidence blob sidecar: {sidecar_path}"
            )
    else:
        atomic_write_json(sidecar_path, record)
    return record


def validate_evidence_blob(run_dir: Path, record: Mapping[str, Any]) -> None:
    blob_path = evidence_blob_path(run_dir, record["sha256"], record["format"])
    if not blob_path.is_file():
        raise EvidenceError(f"Missing evidence blob: {blob_path}")
    actual = sha256(blob_path.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        raise EvidenceError(
            f"Evidence blob content integrity check failed: {blob_path}; expected "
            f"{record['sha256']}, found {actual}."
        )
    sidecar_path = blob_path.with_suffix(blob_path.suffix + ".json")
    try:
        stored = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Evidence blob sidecar is unreadable: {sidecar_path}: {exc}") from exc
    if canonicalize(stored) != canonicalize(dict(record)):
        raise EvidenceError(f"Evidence blob sidecar does not match its declared record: {sidecar_path}")


def write_evidence_index(
    staged_run: Path, records: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    payload = canonicalize(
        {
            "schema_version": EVIDENCE_INDEX_SCHEMA_VERSION,
            "records": list(records),
        }
    )
    evidence_index_digest = integrity_digest(payload)
    index = {**payload, "evidence_index_digest": evidence_index_digest}
    index_path = Path(staged_run) / "artifacts" / "imports" / "evidence" / "index.json"
    atomic_write_json(index_path, index)
    return index


def validate_evidence_index(run_dir: Path, expected_digest: str) -> Mapping[str, Any]:
    index_path = Path(run_dir) / "artifacts" / "imports" / "evidence" / "index.json"
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Evidence index is unreadable: {index_path}: {exc}") from exc
    if index.get("schema_version") != EVIDENCE_INDEX_SCHEMA_VERSION:
        raise EvidenceError(f"Unsupported evidence index schema: {index_path}")
    payload = {key: value for key, value in index.items() if key != "evidence_index_digest"}
    if integrity_digest(canonicalize(payload)) != index.get("evidence_index_digest"):
        raise EvidenceError(f"Evidence index integrity check failed: {index_path}")
    if index["evidence_index_digest"] != expected_digest:
        raise EvidenceError(
            f"Evidence index digest does not match its expected binding: {index_path}"
        )
    for record in index["records"]:
        validate_evidence_blob(run_dir, record)
    return index
