"""Tests for verified source opening and content-addressed evidence storage
(Task 11, reduced scope)."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from choicebench.importing.csv_adapter import OpenedSource
from choicebench.importing.evidence import (
    EvidenceError,
    open_verified_source,
    validate_evidence_blob,
    validate_evidence_index,
    write_evidence_blob,
    write_evidence_index,
)
from choicebench.importing.schema import CsvDialectSpec, OptionMappingSpec, SourceArtifactSpec


def _declaration(tmp_path, data: bytes, source_id="results"):
    path = tmp_path / "results.csv"
    path.write_bytes(data)
    return SourceArtifactSpec(
        source_id=source_id,
        path=path,
        logical_path="inputs/results.csv",
        expected_sha256=sha256(data).hexdigest(),
        format="csv",
        format_version="1",
        classification="raw",
        dialect=CsvDialectSpec(),
        columns={"question_id": "qid"},
        expected_columns=("qid",),
        ignored_columns={},
        null_values=(),
        numeric_columns=(),
        option_mapping=OptionMappingSpec(
            mode="ordered_columns", ordered_columns=("a", "b"),
            structured_column=None, structured_label_key=None, structured_text_key=None,
        ),
        extra_field_policy="reject_unmapped",
        preserve_namespace="ext",
        source_run_id=None,
        source_repository=None,
        source_commit=None,
        notes={},
    )


def test_open_verified_source_accepts_matching_checksum(tmp_path):
    declaration = _declaration(tmp_path, b"qid,a,b\nq1,x,y\n")
    opened = open_verified_source(declaration)
    assert opened.source_id == "results"
    assert opened.sha256 == declaration.expected_sha256
    assert opened.data == b"qid,a,b\nq1,x,y\n"


def test_open_verified_source_rejects_checksum_mismatch(tmp_path):
    declaration = _declaration(tmp_path, b"qid,a,b\nq1,x,y\n")
    declaration.path.write_bytes(b"qid,a,b\nq1,TAMPERED,y\n")
    with pytest.raises(EvidenceError, match="checksum"):
        open_verified_source(declaration)


def test_open_verified_source_rejects_missing_file(tmp_path):
    declaration = _declaration(tmp_path, b"qid,a,b\nq1,x,y\n")
    declaration.path.unlink()
    with pytest.raises(EvidenceError, match="regular file"):
        open_verified_source(declaration)


def test_open_verified_source_rejects_symlink(tmp_path):
    declaration = _declaration(tmp_path, b"qid,a,b\nq1,x,y\n")
    real_path = declaration.path
    link_path = tmp_path / "link.csv"
    link_path.symlink_to(real_path)
    linked_declaration = replace(declaration, path=link_path)
    with pytest.raises(EvidenceError, match="symlink"):
        open_verified_source(linked_declaration)


def test_open_verified_source_rejects_path_outside_containment_root(tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    declaration = _declaration(outside_dir, b"qid,a,b\nq1,x,y\n")
    containment_root = tmp_path / "inside"
    containment_root.mkdir()
    with pytest.raises(EvidenceError, match="containment"):
        open_verified_source(declaration, containment_root=containment_root)


def test_write_and_validate_evidence_blob_round_trip(tmp_path):
    data = b"qid,a,b\nq1,x,y\n"
    source = OpenedSource(
        source_id="results", audit_path=tmp_path / "results.csv",
        logical_path="inputs/results.csv", data=data, sha256=sha256(data).hexdigest(),
    )
    record = write_evidence_blob(tmp_path, source, references=("q1",))
    validate_evidence_blob(tmp_path, record)  # must not raise
    assert record["references"] == ["q1"]


def test_write_evidence_blob_is_idempotent_for_identical_bytes(tmp_path):
    data = b"qid,a,b\nq1,x,y\n"
    source = OpenedSource(
        source_id="results", audit_path=tmp_path / "results.csv",
        logical_path="inputs/results.csv", data=data, sha256=sha256(data).hexdigest(),
    )
    first = write_evidence_blob(tmp_path, source, references=("q1",))
    second = write_evidence_blob(tmp_path, source, references=("q1",))
    assert first == second


def test_validate_evidence_blob_rejects_tampered_bytes(tmp_path):
    data = b"qid,a,b\nq1,x,y\n"
    source = OpenedSource(
        source_id="results", audit_path=tmp_path / "results.csv",
        logical_path="inputs/results.csv", data=data, sha256=sha256(data).hexdigest(),
    )
    record = write_evidence_blob(tmp_path, source, references=("q1",))
    from choicebench.importing.evidence import evidence_blob_path

    blob_path = evidence_blob_path(tmp_path, source.sha256, "csv")
    blob_path.write_bytes(data + b"tampered")
    with pytest.raises(EvidenceError, match="content integrity"):
        validate_evidence_blob(tmp_path, record)


def test_evidence_index_round_trips(tmp_path):
    data = b"qid,a,b\nq1,x,y\n"
    source = OpenedSource(
        source_id="results", audit_path=tmp_path / "results.csv",
        logical_path="inputs/results.csv", data=data, sha256=sha256(data).hexdigest(),
    )
    record = write_evidence_blob(tmp_path, source, references=("q1",))
    index = write_evidence_index(tmp_path, [record])
    validated = validate_evidence_index(tmp_path, index["evidence_index_digest"])
    assert validated["records"] == [record]


def test_evidence_index_rejects_digest_mismatch(tmp_path):
    data = b"qid,a,b\nq1,x,y\n"
    source = OpenedSource(
        source_id="results", audit_path=tmp_path / "results.csv",
        logical_path="inputs/results.csv", data=data, sha256=sha256(data).hexdigest(),
    )
    record = write_evidence_blob(tmp_path, source, references=("q1",))
    write_evidence_index(tmp_path, [record])
    with pytest.raises(EvidenceError, match="digest"):
        validate_evidence_index(tmp_path, "0" * 64)


def test_evidence_index_rejects_missing_blob(tmp_path):
    data = b"qid,a,b\nq1,x,y\n"
    source = OpenedSource(
        source_id="results", audit_path=tmp_path / "results.csv",
        logical_path="inputs/results.csv", data=data, sha256=sha256(data).hexdigest(),
    )
    record = write_evidence_blob(tmp_path, source, references=("q1",))
    index = write_evidence_index(tmp_path, [record])
    from choicebench.importing.evidence import evidence_blob_path

    evidence_blob_path(tmp_path, source.sha256, "csv").unlink()
    with pytest.raises(EvidenceError, match="Missing evidence blob"):
        validate_evidence_index(tmp_path, index["evidence_index_digest"])
