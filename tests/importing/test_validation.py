"""Reduced-scope tests for importing.validation: question-identity join,
evidence-status computation with declaration cross-check, and evaluable-row
derivation. Reuses tests/importing/test_identity.py's condition/dataset
fixtures instead of re-deriving them here.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from choicebench.importing.csv_adapter import AdaptedTable, LogicalRecordSpan, SourceRow
from choicebench.importing.validation import (
    ImportValidationError,
    normalize_realization_rows,
    prepare_realization_validation_artifact,
    validate_realization_validation_artifact,
    validate_source_rows,
    write_realization_validation_artifact,
)
from tests.importing.test_identity import _condition, _dataset

_MAPPING = {
    "question_id": "qid",
    "question_text": "question",
    "correct_option": "gold_answer",
    "prediction": "predicted_letter",
}


def _row(index, qid, question, gold, a, b, predicted):
    values = {
        "qid": qid,
        "question": question,
        "gold_answer": gold,
        "opt_a": a,
        "opt_b": b,
        "predicted_letter": predicted,
    }
    return SourceRow(
        values=values,
        span=LogicalRecordSpan(index=index, start=index * 10, end=index * 10 + 9, terminator=b"\n"),
        raw_sha256=f"{index:064x}",
    )


def _valid_rows():
    return (
        _row(0, "q1", "One?", "A", "x", "y", "a"),
        _row(1, "q2", "Two?", "B", "m", "n", "b"),
    )


def _table(rows):
    return AdaptedTable(columns=tuple(_MAPPING.values()), rows=rows, source_sha256="1" * 64)


def test_validate_source_rows_accepts_matching_rows():
    condition = _condition()
    expected = _dataset()
    validated = validate_source_rows(
        _table(_valid_rows()), source_id="results", mapping=_MAPPING,
        condition=condition, expected=expected,
    )
    assert validated.findings == ()
    assert set(validated.rows_by_question_id) == {"q1", "q2"}


def test_validate_source_rows_flags_null_question_id():
    rows = (_row(0, None, "One?", "A", "x", "y", "a"), _valid_rows()[1])
    validated = validate_source_rows(
        _table(rows), source_id="results", mapping=_MAPPING,
        condition=_condition(), expected=_dataset(),
    )
    codes = {f.code for f in validated.findings}
    assert "NULL_QUESTION_ID" in codes
    assert "MISSING_QUESTION_ID" in codes  # q1 never arrived


def test_validate_source_rows_flags_duplicate_question_id():
    rows = (*_valid_rows(), _row(2, "q1", "One?", "A", "x", "y", "a"))
    validated = validate_source_rows(
        _table(rows), source_id="results", mapping=_MAPPING,
        condition=_condition(), expected=_dataset(),
    )
    duplicate_findings = [f for f in validated.findings if f.code == "DUPLICATE_QUESTION_ID"]
    assert len(duplicate_findings) == 2  # both occurrences flagged
    assert "q1" not in validated.rows_by_question_id  # ambiguous, not usable


def test_validate_source_rows_flags_unexpected_question_id():
    rows = (*_valid_rows(), _row(2, "q9", "Nine?", "A", "x", "y", "a"))
    validated = validate_source_rows(
        _table(rows), source_id="results", mapping=_MAPPING,
        condition=_condition(), expected=_dataset(),
    )
    assert any(f.code == "UNEXPECTED_QUESTION_ID" and f.question_id == "q9" for f in validated.findings)


def test_validate_source_rows_flags_missing_question_id():
    validated = validate_source_rows(
        _table(_valid_rows()[:1]), source_id="results", mapping=_MAPPING,
        condition=_condition(), expected=_dataset(),
    )
    assert any(f.code == "MISSING_QUESTION_ID" and f.question_id == "q2" for f in validated.findings)


def test_validate_source_rows_flags_gold_mismatch():
    rows = (_row(0, "q1", "One?", "B", "x", "y", "a"), _valid_rows()[1])
    validated = validate_source_rows(
        _table(rows), source_id="results", mapping=_MAPPING,
        condition=_condition(), expected=_dataset(),
    )
    assert any(f.code == "CORRECT_OPTION_MISMATCH" and f.question_id == "q1" for f in validated.findings)


def test_validate_source_rows_flags_invalid_prediction():
    rows = (_row(0, "q1", "One?", "A", "x", "y", "z"), _valid_rows()[1])
    validated = validate_source_rows(
        _table(rows), source_id="results", mapping=_MAPPING,
        condition=_condition(), expected=_dataset(),
    )
    assert any(f.code == "INVALID_PREDICTION" and f.question_id == "q1" for f in validated.findings)


def test_validate_source_rows_flags_missing_prediction():
    rows = (_row(0, "q1", "One?", "A", "x", "y", None), _valid_rows()[1])
    validated = validate_source_rows(
        _table(rows), source_id="results", mapping=_MAPPING,
        condition=_condition(), expected=_dataset(),
    )
    assert any(f.code == "MISSING_PREDICTION" and f.question_id == "q1" for f in validated.findings)


def test_validate_source_rows_never_drops_or_reorders_rows():
    rows = (_valid_rows()[1], _valid_rows()[0])  # reversed input order
    validated = validate_source_rows(
        _table(rows), source_id="results", mapping=_MAPPING,
        condition=_condition(), expected=_dataset(),
    )
    assert len(validated.rows_by_question_id) == 2


def test_normalize_realization_rows_computes_complete_status():
    condition = _condition()
    validated = validate_source_rows(
        _table(_valid_rows()), source_id="results", mapping=_MAPPING,
        condition=condition, expected=_dataset(),
    )
    result = normalize_realization_rows(validated, condition=condition, expected=_dataset())
    assert result.computed_evidence_status == "complete"
    assert result.evaluable is True
    assert len(result.evaluable_rows) == 2
    assert {row["question_id"] for row in result.evaluable_rows} == {"q1", "q2"}
    assert all(
        row["prediction_origin"] == "external_historical_inference" for row in result.evaluable_rows
    )


def test_normalize_realization_rows_refuses_declaration_mismatch():
    condition = replace(_condition(), evidence_status="malformed")
    validated = validate_source_rows(
        _table(_valid_rows()), source_id="results", mapping=_MAPPING,
        condition=condition, expected=_dataset(),
    )
    with pytest.raises(ImportValidationError, match="evidence status"):
        normalize_realization_rows(validated, condition=condition, expected=_dataset())


def test_normalize_realization_rows_produces_no_evaluable_rows_for_partial():
    condition = replace(
        _condition(),
        expected_question_ids=("q1", "q2", "q3"),
        evidence_status="partial",
    )
    validated = validate_source_rows(
        _table(_valid_rows()), source_id="results", mapping=_MAPPING,
        condition=condition, expected=_dataset(),
    )
    result = normalize_realization_rows(validated, condition=condition, expected=_dataset())
    assert result.computed_evidence_status == "partial"
    assert result.evaluable is False
    assert result.evaluable_rows == ()


def test_normalize_realization_rows_produces_no_evaluable_rows_for_malformed():
    rows = (_row(0, "q1", "One?", "A", "x", "y", "z"), _valid_rows()[1])  # invalid prediction
    condition = replace(_condition(), evidence_status="malformed")
    validated = validate_source_rows(
        _table(rows), source_id="results", mapping=_MAPPING,
        condition=condition, expected=_dataset(),
    )
    result = normalize_realization_rows(validated, condition=condition, expected=_dataset())
    assert result.computed_evidence_status == "malformed"
    assert result.evaluable is False
    assert result.evaluable_rows == ()
    assert result.defect_question_ids == ("q1",)


def test_normalize_realization_rows_computes_recoverable_status():
    condition = replace(
        _condition(),
        expected_question_ids=("q1", "q2", "q3"),
        evidence_status="recoverable",
        recoverable_question_ids=("q3",),
    )
    validated = validate_source_rows(
        _table(_valid_rows()), source_id="results", mapping=_MAPPING,
        condition=condition, expected=_dataset(),
    )
    result = normalize_realization_rows(validated, condition=condition, expected=_dataset())
    assert result.computed_evidence_status == "recoverable"
    assert result.evaluable is False


def test_normalize_realization_rows_computes_failed_status_for_zero_rows():
    condition = _condition()
    validated = validate_source_rows(
        _table(()), source_id="results", mapping=_MAPPING,
        condition=replace(condition, evidence_status="failed"), expected=_dataset(),
    )
    result = normalize_realization_rows(
        validated, condition=replace(condition, evidence_status="failed"), expected=_dataset()
    )
    assert result.computed_evidence_status == "failed"
    assert result.evaluable is False


def test_normalize_realization_rows_excludes_from_paper_matrix_is_not_evaluable():
    condition = replace(_condition(), scope_disposition="excluded_from_paper_matrix")
    validated = validate_source_rows(
        _table(_valid_rows()), source_id="results", mapping=_MAPPING,
        condition=condition, expected=_dataset(),
    )
    result = normalize_realization_rows(validated, condition=condition, expected=_dataset())
    assert result.computed_evidence_status == "complete"  # row content is fine
    assert result.evaluable is False  # but scope excludes it from evaluation
    assert result.evaluable_rows == ()


def _realization_stub(digest="a" * 64):
    return {"realization_id": "real_" + "b" * 16, "realization_digest": digest}


def test_validation_artifact_round_trips(tmp_path):
    condition = _condition()
    validated = validate_source_rows(
        _table(_valid_rows()), source_id="results", mapping=_MAPPING,
        condition=condition, expected=_dataset(),
    )
    result = normalize_realization_rows(validated, condition=condition, expected=_dataset())
    realization = _realization_stub()
    prepared = prepare_realization_validation_artifact(
        result, realization=realization, evidence_records=()
    )
    path, reused = write_realization_validation_artifact(tmp_path, prepared)
    assert reused is False
    payload = validate_realization_validation_artifact(
        path, manifest={}, realization=realization, expected_sha256=prepared.file_sha256
    )
    assert payload["realization_id"] == realization["realization_id"]

    _, second_reused = write_realization_validation_artifact(tmp_path, prepared)
    assert second_reused is True


def test_validation_artifact_rejects_tampered_bytes(tmp_path):
    condition = _condition()
    validated = validate_source_rows(
        _table(_valid_rows()), source_id="results", mapping=_MAPPING,
        condition=condition, expected=_dataset(),
    )
    result = normalize_realization_rows(validated, condition=condition, expected=_dataset())
    realization = _realization_stub()
    prepared = prepare_realization_validation_artifact(
        result, realization=realization, evidence_records=()
    )
    path, _ = write_realization_validation_artifact(tmp_path, prepared)
    path.write_bytes(prepared.json_bytes + b"\n// tampered")
    with pytest.raises(RuntimeError, match="content integrity"):
        validate_realization_validation_artifact(
            path, manifest={}, realization=realization, expected_sha256=prepared.file_sha256
        )
