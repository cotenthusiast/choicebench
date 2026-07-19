"""Validate imported source rows by question identity and derive per-realization
evidence status and evaluable content.

Reduced scope: this covers the concrete guarantees the repair-import workflow
needs (exact question-identity join, content cross-check against the expected
dataset snapshot, coverage/defect-driven evidence status with a fail-closed
declaration check, and evaluable-row derivation) without the full historical
option-count/method-specific-extension richness of the original plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from choicebench.identity import canonicalize, integrity_digest
from choicebench.importing.csv_adapter import AdaptedTable, LogicalRecordSpan, SourceRow
from choicebench.importing.dataset_reference import ExpectedDataset
from choicebench.importing.schema import ImportConditionSpec
from choicebench.infra.artifacts import atomic_write_bytes

_EVALUABLE_STATUSES = {"complete", "qualified"}
_COVERAGE_ONLY_FINDING_CODES = {"MISSING_QUESTION_ID"}
VALIDATION_ARTIFACT_SCHEMA_VERSION = "choicebench.realization-validation.v1"


class ImportValidationError(ValueError):
    """Raised when imported row content fails fail-closed validation."""


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    source_id: str
    field: str | None
    question_id: str | None
    record_span: LogicalRecordSpan | None
    raw_row_sha256: str | None
    message: str

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "source_id": self.source_id,
            "field": self.field,
            "question_id": self.question_id,
            "record_span": (
                None
                if self.record_span is None
                else {
                    "record_index": self.record_span.index,
                    "start": self.record_span.start,
                    "end": self.record_span.end,
                    "terminator_hex": self.record_span.terminator.hex(),
                }
            ),
            "raw_row_sha256": self.raw_row_sha256,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidatedSourceRows:
    source_id: str
    rows_by_question_id: Mapping[str, SourceRow]
    expected_question_ids: tuple[str, ...]
    findings: tuple[ValidationFinding, ...]
    findings_digest: str


def _expected_row(expected: ExpectedDataset, question_id: str) -> Mapping[str, Any] | None:
    matches = expected.frame[expected.frame["question_id"].astype(str) == question_id]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def _expected_choice_letters(expected_row: Mapping[str, Any]) -> list[str]:
    """The valid answer letters for a question, derived from its ordered
    choices_json array (index 0 -> "A", 1 -> "B", ...) -- the same ordering
    ChoiceBench's own option map already uses to define correct_option."""
    raw = expected_row.get("choices_json")
    if not raw:
        return []
    try:
        choices = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(choices, list):
        return []
    return [chr(ord("A") + index) for index in range(len(choices))]


def validate_source_rows(
    table: AdaptedTable,
    *,
    source_id: str,
    mapping: Mapping[str, str],
    condition: ImportConditionSpec,
    expected: ExpectedDataset,
) -> ValidatedSourceRows:
    """Join source rows to the expected dataset by exact string question_id.

    Every declared field is compared against the expected snapshot; source
    values never override it. No row is dropped, padded, deduplicated,
    reordered, or repaired -- every input row is either kept (with any
    findings recorded alongside it) or excluded only when its question_id is
    genuinely ambiguous (duplicated within this source).
    """
    if "question_id" not in mapping:
        raise ImportValidationError(f"{source_id}: mapping declares no question_id column.")
    question_id_column = mapping["question_id"]
    expected_ids = tuple(condition.expected_question_ids)
    expected_id_set = set(expected_ids)

    findings: list[ValidationFinding] = []
    rows_by_question_id: dict[str, SourceRow] = {}
    rows_by_raw_id: dict[str, list[SourceRow]] = {}

    for row in table.rows:
        raw_qid = row.values.get(question_id_column)
        if raw_qid is None or not str(raw_qid).strip():
            findings.append(
                ValidationFinding(
                    code="NULL_QUESTION_ID",
                    source_id=source_id,
                    field=question_id_column,
                    question_id=None,
                    record_span=row.span,
                    raw_row_sha256=row.raw_sha256,
                    message="Row has a null or empty question_id.",
                )
            )
            continue
        rows_by_raw_id.setdefault(str(raw_qid), []).append(row)

    for qid, rows in rows_by_raw_id.items():
        if len(rows) > 1:
            for row in rows:
                findings.append(
                    ValidationFinding(
                        code="DUPLICATE_QUESTION_ID",
                        source_id=source_id,
                        field=question_id_column,
                        question_id=qid,
                        record_span=row.span,
                        raw_row_sha256=row.raw_sha256,
                        message=f"question_id {qid!r} occurs {len(rows)} times in this source.",
                    )
                )
            continue
        row = rows[0]
        rows_by_question_id[qid] = row
        if qid not in expected_id_set:
            findings.append(
                ValidationFinding(
                    code="UNEXPECTED_QUESTION_ID",
                    source_id=source_id,
                    field=question_id_column,
                    question_id=qid,
                    record_span=row.span,
                    raw_row_sha256=row.raw_sha256,
                    message=f"question_id {qid!r} is not in the expected dataset selection.",
                )
            )
            continue
        expected_row = _expected_row(expected, qid)
        if expected_row is None:
            continue
        for semantic_field in ("question_text", "correct_option"):
            source_column = mapping.get(semantic_field)
            if source_column is None:
                continue
            source_value = row.values.get(source_column)
            expected_value = expected_row.get(semantic_field)
            normalized_source = None if source_value is None else str(source_value).strip()
            normalized_expected = None if expected_value is None else str(expected_value).strip()
            if semantic_field == "correct_option":
                normalized_source = normalized_source and normalized_source.upper()
                normalized_expected = normalized_expected and normalized_expected.upper()
            if normalized_source is None or normalized_source != normalized_expected:
                findings.append(
                    ValidationFinding(
                        code=f"{semantic_field.upper()}_MISMATCH",
                        source_id=source_id,
                        field=source_column,
                        question_id=qid,
                        record_span=row.span,
                        raw_row_sha256=row.raw_sha256,
                        message=f"{semantic_field} does not match the expected dataset snapshot.",
                    )
                )
        expected_choice_letters = _expected_choice_letters(expected_row)
        prediction_column = mapping.get("prediction")
        if prediction_column is not None:
            prediction_value = row.values.get(prediction_column)
            if prediction_value is None or not str(prediction_value).strip():
                findings.append(
                    ValidationFinding(
                        code="MISSING_PREDICTION",
                        source_id=source_id,
                        field=prediction_column,
                        question_id=qid,
                        record_span=row.span,
                        raw_row_sha256=row.raw_sha256,
                        message="Row has no parsed prediction.",
                    )
                )
            else:
                letter = str(prediction_value).strip().upper()
                if letter not in expected_choice_letters:
                    findings.append(
                        ValidationFinding(
                            code="INVALID_PREDICTION",
                            source_id=source_id,
                            field=prediction_column,
                            question_id=qid,
                            record_span=row.span,
                            raw_row_sha256=row.raw_sha256,
                            message=(
                                f"Parsed prediction {prediction_value!r} is not one of "
                                f"this question's options {expected_choice_letters}."
                            ),
                        )
                    )

    for qid in sorted(expected_id_set - set(rows_by_question_id)):
        findings.append(
            ValidationFinding(
                code="MISSING_QUESTION_ID",
                source_id=source_id,
                field=None,
                question_id=qid,
                record_span=None,
                raw_row_sha256=None,
                message=f"Expected question_id {qid!r} is absent from this source.",
            )
        )

    findings_tuple = tuple(findings)
    findings_digest = integrity_digest([finding.to_json() for finding in findings_tuple])
    return ValidatedSourceRows(
        source_id=source_id,
        rows_by_question_id=rows_by_question_id,
        expected_question_ids=expected_ids,
        findings=findings_tuple,
        findings_digest=findings_digest,
    )


@dataclass(frozen=True)
class RealizationValidation:
    evaluable_rows: tuple[Mapping[str, Any], ...]
    findings: tuple[ValidationFinding, ...]
    validation_digest: str
    computed_evidence_status: str
    evaluable: bool
    qualifications: tuple[Mapping[str, Any], ...]
    limitations: tuple[Mapping[str, Any], ...]
    defect_question_ids: tuple[str, ...]


def normalize_realization_rows(
    validated: ValidatedSourceRows,
    *,
    condition: ImportConditionSpec,
    expected: ExpectedDataset,
    mapping: Mapping[str, str] | None = None,
    experiment_id: str | None = None,
    realization_id: str | None = None,
) -> RealizationValidation:
    """Recompute evidence status from exact coverage/defects and refuse a
    declaration mismatch. Fill evaluative content only from ExpectedDataset;
    source values were already checked, never trusted, in validate_source_rows.
    """
    expected_ids = set(validated.expected_question_ids)
    present_ids = set(validated.rows_by_question_id)
    missing_ids = expected_ids - present_ids
    defect_question_ids = tuple(
        sorted(
            {
                finding.question_id
                for finding in validated.findings
                if finding.question_id is not None
                and finding.code not in _COVERAGE_ONLY_FINDING_CODES
            }
        )
    )
    recoverable_ids = set(condition.recoverable_question_ids)

    if not present_ids and expected_ids:
        computed = "failed"
    elif defect_question_ids:
        computed = "malformed"
    elif missing_ids and missing_ids <= recoverable_ids and recoverable_ids:
        computed = "recoverable"
    elif missing_ids:
        computed = "partial"
    else:
        computed = "qualified" if condition.qualifications else "complete"

    if computed != condition.evidence_status:
        raise ImportValidationError(
            f"Computed evidence status {computed!r} does not match the declared "
            f"evidence status {condition.evidence_status!r}."
        )

    evaluable = computed in _EVALUABLE_STATUSES and condition.scope_disposition == "included"

    prediction_column = None if mapping is None else mapping.get("prediction")
    evaluable_rows: list[dict[str, Any]] = []
    if evaluable:
        per_question_origin = condition.result_origin.per_question_prediction_origins
        default_origin = condition.result_origin.default_prediction_origin
        for qid in validated.expected_question_ids:
            if qid not in validated.rows_by_question_id:
                continue
            expected_row = _expected_row(expected, qid)
            origin = per_question_origin.get(qid, default_origin)
            if origin is None:
                raise ImportValidationError(
                    f"No declared prediction_origin for evaluable question {qid!r}."
                )
            predicted_option = None
            if prediction_column is not None:
                raw_prediction = validated.rows_by_question_id[qid].values.get(prediction_column)
                predicted_option = None if raw_prediction is None else str(raw_prediction).strip().upper()
            row: dict[str, Any] = {
                "question_id": qid,
                "question_text": expected_row.get("question_text"),
                "correct_option": expected_row.get("correct_option"),
                "choices_json": expected_row.get("choices_json"),
                "prediction_origin": origin,
                "predicted_option": predicted_option,
            }
            evaluable_rows.append(row)

    validation_payload = {
        "schema_version": "choicebench.realization-validation-content.v1",
        "computed_evidence_status": computed,
        "evaluable": evaluable,
        "findings_digest": validated.findings_digest,
        "evaluable_rows": canonicalize(evaluable_rows),
    }
    validation_digest = integrity_digest(validation_payload)

    return RealizationValidation(
        evaluable_rows=tuple(evaluable_rows),
        findings=validated.findings,
        validation_digest=validation_digest,
        computed_evidence_status=computed,
        evaluable=evaluable,
        qualifications=tuple(condition.qualifications),
        limitations=tuple(condition.limitations),
        defect_question_ids=defect_question_ids,
    )


@dataclass(frozen=True)
class PreparedValidationArtifact:
    relative_path: str
    json_bytes: bytes
    file_sha256: str
    validation_digest: str


def prepare_realization_validation_artifact(
    validation: RealizationValidation,
    *,
    realization: Mapping[str, Any],
    evidence_records: Sequence[Mapping[str, Any]],
) -> PreparedValidationArtifact:
    realization_id = realization["realization_id"]
    payload = canonicalize(
        {
            "schema_version": VALIDATION_ARTIFACT_SCHEMA_VERSION,
            "realization_id": realization_id,
            "realization_digest": realization["realization_digest"],
            "validation_digest": validation.validation_digest,
            "computed_evidence_status": validation.computed_evidence_status,
            "evaluable": validation.evaluable,
            "qualifications": list(validation.qualifications),
            "limitations": list(validation.limitations),
            "defect_question_ids": list(validation.defect_question_ids),
            "findings": [finding.to_json() for finding in validation.findings],
            "evidence_records": list(evidence_records),
        }
    )
    json_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return PreparedValidationArtifact(
        relative_path=f"artifacts/imports/validation/{realization_id}.json",
        json_bytes=json_bytes,
        file_sha256=sha256(json_bytes).hexdigest(),
        validation_digest=validation.validation_digest,
    )


def write_realization_validation_artifact(
    staged_run: Path, prepared: PreparedValidationArtifact
) -> tuple[Path, bool]:
    path = Path(staged_run) / prepared.relative_path
    if path.exists():
        existing = path.read_bytes()
        if existing == prepared.json_bytes:
            return path, True
        raise RuntimeError(f"Refusing to overwrite a divergent validation artifact: {path}")
    atomic_write_bytes(path, prepared.json_bytes)
    return path, False


def validate_realization_validation_artifact(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    realization: Mapping[str, Any],
    expected_sha256: str,
) -> Mapping[str, Any]:
    path = Path(path)
    try:
        actual_bytes = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Validation artifact is unreadable: {path}: {exc}") from exc
    actual_sha256 = sha256(actual_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Validation artifact content integrity check failed: {path}; expected "
            f"{expected_sha256}, found {actual_sha256}."
        )
    try:
        payload = json.loads(actual_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Validation artifact is not valid JSON: {path}: {exc}") from exc
    if payload.get("schema_version") != VALIDATION_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported validation artifact schema: {path}")
    if (
        payload.get("realization_id") != realization.get("realization_id")
        or payload.get("realization_digest") != realization.get("realization_digest")
    ):
        raise RuntimeError(f"Validation artifact realization binding is inconsistent: {path}")
    return payload
