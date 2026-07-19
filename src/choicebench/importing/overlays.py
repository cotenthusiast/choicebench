"""Pure derivation of a repair/offline-transformation overlay from an
already-verified base realization.

Scope boundary: this module never opens a filesystem path, runs inference, or
merges the derived rows into a full realization -- it only validates the
overlay's binding to its base/authorization and produces the lineage/row
pieces for the *replaced* questions. Merging those with the base's retained
rows into a complete result set and constructing the final realization is
the import engine's job (it alone has the full base row set, the base run
directory, and the destination for the new run).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from choicebench.identity import integrity_digest
from choicebench.importing.authorization import ValidatedAuthorization
from choicebench.importing.csv_adapter import AdaptedTable
from choicebench.importing.dataset_reference import ExpectedDataset
from choicebench.importing.identity import make_lineage_component, make_result_origin
from choicebench.importing.schema import OverlaySpec

_EVIDENCE_STATUSES = {
    "complete", "qualified", "partial", "malformed", "recoverable", "failed",
}
_REPAIR_ORIGINS = {"native_inference", "external_repair_inference"}


class OverlayError(ValueError):
    """Raised when an overlay declaration or its derivation is invalid or unsafe."""


@dataclass(frozen=True)
class VerifiedBaseRealization:
    condition_id: str
    condition_digest: str
    realization_id: str
    realization_digest: str
    evidence_index_digest: str
    evidence_source_digests: Mapping[str, str]
    validation_artifact_sha256: str
    result_sha256: str | None
    rows_by_question_id: Mapping[str, Mapping[str, Any]]
    prediction_origins: Mapping[str, str]


@dataclass(frozen=True)
class DerivedRealizationPayload:
    condition_id: str
    condition_digest: str
    preownership_rows: tuple[Mapping[str, Any], ...]
    lineage_components: tuple[Mapping[str, Any], ...]
    result_origin: Mapping[str, Any]
    evidence_status: str
    replacement_question_ids: tuple[str, ...]


def _rows_by_question_id(table: AdaptedTable, mapping: Mapping[str, str]) -> dict[str, Any]:
    question_id_column = mapping["question_id"]
    rows_by_qid: dict[str, list] = {}
    for row in table.rows:
        qid = row.values.get(question_id_column)
        if qid is None:
            continue
        rows_by_qid.setdefault(str(qid), []).append(row)
    return rows_by_qid


def derive_overlay(
    *,
    base: VerifiedBaseRealization,
    overlay: OverlaySpec,
    authorization: ValidatedAuthorization,
    overlay_table: AdaptedTable,
    overlay_mapping: Mapping[str, str],
    expected: ExpectedDataset,
) -> DerivedRealizationPayload:
    """Derive the replaced-row lineage/content for one authorized overlay.

    `base` must be a value the caller obtained by independently verifying the
    full base run graph (never a bare path or unverified dict) -- this
    function performs no filesystem or verification work of its own, only
    binding checks against the values it is given.
    """
    if authorization.condition_digest != base.condition_digest:
        raise OverlayError(
            "Authorization condition does not match the base realization's "
            "condition; an authorization cannot be borrowed across conditions."
        )
    if (
        overlay.base_condition_digest != base.condition_digest
        or overlay.base_realization_id != base.realization_id
        or overlay.base_realization_digest != base.realization_digest
    ):
        raise OverlayError(
            "Overlay base condition/realization binding does not match the "
            "verified base realization."
        )
    if overlay.base_validation_artifact_sha256 != base.validation_artifact_sha256:
        raise OverlayError("Overlay base validation-artifact checksum does not match.")
    if overlay.base_result_sha256 != base.result_sha256:
        raise OverlayError("Overlay base result checksum does not match.")
    if overlay.authorization_id != authorization.bundle_id:
        raise OverlayError(
            "Overlay declares a different authorization bundle than the one supplied."
        )

    if set(overlay.base_evidence_digests) != set(authorization.input_evidence_digests):
        raise OverlayError(
            "Overlay base_evidence_digests do not exactly match the authorization's "
            "input evidence digests."
        )
    for key, digest in authorization.input_evidence_digests.items():
        if (
            overlay.base_evidence_digests.get(key) != digest
            or base.evidence_source_digests.get(key) != digest
        ):
            raise OverlayError(
                f"Overlay evidence digest {key!r} is not owned by the base realization's "
                "own evidence sources."
            )

    replacement_ids = set(overlay.replacement_reasons)
    if not replacement_ids:
        raise OverlayError("Overlay declares no replacement question IDs.")
    unauthorized = replacement_ids - set(authorization.question_reasons)
    if unauthorized:
        raise OverlayError(
            f"Overlay replaces question(s) {sorted(unauthorized)} that are not granted "
            "by its authorization."
        )
    unexpected = replacement_ids - set(expected.selected_question_ids)
    if unexpected:
        raise OverlayError(
            f"Overlay replaces question(s) {sorted(unexpected)} outside the expected "
            "dataset selection."
        )

    per_question_origin = overlay.result_origin.per_question_prediction_origins
    default_origin = overlay.result_origin.default_prediction_origin

    if authorization.authorization_type == "inference_repair":
        operation_type = "repair_overlay"
        if overlay.result_origin.derivation_origin != operation_type:
            raise OverlayError(
                f"inference_repair overlay must declare derivation_origin="
                f"{operation_type!r}, found {overlay.result_origin.derivation_origin!r}."
            )
        extra_origins = set(per_question_origin) - replacement_ids
        if extra_origins:
            raise OverlayError(
                f"inference_repair origin assignment(s) for unreplaced question(s) "
                f"{sorted(extra_origins)} are not permitted."
            )
        resolved_origins: dict[str, str] = {}
        for qid in replacement_ids:
            origin = per_question_origin.get(qid, default_origin)
            if origin is None:
                raise OverlayError(
                    f"inference_repair overlay has no declared origin assignment for "
                    f"replaced question {qid!r}."
                )
            if origin not in _REPAIR_ORIGINS:
                raise OverlayError(
                    f"inference_repair origin {origin!r} for question {qid!r} is not a "
                    f"valid repair prediction origin {sorted(_REPAIR_ORIGINS)}."
                )
            resolved_origins[qid] = origin
    elif authorization.authorization_type == "offline_transformation":
        operation_type = "offline_transformation"
        if overlay.result_origin.derivation_origin != operation_type:
            raise OverlayError(
                f"offline_transformation overlay must declare derivation_origin="
                f"{operation_type!r}, found {overlay.result_origin.derivation_origin!r}."
            )
        if default_origin is not None:
            raise OverlayError(
                "offline_transformation cannot declare a default prediction origin; it "
                "retains each underlying response's own origin."
            )
        resolved_origins = {}
        for qid in replacement_ids:
            origin = per_question_origin.get(qid) or base.prediction_origins.get(qid)
            if origin is None:
                raise OverlayError(
                    f"offline_transformation has no underlying prediction origin for "
                    f"question {qid!r}."
                )
            resolved_origins[qid] = origin
    else:
        raise OverlayError(
            f"Unsupported authorization_type {authorization.authorization_type!r}."
        )

    overlay_rows_by_qid = _rows_by_question_id(overlay_table, overlay_mapping)
    prediction_column = overlay_mapping.get("prediction")
    preownership_rows: list[dict[str, Any]] = []
    lineage_components: list[Mapping[str, Any]] = []
    for qid in sorted(replacement_ids):
        rows = overlay_rows_by_qid.get(qid, [])
        if len(rows) != 1:
            raise OverlayError(
                f"Overlay source has {len(rows)} row(s) for replaced question "
                f"{qid!r}; exactly one is required."
            )
        overlay_row = rows[0]
        expected_matches = expected.frame[expected.frame["question_id"].astype(str) == qid]
        if expected_matches.empty:
            raise OverlayError(f"Replaced question {qid!r} is not in the expected dataset.")
        expected_row = expected_matches.iloc[0].to_dict()
        predicted_option = None
        if prediction_column is not None:
            raw_prediction = overlay_row.values.get(prediction_column)
            predicted_option = None if raw_prediction is None else str(raw_prediction).strip().upper()
        origin = resolved_origins[qid]
        row_payload = {
            "question_id": qid,
            "question_text": expected_row.get("question_text"),
            "correct_option": expected_row.get("correct_option"),
            "choices_json": expected_row.get("choices_json"),
            "prediction_origin": origin,
            "predicted_option": predicted_option,
        }
        preownership_rows.append(row_payload)

        input_digest = integrity_digest(
            {"question_id": qid, "base_row": base.rows_by_question_id.get(qid)}
        )
        preownership_output_digest = integrity_digest({"question_id": qid, "row": row_payload})
        component = make_lineage_component(
            operation_type=operation_type,
            question_id=qid,
            parent_digests=(base.realization_digest,),
            source_digests=(overlay_table.source_sha256,),
            authorization_digest=authorization.bundle_digest,
            implementation=dict(overlay.implementation),
            implementation_mode="declared_external",
            parameters={},
            input_digest=input_digest,
            preownership_output_digest=preownership_output_digest,
            prediction_origin=origin,
        )
        lineage_components.append(component)

    row_assignments = [
        (component["identity"]["question_id"], component["identity"]["prediction_origin"], component["lineage_id"])
        for component in lineage_components
    ]
    result_origin = make_result_origin(derivation_origin=operation_type, row_assignments=row_assignments)

    if overlay.expected_evidence_status not in _EVIDENCE_STATUSES:
        raise OverlayError(
            f"Overlay expected_evidence_status {overlay.expected_evidence_status!r} is invalid."
        )

    return DerivedRealizationPayload(
        condition_id=base.condition_id,
        condition_digest=base.condition_digest,
        preownership_rows=tuple(preownership_rows),
        lineage_components=tuple(lineage_components),
        result_origin=result_origin,
        evidence_status=overlay.expected_evidence_status,
        replacement_question_ids=tuple(sorted(replacement_ids)),
    )
