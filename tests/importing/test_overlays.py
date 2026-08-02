"""Tests for pure overlay derivation (Task 10, reduced scope)."""

from __future__ import annotations

from pathlib import Path

import pytest

from choicebench.importing.authorization import ValidatedAuthorization
from choicebench.importing.csv_adapter import AdaptedTable, LogicalRecordSpan, SourceRow
from choicebench.importing.overlays import (
    OverlayError,
    VerifiedBaseRealization,
    derive_overlay,
)
from choicebench.importing.schema import OverlaySpec, ResultOriginSpec
from tests.importing.test_identity import _dataset

_BASE_CONDITION_DIGEST = "c" * 64
_BASE_REALIZATION_ID = "real_" + "a" * 16
_BASE_REALIZATION_DIGEST = "d" * 64
_EVIDENCE_DIGESTS = {"ev1": "1" * 64}
_VALIDATION_SHA = "3" * 64
_RESULT_SHA = "4" * 64
_AUTH_BUNDLE_ID = "auth_" + "b" * 16
_AUTH_BUNDLE_DIGEST = "5" * 64


def _base(**overrides):
    defaults = dict(
        condition_id="cond_a",
        condition_digest=_BASE_CONDITION_DIGEST,
        realization_id=_BASE_REALIZATION_ID,
        realization_digest=_BASE_REALIZATION_DIGEST,
        evidence_index_digest="2" * 64,
        evidence_source_digests=_EVIDENCE_DIGESTS,
        validation_artifact_sha256=_VALIDATION_SHA,
        result_sha256=_RESULT_SHA,
        rows_by_question_id={"q1": {"question_id": "q1"}, "q2": {"question_id": "q2"}},
        prediction_origins={"q1": "external_historical_inference", "q2": "external_historical_inference"},
    )
    defaults.update(overrides)
    return VerifiedBaseRealization(**defaults)


def _authorization(*, authorization_type, executable, question_reasons):
    return ValidatedAuthorization(
        bundle_id=_AUTH_BUNDLE_ID,
        bundle_digest=_AUTH_BUNDLE_DIGEST,
        authorization_type=authorization_type,
        condition_digest=_BASE_CONDITION_DIGEST,
        question_reasons=dict(question_reasons),
        authority="principal-investigator",
        purpose="repair",
        executable=executable,
        source_sha256="6" * 64,
        input_evidence_digests=_EVIDENCE_DIGESTS,
        expected_snapshot_digest="7" * 64,
    )


def _overlay(*, result_origin, replacement_reasons, authorization_id=_AUTH_BUNDLE_ID, **overrides):
    defaults = dict(
        overlay_id="ovl_1",
        base_run_path=Path("/tmp/base-run"),
        base_condition_digest=_BASE_CONDITION_DIGEST,
        base_realization_id=_BASE_REALIZATION_ID,
        base_realization_digest=_BASE_REALIZATION_DIGEST,
        base_evidence_digests=_EVIDENCE_DIGESTS,
        base_validation_artifact_sha256=_VALIDATION_SHA,
        base_result_sha256=_RESULT_SHA,
        source_id="overlay-source",
        authorization_id=authorization_id,
        replacement_reasons=dict(replacement_reasons),
        result_origin=result_origin,
        lineage_notes={},
        # source_digest must match the overlay source's own checksum ("b" * 64,
        # see _overlay_table) -- make_lineage_component requires a declared
        # external implementation's code to be one of the declared sources.
        implementation={"qualified_name": "external:repair_tool", "source_digest": "b" * 64},
        input_digest="9" * 64,
        preownership_output_digest="a" * 64,
        expected_evidence_status="complete",
    )
    defaults.update(overrides)
    return OverlaySpec(**defaults)


def _overlay_table(rows):
    return AdaptedTable(columns=("qid", "predicted_letter"), rows=rows, source_sha256="b" * 64)


def _overlay_row(qid, predicted):
    return SourceRow(
        values={"qid": qid, "predicted_letter": predicted},
        span=LogicalRecordSpan(index=0, start=0, end=9, terminator=b"\n"),
        raw_sha256="0" * 64,
    )


_MAPPING = {"question_id": "qid", "prediction": "predicted_letter"}


def test_derive_overlay_valid_inference_repair():
    authorization = _authorization(
        authorization_type="inference_repair", executable=True, question_reasons={"q1": "malformed"}
    )
    overlay = _overlay(
        result_origin=ResultOriginSpec(
            derivation_origin="repair_overlay",
            default_prediction_origin=None,
            per_question_prediction_origins={"q1": "native_inference"},
        ),
        replacement_reasons={"q1": "repair malformed answer"},
    )
    derived = derive_overlay(
        base=_base(), overlay=overlay, authorization=authorization,
        overlay_table=_overlay_table((_overlay_row("q1", "B"),)),
        overlay_mapping=_MAPPING, expected=_dataset(),
    )
    assert derived.replacement_question_ids == ("q1",)
    assert derived.preownership_rows[0]["prediction_origin"] == "native_inference"
    assert derived.preownership_rows[0]["predicted_option"] == "B"
    assert derived.result_origin["derivation_origin"] == "repair_overlay"
    assert derived.lineage_components[0]["identity"]["operation_type"] == "repair_overlay"
    assert derived.lineage_components[0]["identity"]["authorization_digest"] == _AUTH_BUNDLE_DIGEST


def test_derive_overlay_valid_offline_transformation_retains_underlying_origin():
    authorization = _authorization(
        authorization_type="offline_transformation", executable=False, question_reasons={"q2": "rematch"}
    )
    overlay = _overlay(
        result_origin=ResultOriginSpec(
            derivation_origin="offline_transformation",
            default_prediction_origin=None,
            per_question_prediction_origins={},
        ),
        replacement_reasons={"q2": "offline semantic rematch"},
    )
    derived = derive_overlay(
        base=_base(), overlay=overlay, authorization=authorization,
        overlay_table=_overlay_table((_overlay_row("q2", "A"),)),
        overlay_mapping=_MAPPING, expected=_dataset(),
    )
    # base.prediction_origins["q2"] == "external_historical_inference" -- retained, not reassigned
    assert derived.preownership_rows[0]["prediction_origin"] == "external_historical_inference"
    assert derived.result_origin["derivation_origin"] == "offline_transformation"


def test_derive_overlay_refuses_cross_condition_authorization():
    authorization = _authorization(
        authorization_type="inference_repair", executable=True, question_reasons={"q1": "malformed"}
    )
    other_base = _base(condition_digest="e" * 64)
    overlay = _overlay(
        result_origin=ResultOriginSpec(
            derivation_origin="repair_overlay", default_prediction_origin=None,
            per_question_prediction_origins={"q1": "native_inference"},
        ),
        replacement_reasons={"q1": "reason"},
        base_condition_digest="e" * 64,
    )
    with pytest.raises(OverlayError, match="condition"):
        derive_overlay(
            base=other_base, overlay=overlay, authorization=authorization,
            overlay_table=_overlay_table((_overlay_row("q1", "B"),)),
            overlay_mapping=_MAPPING, expected=_dataset(),
        )


def test_derive_overlay_refuses_base_realization_mismatch():
    authorization = _authorization(
        authorization_type="inference_repair", executable=True, question_reasons={"q1": "malformed"}
    )
    overlay = _overlay(
        result_origin=ResultOriginSpec(
            derivation_origin="repair_overlay", default_prediction_origin=None,
            per_question_prediction_origins={"q1": "native_inference"},
        ),
        replacement_reasons={"q1": "reason"},
        base_realization_digest="f" * 64,  # forged
    )
    with pytest.raises(OverlayError, match="base"):
        derive_overlay(
            base=_base(), overlay=overlay, authorization=authorization,
            overlay_table=_overlay_table((_overlay_row("q1", "B"),)),
            overlay_mapping=_MAPPING, expected=_dataset(),
        )


def test_derive_overlay_refuses_evidence_digest_mismatch():
    authorization = _authorization(
        authorization_type="inference_repair", executable=True, question_reasons={"q1": "malformed"}
    )
    overlay = _overlay(
        result_origin=ResultOriginSpec(
            derivation_origin="repair_overlay", default_prediction_origin=None,
            per_question_prediction_origins={"q1": "native_inference"},
        ),
        replacement_reasons={"q1": "reason"},
        base_evidence_digests={"ev1": "9" * 64},  # diverges from authorization/base
    )
    with pytest.raises(OverlayError, match="evidence"):
        derive_overlay(
            base=_base(), overlay=overlay, authorization=authorization,
            overlay_table=_overlay_table((_overlay_row("q1", "B"),)),
            overlay_mapping=_MAPPING, expected=_dataset(),
        )


def test_derive_overlay_refuses_unauthorized_replacement():
    authorization = _authorization(
        authorization_type="inference_repair", executable=True, question_reasons={"q1": "malformed"}
    )
    overlay = _overlay(
        result_origin=ResultOriginSpec(
            derivation_origin="repair_overlay", default_prediction_origin=None,
            per_question_prediction_origins={"q2": "native_inference"},
        ),
        replacement_reasons={"q2": "not actually authorized"},
    )
    with pytest.raises(OverlayError, match="not granted"):
        derive_overlay(
            base=_base(), overlay=overlay, authorization=authorization,
            overlay_table=_overlay_table((_overlay_row("q2", "A"),)),
            overlay_mapping=_MAPPING, expected=_dataset(),
        )


def test_derive_overlay_refuses_missing_repair_origin_assignment():
    authorization = _authorization(
        authorization_type="inference_repair", executable=True, question_reasons={"q1": "malformed"}
    )
    overlay = _overlay(
        result_origin=ResultOriginSpec(
            derivation_origin="repair_overlay", default_prediction_origin=None,
            per_question_prediction_origins={},  # no assignment for q1
        ),
        replacement_reasons={"q1": "reason"},
    )
    with pytest.raises(OverlayError, match="origin assignment"):
        derive_overlay(
            base=_base(), overlay=overlay, authorization=authorization,
            overlay_table=_overlay_table((_overlay_row("q1", "B"),)),
            overlay_mapping=_MAPPING, expected=_dataset(),
        )


def test_derive_overlay_refuses_invalid_repair_origin_value():
    authorization = _authorization(
        authorization_type="inference_repair", executable=True, question_reasons={"q1": "malformed"}
    )
    overlay = _overlay(
        result_origin=ResultOriginSpec(
            derivation_origin="repair_overlay", default_prediction_origin=None,
            per_question_prediction_origins={"q1": "mixed"},
        ),
        replacement_reasons={"q1": "reason"},
    )
    with pytest.raises(OverlayError, match="valid repair prediction origin"):
        derive_overlay(
            base=_base(), overlay=overlay, authorization=authorization,
            overlay_table=_overlay_table((_overlay_row("q1", "B"),)),
            overlay_mapping=_MAPPING, expected=_dataset(),
        )


def test_derive_overlay_refuses_duplicate_replacement_row():
    authorization = _authorization(
        authorization_type="inference_repair", executable=True, question_reasons={"q1": "malformed"}
    )
    overlay = _overlay(
        result_origin=ResultOriginSpec(
            derivation_origin="repair_overlay", default_prediction_origin=None,
            per_question_prediction_origins={"q1": "native_inference"},
        ),
        replacement_reasons={"q1": "reason"},
    )
    with pytest.raises(OverlayError, match="exactly one"):
        derive_overlay(
            base=_base(), overlay=overlay, authorization=authorization,
            overlay_table=_overlay_table((_overlay_row("q1", "B"), _overlay_row("q1", "A"))),
            overlay_mapping=_MAPPING, expected=_dataset(),
        )


def test_derive_overlay_refuses_authorization_bundle_mismatch():
    authorization = _authorization(
        authorization_type="inference_repair", executable=True, question_reasons={"q1": "malformed"}
    )
    overlay = _overlay(
        result_origin=ResultOriginSpec(
            derivation_origin="repair_overlay", default_prediction_origin=None,
            per_question_prediction_origins={"q1": "native_inference"},
        ),
        replacement_reasons={"q1": "reason"},
        authorization_id="auth_" + "9" * 16,  # different bundle than the one supplied
    )
    with pytest.raises(OverlayError, match="authorization bundle"):
        derive_overlay(
            base=_base(), overlay=overlay, authorization=authorization,
            overlay_table=_overlay_table((_overlay_row("q1", "B"),)),
            overlay_mapping=_MAPPING, expected=_dataset(),
        )
