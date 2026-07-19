"""Tests for typed authorization bundle validation (Task 10, reduced scope)."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from choicebench.identity import integrity_digest, short_id
from choicebench.importing.authorization import (
    AuthorizationError,
    authorization_for_condition,
    validate_authorization_bundle,
)
from choicebench.importing.csv_adapter import OpenedSource
from choicebench.importing.schema import AuthorizationSpec
from tests.importing.test_identity import _dataset


def _source(data: bytes = b"authorization,payload\n1,2\n") -> OpenedSource:
    return OpenedSource(
        source_id="auth-source",
        audit_path=Path("/machine-a/authorization.csv"),
        logical_path="inputs/authorization.csv",
        data=data,
        sha256=sha256(data).hexdigest(),
    )


def _bundle_payload(*, authorization_type, executable, condition_digest, grants, source, purpose="repair"):
    return {
        "schema_version": "choicebench.authorization-bundle.v1",
        "authorization_type": authorization_type,
        "grants": {condition_digest: dict(grants)},
        "authority": "principal-investigator",
        "purpose": purpose,
        "executable": executable,
        "source_sha256": source.sha256,
        "input_evidence_digests": {"ev1": "1" * 64},
        "expected_snapshot_digests": {condition_digest: "2" * 64},
    }


def _declaration(*, authorization_type, executable, condition_digest, grants, source, authorization_id=None):
    payload = _bundle_payload(
        authorization_type=authorization_type, executable=executable,
        condition_digest=condition_digest, grants=grants, source=source,
    )
    computed_id = short_id("auth", payload)
    return AuthorizationSpec(
        authorization_id=authorization_id or computed_id,
        authorization_type=authorization_type,
        source_id="auth-source",
        condition_question_reasons={condition_digest: dict(grants)},
        authority="principal-investigator",
        purpose="repair",
        executable=executable,
        input_evidence_digests={"ev1": "1" * 64},
        expected_snapshot_digests={condition_digest: "2" * 64},
    )


def test_validate_authorization_bundle_accepts_a_well_formed_inference_repair_grant():
    expected = _dataset()
    condition_digest = "c" * 64
    source = _source()
    declaration = _declaration(
        authorization_type="inference_repair", executable=True,
        condition_digest=condition_digest, grants={"q1": "malformed prediction"}, source=source,
    )
    bundle = validate_authorization_bundle(
        declaration, opened_source=source,
        condition_digests={"cond_key": condition_digest},
        expected={condition_digest: expected},
    )
    assert bundle.authorization_type == "inference_repair"
    assert bundle.grants[condition_digest] == {"q1": "malformed prediction"}


def test_validate_authorization_bundle_rejects_source_id_mismatch():
    expected = _dataset()
    condition_digest = "c" * 64
    source = _source()
    declaration = _declaration(
        authorization_type="inference_repair", executable=True,
        condition_digest=condition_digest, grants={"q1": "reason"}, source=source,
    )
    wrong_id_source = OpenedSource(
        source_id="a-different-source-id",
        audit_path=Path("/machine-a/authorization.csv"),
        logical_path="inputs/authorization.csv",
        data=source.data,
        sha256=source.sha256,
    )
    with pytest.raises(AuthorizationError, match="source"):
        validate_authorization_bundle(
            declaration, opened_source=wrong_id_source,
            condition_digests={"cond_key": condition_digest},
            expected={condition_digest: expected},
        )


def test_validate_authorization_bundle_rejects_tampered_source_bytes():
    """Bundle identity is bound to the exact opened source bytes: swapping the
    source content for a different byte-identical-source_id file changes the
    recomputed authorization_id, so a declared ID from the original bytes is
    refused rather than silently accepted against different content."""
    expected = _dataset()
    condition_digest = "c" * 64
    source = _source()
    declaration = _declaration(
        authorization_type="inference_repair", executable=True,
        condition_digest=condition_digest, grants={"q1": "reason"}, source=source,
    )
    tampered_source = _source(data=b"different,bytes\n9,9\n")
    with pytest.raises(AuthorizationError, match="recomputed digest"):
        validate_authorization_bundle(
            declaration, opened_source=tampered_source,
            condition_digests={"cond_key": condition_digest},
            expected={condition_digest: expected},
        )


def test_validate_authorization_bundle_rejects_inference_repair_marked_nonexecutable():
    expected = _dataset()
    condition_digest = "c" * 64
    source = _source()
    declaration = _declaration(
        authorization_type="inference_repair", executable=False,
        condition_digest=condition_digest, grants={"q1": "reason"}, source=source,
    )
    with pytest.raises(AuthorizationError, match="executable"):
        validate_authorization_bundle(
            declaration, opened_source=source,
            condition_digests={"cond_key": condition_digest},
            expected={condition_digest: expected},
        )


def test_validate_authorization_bundle_rejects_offline_transformation_marked_executable():
    expected = _dataset()
    condition_digest = "c" * 64
    source = _source()
    declaration = _declaration(
        authorization_type="offline_transformation", executable=True,
        condition_digest=condition_digest, grants={"q1": "reason"}, source=source,
    )
    with pytest.raises(AuthorizationError, match="executable"):
        validate_authorization_bundle(
            declaration, opened_source=source,
            condition_digests={"cond_key": condition_digest},
            expected={condition_digest: expected},
        )


def test_validate_authorization_bundle_rejects_unknown_condition():
    expected = _dataset()
    condition_digest = "c" * 64
    source = _source()
    declaration = _declaration(
        authorization_type="inference_repair", executable=True,
        condition_digest=condition_digest, grants={"q1": "reason"}, source=source,
    )
    with pytest.raises(AuthorizationError, match="known conditions"):
        validate_authorization_bundle(
            declaration, opened_source=source,
            condition_digests={"cond_key": "d" * 64},  # different digest
            expected={condition_digest: expected},
        )


def test_validate_authorization_bundle_rejects_question_outside_selection():
    expected = _dataset()  # selected_question_ids are q2, q1
    condition_digest = "c" * 64
    source = _source()
    declaration = _declaration(
        authorization_type="inference_repair", executable=True,
        condition_digest=condition_digest, grants={"q9": "reason"}, source=source,
    )
    with pytest.raises(AuthorizationError, match="expected selection"):
        validate_authorization_bundle(
            declaration, opened_source=source,
            condition_digests={"cond_key": condition_digest},
            expected={condition_digest: expected},
        )


def test_validate_authorization_bundle_rejects_empty_reason():
    expected = _dataset()
    condition_digest = "c" * 64
    source = _source()
    declaration = _declaration(
        authorization_type="inference_repair", executable=True,
        condition_digest=condition_digest, grants={"q1": "   "}, source=source,
    )
    with pytest.raises(AuthorizationError, match="explicit reason"):
        validate_authorization_bundle(
            declaration, opened_source=source,
            condition_digests={"cond_key": condition_digest},
            expected={condition_digest: expected},
        )


def test_validate_authorization_bundle_rejects_self_assigned_id():
    expected = _dataset()
    condition_digest = "c" * 64
    source = _source()
    declaration = _declaration(
        authorization_type="inference_repair", executable=True,
        condition_digest=condition_digest, grants={"q1": "reason"}, source=source,
        authorization_id="auth_" + "0" * 16,
    )
    with pytest.raises(AuthorizationError, match="recomputed digest"):
        validate_authorization_bundle(
            declaration, opened_source=source,
            condition_digests={"cond_key": condition_digest},
            expected={condition_digest: expected},
        )


def test_authorization_for_condition_slices_one_condition():
    expected = _dataset()
    condition_digest = "c" * 64
    source = _source()
    declaration = _declaration(
        authorization_type="offline_transformation", executable=False,
        condition_digest=condition_digest, grants={"q1": "reason"}, source=source,
    )
    bundle = validate_authorization_bundle(
        declaration, opened_source=source,
        condition_digests={"cond_key": condition_digest},
        expected={condition_digest: expected},
    )
    sliced = authorization_for_condition(bundle, condition_digest=condition_digest)
    assert sliced.bundle_id == bundle.authorization_id
    assert sliced.bundle_digest == bundle.authorization_digest
    assert sliced.question_reasons == {"q1": "reason"}


def test_authorization_for_condition_refuses_unauthorized_condition():
    expected = _dataset()
    condition_digest = "c" * 64
    source = _source()
    declaration = _declaration(
        authorization_type="offline_transformation", executable=False,
        condition_digest=condition_digest, grants={"q1": "reason"}, source=source,
    )
    bundle = validate_authorization_bundle(
        declaration, opened_source=source,
        condition_digests={"cond_key": condition_digest},
        expected={condition_digest: expected},
    )
    with pytest.raises(AuthorizationError, match="self-authorize"):
        authorization_for_condition(bundle, condition_digest="d" * 64)
