"""Additive manifest-v3 tests: new schema alongside the untouched v2 path.

Scope is deliberately reduced from the original plan: no ManifestView
cross-version normalization and no legacy-v2-in-v3 synthesis. Every manifest
this module builds is a fresh v3 import; existing v2 manifests keep using the
unmodified v2 functions exercised by tests/importing/test_manifest_v2_compat.py.
"""

from __future__ import annotations

import pytest

from choicebench.identity import CANONICALIZATION_VERSION, integrity_digest, short_id
from choicebench.manifest import (
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_V3_SCHEMA_VERSION,
    PROTOCOL_V3_VERSION,
    RUN_STATE_V3_SCHEMA_VERSION,
    ManifestCompatibilityError,
    initial_run_state_v3,
    make_manifest_v3,
    validate_manifest,
    validate_manifest_v3,
    validate_run_state_v3,
)


def _condition_identity_triple():
    """A condition_id/digest pair that genuinely matches its own identity payload,
    so validate_manifest_v3's recomputation check passes on the happy path."""
    identity = {"seed": 1}
    digest = integrity_digest(identity)
    condition_id = short_id("cond", identity)
    return condition_id, digest, identity


def _realization_record(condition_id, condition_digest, realization_identity):
    payload = {
        "schema_version": "choicebench.realization.v1",
        "condition_id": condition_id,
        "condition_digest": condition_digest,
        "realization": realization_identity,
    }
    return {
        "realization_id": short_id("real", payload),
        "realization_digest": integrity_digest(payload),
        "condition_id": condition_id,
        "condition_digest": condition_digest,
        "identity": payload,
    }


@pytest.fixture
def v3_payload():
    condition_id, condition_digest, identity = _condition_identity_triple()
    condition = {
        "condition_id": condition_id,
        "condition_digest": condition_digest,
        "identity": identity,
    }
    realization = _realization_record(condition_id, condition_digest, {"source": "x"})
    return {
        "protocol_version": PROTOCOL_V3_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "semantic_conditions": {condition_id: condition},
        "realizations": {realization["realization_id"]: realization},
    }, condition_id, realization["realization_id"]


def test_make_manifest_v3_computes_schema_experiment_and_payload_digests(v3_payload):
    payload, condition_id, realization_id = v3_payload
    manifest = make_manifest_v3(payload)
    assert manifest["schema_version"] == MANIFEST_V3_SCHEMA_VERSION
    assert manifest["experiment_id"].startswith("exp_")
    assert manifest["experiment_digest"] == integrity_digest(manifest["payload"])
    assert manifest["payload_digest"] == integrity_digest(manifest["payload"])
    assert manifest["payload"]["semantic_conditions"][condition_id]["condition_id"] == condition_id
    assert realization_id in manifest["payload"]["realizations"]


def test_make_manifest_v3_excludes_audit_from_identity(v3_payload):
    payload, _, _ = v3_payload
    first = make_manifest_v3(payload, audit={"source_location": "/a/one"})
    second = make_manifest_v3(payload, audit={"source_location": "/b/two"})
    assert first["experiment_id"] == second["experiment_id"]
    assert first["experiment_digest"] == second["experiment_digest"]
    assert first["audit"] != second["audit"]


def test_validate_manifest_v3_accepts_a_well_formed_manifest(v3_payload):
    payload, _, _ = v3_payload
    manifest = make_manifest_v3(payload)
    validate_manifest_v3(manifest)  # must not raise


def test_validate_manifest_v3_rejects_wrong_schema_version(v3_payload):
    payload, _, _ = v3_payload
    manifest = make_manifest_v3(payload)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    with pytest.raises(ManifestCompatibilityError):
        validate_manifest_v3(manifest)


def test_validate_manifest_v3_rejects_tampered_payload_digest(v3_payload):
    payload, _, _ = v3_payload
    manifest = make_manifest_v3(payload)
    manifest["payload"]["semantic_conditions"] = {}
    with pytest.raises(ManifestCompatibilityError):
        validate_manifest_v3(manifest)


def test_validate_manifest_v3_recomputes_condition_id_and_digest(v3_payload):
    payload, condition_id, _ = v3_payload
    manifest = make_manifest_v3(payload)
    forged = dict(manifest["payload"]["semantic_conditions"][condition_id])
    forged["condition_digest"] = "f" * 64
    manifest["payload"]["semantic_conditions"][condition_id] = forged
    manifest["payload_digest"] = integrity_digest(manifest["payload"])
    manifest["experiment_digest"] = integrity_digest(manifest["payload"])
    manifest["experiment_id"] = f"exp_{manifest['experiment_digest'][:16]}"
    with pytest.raises(ManifestCompatibilityError, match="condition"):
        validate_manifest_v3(manifest)


def test_validate_manifest_v3_recomputes_realization_id_and_digest(v3_payload):
    payload, _, realization_id = v3_payload
    manifest = make_manifest_v3(payload)
    forged = dict(manifest["payload"]["realizations"][realization_id])
    forged["realization_digest"] = "f" * 64
    manifest["payload"]["realizations"][realization_id] = forged
    manifest["payload_digest"] = integrity_digest(manifest["payload"])
    manifest["experiment_digest"] = integrity_digest(manifest["payload"])
    manifest["experiment_id"] = f"exp_{manifest['experiment_digest'][:16]}"
    with pytest.raises(ManifestCompatibilityError, match="realization"):
        validate_manifest_v3(manifest)


def test_validate_manifest_v3_requires_realization_condition_binding(v3_payload):
    payload, condition_id, realization_id = v3_payload
    manifest = make_manifest_v3(payload)
    forged_realization = dict(manifest["payload"]["realizations"][realization_id])
    forged_realization["condition_digest"] = "e" * 64
    manifest["payload"]["realizations"][realization_id] = forged_realization
    manifest["payload_digest"] = integrity_digest(manifest["payload"])
    manifest["experiment_digest"] = integrity_digest(manifest["payload"])
    manifest["experiment_id"] = f"exp_{manifest['experiment_digest'][:16]}"
    with pytest.raises(ManifestCompatibilityError, match="condition"):
        validate_manifest_v3(manifest)


def test_validate_manifest_v3_rejects_realization_with_unknown_condition(v3_payload):
    payload, condition_id, realization_id = v3_payload
    manifest = make_manifest_v3(payload)
    del manifest["payload"]["semantic_conditions"][condition_id]
    manifest["payload_digest"] = integrity_digest(manifest["payload"])
    manifest["experiment_digest"] = integrity_digest(manifest["payload"])
    manifest["experiment_id"] = f"exp_{manifest['experiment_digest'][:16]}"
    with pytest.raises(ManifestCompatibilityError, match="condition"):
        validate_manifest_v3(manifest)


def test_run_state_v3_is_keyed_by_realization(v3_payload):
    payload, _, realization_id = v3_payload
    manifest = make_manifest_v3(payload)
    state = initial_run_state_v3(manifest)
    assert state["schema_version"] == RUN_STATE_V3_SCHEMA_VERSION
    assert state["experiment_id"] == manifest["experiment_id"]
    assert state["realizations"] == {realization_id: {"status": "pending"}}
    validate_run_state_v3(state, manifest)  # must not raise


def test_run_state_v3_rejects_state_for_a_different_manifest(v3_payload):
    payload, _, _ = v3_payload
    manifest = make_manifest_v3(payload)
    other_manifest = make_manifest_v3(
        {**payload, "semantic_conditions": {}, "realizations": {}}
    )
    state = initial_run_state_v3(manifest)
    with pytest.raises(ManifestCompatibilityError):
        validate_run_state_v3(state, other_manifest)


def test_run_state_v3_rejects_realization_grid_drift(v3_payload):
    payload, _, realization_id = v3_payload
    manifest = make_manifest_v3(payload)
    state = initial_run_state_v3(manifest)
    del state["realizations"][realization_id]
    with pytest.raises(ManifestCompatibilityError):
        validate_run_state_v3(state, manifest)


def test_v2_manifest_functions_are_unaffected_by_v3_additions():
    """Sanity check that importing the v3 symbols doesn't change v2 behavior."""
    v2_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": "exp_0000000000000000",
        "experiment_digest": "0" * 64,
        "payload_digest": "0" * 64,
        "created_at": "now",
        "payload": {"protocol_version": "choicebench.protocol.v2", "config": {}},
    }
    with pytest.raises(ManifestCompatibilityError):
        validate_manifest(v2_manifest)  # digest mismatch, but still v2-only code path
