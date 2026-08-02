"""Validate typed repair/offline-transformation authorization bundles.

An authorization bundle grants specific (condition_digest, question_id) pairs
permission to be replaced, typed as either `inference_repair` (executable --
a real model may be run) or `offline_transformation` (non-executing semantic
rematching only). This module never runs inference or opens a filesystem
path itself; it validates an already-opened source and already-known
condition digests/expected datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from choicebench.identity import integrity_digest, short_id
from choicebench.importing.csv_adapter import OpenedSource
from choicebench.importing.dataset_reference import ExpectedDataset
from choicebench.importing.schema import AuthorizationSpec


class AuthorizationError(ValueError):
    """Raised when an authorization bundle or its use is invalid or unsafe."""


@dataclass(frozen=True)
class ValidatedAuthorizationBundle:
    authorization_id: str
    authorization_digest: str
    authorization_type: str
    grants: Mapping[str, Mapping[str, str]]  # condition_digest -> question_id -> reason
    authority: str
    purpose: str
    executable: bool
    source_sha256: str
    input_evidence_digests: Mapping[str, str]
    expected_snapshot_digests: Mapping[str, str]


@dataclass(frozen=True)
class ValidatedAuthorization:
    bundle_id: str
    bundle_digest: str
    authorization_type: str
    condition_digest: str
    question_reasons: Mapping[str, str]
    authority: str
    purpose: str
    executable: bool
    source_sha256: str
    input_evidence_digests: Mapping[str, str]
    expected_snapshot_digest: str


def validate_authorization_bundle(
    declaration: AuthorizationSpec,
    *,
    opened_source: OpenedSource,
    condition_digests: Mapping[str, str],
    expected: Mapping[str, ExpectedDataset],
) -> ValidatedAuthorizationBundle:
    """Validate one authorization artifact against an independently opened
    source and the caller's own known condition digests/expected datasets.

    Fails closed on: source tampering, an authorization_type/executable
    mismatch, a grant referencing an unknown condition or an out-of-selection
    question, an empty reason, or a self-assigned authorization_id that does
    not match the bundle's own recomputed digest.
    """
    if opened_source.source_id != declaration.source_id:
        raise AuthorizationError(
            f"Authorization {declaration.authorization_id!r} declares source "
            f"{declaration.source_id!r} but was validated against a differently "
            f"declared opened source {opened_source.source_id!r}."
        )
    if declaration.authorization_type == "inference_repair" and not declaration.executable:
        raise AuthorizationError(
            f"Authorization {declaration.authorization_id!r} declares inference_repair "
            "but executable=false."
        )
    if declaration.authorization_type == "offline_transformation" and declaration.executable:
        raise AuthorizationError(
            f"Authorization {declaration.authorization_id!r} declares "
            "offline_transformation but executable=true."
        )

    grants: dict[str, dict[str, str]] = {}
    for condition_digest, question_reasons in declaration.condition_question_reasons.items():
        if condition_digest not in condition_digests.values():
            raise AuthorizationError(
                f"Authorization {declaration.authorization_id!r} grants a condition "
                f"{condition_digest!r} that is not among the caller's known conditions."
            )
        dataset = expected.get(condition_digest)
        if not isinstance(question_reasons, Mapping) or not question_reasons:
            raise AuthorizationError(
                f"Authorization {declaration.authorization_id!r} has an empty grant for "
                f"condition {condition_digest!r}."
            )
        selected = set(dataset.selected_question_ids) if dataset is not None else None
        normalized_reasons: dict[str, str] = {}
        for question_id, reason in question_reasons.items():
            if not isinstance(reason, str) or not reason.strip():
                raise AuthorizationError(
                    f"Authorization {declaration.authorization_id!r} grant for "
                    f"{condition_digest!r}/{question_id!r} has no explicit reason."
                )
            if selected is not None and question_id not in selected:
                raise AuthorizationError(
                    f"Authorization {declaration.authorization_id!r} grants question "
                    f"{question_id!r} that is not in condition {condition_digest!r}'s "
                    "expected selection."
                )
            normalized_reasons[question_id] = reason
        grants[condition_digest] = normalized_reasons
    if not grants:
        raise AuthorizationError(
            f"Authorization {declaration.authorization_id!r} grants no conditions."
        )

    bundle_payload = {
        "schema_version": "choicebench.authorization-bundle.v1",
        "authorization_type": declaration.authorization_type,
        "grants": grants,
        "authority": declaration.authority,
        "purpose": declaration.purpose,
        "executable": declaration.executable,
        "source_sha256": opened_source.sha256,
        "input_evidence_digests": dict(declaration.input_evidence_digests),
        "expected_snapshot_digests": dict(declaration.expected_snapshot_digests),
    }
    authorization_digest = integrity_digest(bundle_payload)
    expected_id = short_id("auth", bundle_payload)
    if declaration.authorization_id != expected_id:
        raise AuthorizationError(
            f"Authorization ID {declaration.authorization_id!r} does not match its own "
            f"recomputed digest (expected {expected_id!r}); it cannot self-assign an ID."
        )
    if set(declaration.expected_snapshot_digests) != set(grants):
        raise AuthorizationError(
            f"Authorization {declaration.authorization_id!r} expected_snapshot_digests "
            "do not exactly cover its granted conditions."
        )

    return ValidatedAuthorizationBundle(
        authorization_id=expected_id,
        authorization_digest=authorization_digest,
        authorization_type=declaration.authorization_type,
        grants=grants,
        authority=declaration.authority,
        purpose=declaration.purpose,
        executable=declaration.executable,
        source_sha256=opened_source.sha256,
        input_evidence_digests=dict(declaration.input_evidence_digests),
        expected_snapshot_digests=dict(declaration.expected_snapshot_digests),
    )


def authorization_for_condition(
    bundle: ValidatedAuthorizationBundle, *, condition_digest: str
) -> ValidatedAuthorization:
    """Slice one condition's grant out of a bundle without weakening its
    aggregate identity: bundle_id/bundle_digest are the whole bundle's, so a
    slice can be traced back to (and cannot silently diverge from) the
    exact bundle it came from, and it can never borrow another condition's
    grants."""
    question_reasons = bundle.grants.get(condition_digest)
    if question_reasons is None:
        raise AuthorizationError(
            f"Authorization {bundle.authorization_id!r} grants no questions for "
            f"condition {condition_digest!r}; this condition cannot self-authorize."
        )
    expected_snapshot_digest = bundle.expected_snapshot_digests.get(condition_digest)
    if expected_snapshot_digest is None:
        raise AuthorizationError(
            f"Authorization {bundle.authorization_id!r} has no expected snapshot digest "
            f"for condition {condition_digest!r}."
        )
    return ValidatedAuthorization(
        bundle_id=bundle.authorization_id,
        bundle_digest=bundle.authorization_digest,
        authorization_type=bundle.authorization_type,
        condition_digest=condition_digest,
        question_reasons=dict(question_reasons),
        authority=bundle.authority,
        purpose=bundle.purpose,
        executable=bundle.executable,
        source_sha256=bundle.source_sha256,
        input_evidence_digests=dict(bundle.input_evidence_digests),
        expected_snapshot_digest=expected_snapshot_digest,
    )
