"""Tests for the overlay merge-into-new-run orchestration
(execute_overlay_import / _merge_overlay_realization). Builds a real base run
via test_engine.py's fixtures, then applies an authorized overlay on top.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from choicebench.identity import short_id
from choicebench.importing.dataset_reference import build_expected_dataset
from choicebench.importing.engine import (
    ImportEngineError,
    ImportRequest,
    OverlayImportRequest,
    execute_import,
    execute_overlay_import,
    verify_import_run,
)
from choicebench.importing.evidence import open_verified_source
from choicebench.importing.schema import (
    AuthorizationSpec,
    CsvDialectSpec,
    OptionMappingSpec,
    OverlaySpec,
    ResultOriginSpec,
    SourceArtifactSpec,
)
from tests.importing.test_engine import _spec

_AUTH_CSV = b"authorization,payload\n1,2\n"
_OVERLAY_CSV = "qid,predicted_letter\nq1,B\n"


def _make_base_run(tmp_path: Path):
    spec = _spec(tmp_path)
    request = ImportRequest(spec=spec, run_id="base-run", workspace_root=tmp_path, strict=True)
    execute_import(request, dry_run=False)
    base_run_dir = tmp_path / "runs" / "base-run"
    verified = verify_import_run(base_run_dir)
    (base,) = verified.realizations.values()
    return base_run_dir, base


def _base_expected_dataset(tmp_path: Path):
    spec = _spec(tmp_path)
    (dataset_ref,) = spec.datasets
    (results_source,) = spec.sources
    opened_results = open_verified_source(results_source, containment_root=tmp_path)
    return build_expected_dataset(dataset_ref, opened_sources={"results": opened_results})


def _authorization_source(tmp_path: Path) -> SourceArtifactSpec:
    auth_path = tmp_path / "authorization.csv"
    auth_path.write_bytes(_AUTH_CSV)
    return SourceArtifactSpec(
        source_id="auth-source",
        path=auth_path,
        logical_path="inputs/authorization.csv",
        expected_sha256=sha256(_AUTH_CSV).hexdigest(),
        format="csv",
        format_version="producer-v1",
        classification="raw",
        dialect=CsvDialectSpec(),
        columns={},
        expected_columns=("authorization", "payload"),
        ignored_columns={},
        null_values=(),
        numeric_columns=(),
        option_mapping=OptionMappingSpec(
            mode="ordered_columns", ordered_columns=(), structured_column=None,
            structured_label_key=None, structured_text_key=None,
        ),
        extra_field_policy="preserve_unmapped",
        preserve_namespace="producer",
        source_run_id=None, source_repository=None, source_commit=None,
        notes={},
    )


def _overlay_source(tmp_path: Path, csv_text: str = _OVERLAY_CSV, filename: str = "overlay.csv") -> SourceArtifactSpec:
    overlay_path = tmp_path / filename
    overlay_path.write_bytes(csv_text.encode("utf-8"))
    return SourceArtifactSpec(
        source_id="overlay-source",
        path=overlay_path,
        logical_path="inputs/overlay.csv",
        expected_sha256=sha256(csv_text.encode("utf-8")).hexdigest(),
        format="csv",
        format_version="producer-v1",
        classification="repaired",
        dialect=CsvDialectSpec(),
        columns={"question_id": "qid", "prediction": "predicted_letter"},
        expected_columns=("qid", "predicted_letter"),
        ignored_columns={},
        null_values=(),
        numeric_columns=(),
        option_mapping=OptionMappingSpec(
            mode="ordered_columns", ordered_columns=(), structured_column=None,
            structured_label_key=None, structured_text_key=None,
        ),
        extra_field_policy="preserve_unmapped",
        preserve_namespace="producer",
        source_run_id=None, source_repository=None, source_commit=None,
        notes={},
    )


def _authorization(
    *, base, authorization_type: str, executable: bool,
    grants: dict[str, str] | None = None,
) -> AuthorizationSpec:
    grants = grants or {"q1": "malformed prediction"}
    payload = {
        "schema_version": "choicebench.authorization-bundle.v1",
        "authorization_type": authorization_type,
        "grants": {base.condition_digest: grants},
        "authority": "principal-investigator",
        "purpose": "repair",
        "executable": executable,
        "source_sha256": sha256(_AUTH_CSV).hexdigest(),
        "input_evidence_digests": dict(base.evidence_source_digests),
        "expected_snapshot_digests": {base.condition_digest: "2" * 64},
    }
    authorization_id = short_id("auth", payload)
    return AuthorizationSpec(
        authorization_id=authorization_id,
        authorization_type=authorization_type,
        source_id="auth-source",
        condition_question_reasons={base.condition_digest: grants},
        authority="principal-investigator",
        purpose="repair",
        executable=executable,
        input_evidence_digests=dict(base.evidence_source_digests),
        expected_snapshot_digests={base.condition_digest: "2" * 64},
    )


def _overlay_spec(*, base, authorization: AuthorizationSpec) -> OverlaySpec:
    return OverlaySpec(
        overlay_id="ovl_1",
        base_run_path=Path("/tmp/base-run"),
        base_condition_digest=base.condition_digest,
        base_realization_id=base.realization_id,
        base_realization_digest=base.realization_digest,
        base_evidence_digests=dict(base.evidence_source_digests),
        base_validation_artifact_sha256=base.validation_artifact_sha256,
        base_result_sha256=base.result_sha256,
        source_id="overlay-source",
        authorization_id=authorization.authorization_id,
        replacement_reasons={"q1": "repair malformed answer"},
        result_origin=ResultOriginSpec(
            derivation_origin="repair_overlay",
            default_prediction_origin=None,
            per_question_prediction_origins={"q1": "native_inference"},
        ),
        lineage_notes={},
        implementation={
            "qualified_name": "external:repair_tool",
            "source_digest": sha256(_OVERLAY_CSV.encode("utf-8")).hexdigest(),
        },
        input_digest="9" * 64,
        preownership_output_digest="a" * 64,
        expected_evidence_status="complete",
    )


def _make_overlay_request(tmp_path: Path, base_run_dir: Path, base, *, run_id: str = "overlay-run"):
    auth_source = _authorization_source(tmp_path)
    authorization = _authorization(base=base, authorization_type="inference_repair", executable=True)
    overlay_source = _overlay_source(tmp_path)
    overlay = _overlay_spec(base=base, authorization=authorization)
    expected_dataset = _base_expected_dataset(tmp_path)

    return OverlayImportRequest(
        base_run_dir=base_run_dir,
        base_realization_id=base.realization_id,
        authorization=authorization,
        authorization_source=auth_source,
        overlay=overlay,
        overlay_source=overlay_source,
        overlay_mapping={"question_id": "qid", "prediction": "predicted_letter"},
        condition_digests={"cond_key": base.condition_digest},
        expected_dataset=expected_dataset,
        run_id=run_id,
        workspace_root=tmp_path,
    )


def test_execute_overlay_import_merges_repair_into_a_new_run(tmp_path):
    base_run_dir, base = _make_base_run(tmp_path)
    request = _make_overlay_request(tmp_path, base_run_dir, base)

    report = execute_overlay_import(request, dry_run=False)
    assert report.import_state == "imported"
    assert report.wrote_artifacts is True
    assert report.idempotent_noop is False

    overlay_run_dir = tmp_path / "runs" / "overlay-run"
    verified = verify_import_run(overlay_run_dir)
    (derived,) = verified.realizations.values()
    assert set(derived.rows_by_question_id) == {"q1", "q2"}
    assert derived.rows_by_question_id["q1"]["predicted_option"] == "B"
    assert derived.prediction_origins["q1"] == "native_inference"
    assert derived.prediction_origins["q2"] == "external_historical_inference"
    assert (overlay_run_dir / "artifacts/imports/evidence/index.json").is_file()

    # the immutable base run is untouched
    reverified_base = verify_import_run(base_run_dir)
    (base_again,) = reverified_base.realizations.values()
    assert base_again.realization_digest == base.realization_digest


def test_execute_overlay_import_merges_offline_transformation_retaining_origin(tmp_path):
    base_run_dir, base = _make_base_run(tmp_path)

    auth_source = _authorization_source(tmp_path)
    authorization = _authorization(
        base=base, authorization_type="offline_transformation", executable=False,
        grants={"q2": "offline semantic rematch"},
    )
    overlay_csv = "qid,predicted_letter\nq2,A\n"
    overlay_source = _overlay_source(tmp_path, csv_text=overlay_csv, filename="offline-overlay.csv")
    overlay = OverlaySpec(
        overlay_id="ovl_offline",
        base_run_path=Path("/tmp/base-run"),
        base_condition_digest=base.condition_digest,
        base_realization_id=base.realization_id,
        base_realization_digest=base.realization_digest,
        base_evidence_digests=dict(base.evidence_source_digests),
        base_validation_artifact_sha256=base.validation_artifact_sha256,
        base_result_sha256=base.result_sha256,
        source_id="overlay-source",
        authorization_id=authorization.authorization_id,
        replacement_reasons={"q2": "offline semantic rematch"},
        result_origin=ResultOriginSpec(
            derivation_origin="offline_transformation",
            default_prediction_origin=None,
            per_question_prediction_origins={},
        ),
        lineage_notes={},
        implementation={
            "qualified_name": "external:offline_rematcher",
            "source_digest": sha256(overlay_csv.encode("utf-8")).hexdigest(),
        },
        input_digest="9" * 64,
        preownership_output_digest="a" * 64,
        expected_evidence_status="complete",
    )
    request = OverlayImportRequest(
        base_run_dir=base_run_dir,
        base_realization_id=base.realization_id,
        authorization=authorization,
        authorization_source=auth_source,
        overlay=overlay,
        overlay_source=overlay_source,
        overlay_mapping={"question_id": "qid", "prediction": "predicted_letter"},
        condition_digests={"cond_key": base.condition_digest},
        expected_dataset=_base_expected_dataset(tmp_path),
        run_id="offline-overlay-run",
        workspace_root=tmp_path,
    )

    report = execute_overlay_import(request, dry_run=False)
    assert report.import_state == "imported"

    overlay_run_dir = tmp_path / "runs" / "offline-overlay-run"
    verified = verify_import_run(overlay_run_dir)
    (derived,) = verified.realizations.values()
    assert derived.rows_by_question_id["q2"]["predicted_option"] == "A"
    # offline_transformation retains the underlying response's own origin
    # rather than reassigning it (base.prediction_origins["q2"] is
    # external_historical_inference) -- unlike inference_repair.
    assert derived.prediction_origins["q2"] == "external_historical_inference"
    assert derived.prediction_origins["q1"] == "external_historical_inference"


def test_execute_overlay_import_dry_run_writes_nothing(tmp_path):
    base_run_dir, base = _make_base_run(tmp_path)
    request = _make_overlay_request(tmp_path, base_run_dir, base)

    report = execute_overlay_import(request, dry_run=True)
    assert report.import_state == "validated"
    assert report.wrote_artifacts is False
    assert not (tmp_path / "runs" / "overlay-run").exists()


def test_execute_overlay_import_is_idempotent_noop_on_repeat(tmp_path):
    base_run_dir, base = _make_base_run(tmp_path)
    request = _make_overlay_request(tmp_path, base_run_dir, base)

    execute_overlay_import(request, dry_run=False)
    second = execute_overlay_import(request, dry_run=False)
    assert second.idempotent_noop is True
    assert second.wrote_artifacts is False


def test_execute_overlay_import_rejects_an_unknown_base_realization_id(tmp_path):
    base_run_dir, base = _make_base_run(tmp_path)
    request = _make_overlay_request(tmp_path, base_run_dir, base)
    unknown = OverlayImportRequest(
        base_run_dir=request.base_run_dir,
        base_realization_id="not-the-real-id",
        authorization=request.authorization,
        authorization_source=request.authorization_source,
        overlay=request.overlay,
        overlay_source=request.overlay_source,
        overlay_mapping=request.overlay_mapping,
        condition_digests=request.condition_digests,
        expected_dataset=request.expected_dataset,
        run_id=request.run_id,
        workspace_root=request.workspace_root,
    )
    report = execute_overlay_import(unknown, dry_run=False)
    assert report.import_state == "failed"
    assert report.wrote_artifacts is False
    assert not (tmp_path / "runs" / "overlay-run").exists()


def test_execute_overlay_import_refuses_to_overwrite_a_divergent_existing_run(tmp_path):
    base_run_dir, base = _make_base_run(tmp_path)
    request = _make_overlay_request(tmp_path, base_run_dir, base)
    execute_overlay_import(request, dry_run=False)

    # a second, different overlay (different replaced answer) targeting the
    # same run_id must be refused, not silently treated as a no-op
    divergent_csv = "qid,predicted_letter\nq1,C\n"
    divergent_overlay_source = _overlay_source(tmp_path, csv_text=divergent_csv, filename="divergent-overlay.csv")
    divergent_overlay = OverlaySpec(
        overlay_id="ovl_2",
        base_run_path=request.overlay.base_run_path,
        base_condition_digest=request.overlay.base_condition_digest,
        base_realization_id=request.overlay.base_realization_id,
        base_realization_digest=request.overlay.base_realization_digest,
        base_evidence_digests=request.overlay.base_evidence_digests,
        base_validation_artifact_sha256=request.overlay.base_validation_artifact_sha256,
        base_result_sha256=request.overlay.base_result_sha256,
        source_id="overlay-source",
        authorization_id=request.overlay.authorization_id,
        replacement_reasons={"q1": "repair malformed answer"},
        result_origin=request.overlay.result_origin,
        lineage_notes={},
        implementation={
            "qualified_name": "external:repair_tool",
            "source_digest": sha256(divergent_csv.encode("utf-8")).hexdigest(),
        },
        input_digest="9" * 64,
        preownership_output_digest="a" * 64,
        expected_evidence_status="complete",
    )
    divergent_request = OverlayImportRequest(
        base_run_dir=request.base_run_dir,
        base_realization_id=request.base_realization_id,
        authorization=request.authorization,
        authorization_source=request.authorization_source,
        overlay=divergent_overlay,
        overlay_source=divergent_overlay_source,
        overlay_mapping=request.overlay_mapping,
        condition_digests=request.condition_digests,
        expected_dataset=request.expected_dataset,
        run_id=request.run_id,
        workspace_root=request.workspace_root,
    )
    # matches execute_import's existing base-path contract (engine.py's
    # verify_import_run(expected=...) check): a divergent overwrite is a
    # loud failure that propagates out of the transaction, not a returned
    # "failed" report -- publish() has already fully released its lock/
    # staging state by the time this raises.
    with pytest.raises(ImportEngineError, match="different experiment"):
        execute_overlay_import(divergent_request, dry_run=False)


def test_execute_overlay_import_refuses_a_forged_authorization_source(tmp_path):
    base_run_dir, base = _make_base_run(tmp_path)
    request = _make_overlay_request(tmp_path, base_run_dir, base)
    request.authorization_source.path.write_bytes(b"tampered,bytes\n9,9\n")

    # the forged bytes no longer match expected_sha256; EvidenceError (a
    # ValueError) is caught by execute_overlay_import's own error handling
    # and surfaces as a failed report rather than a silent success.
    report = execute_overlay_import(request, dry_run=False)
    assert report.import_state == "failed"
    assert report.wrote_artifacts is False
    assert not (tmp_path / "runs" / "overlay-run").exists()
