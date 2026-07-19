"""End-to-end tests for the reduced-scope import engine: base (non-overlay)
plan/dry-run/real-import/idempotence/verify. Overlay merge-into-new-run
orchestration (execute_overlay_import) is a separate function covered by
test_engine_overlay.py; this file covers only the base import path.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from choicebench.importing.engine import (
    ImportEngineError,
    ImportRequest,
    build_import_plan,
    execute_import,
    serialize_import_report,
    verify_import_run,
)
from choicebench.importing.schema import load_import_spec

_RESULTS_CSV = "qid,question,choice_a,choice_b,answer,gold,score\nq1,One?,x,y,a,a,0.9\nq2,Two?,m,n,b,b,0.7\n"


def _raw_spec(tmp_path: Path, results_csv: str = _RESULTS_CSV) -> dict:
    results_path = tmp_path / "results.csv"
    results_path.write_bytes(results_csv.encode("utf-8"))
    expected_sha256 = sha256(results_csv.encode("utf-8")).hexdigest()
    return {
        "schema_version": "choicebench.import-spec.v1",
        "import_name": "engine test import",
        "sources": [
            {
                "source_id": "results",
                "path": str(results_path),
                "logical_path": "freeze/results.csv",
                "expected_sha256": expected_sha256,
                "format": "csv",
                "format_version": "producer-v1",
                "classification": "raw",
                "dialect": {
                    "encoding": "utf-8", "bom_policy": "forbid", "decoding_errors": "strict",
                    "delimiter": ",", "quote_character": '"', "escape_character": None,
                    "double_quote": True, "line_terminators": ["crlf", "lf", "cr"],
                    "mixed_line_terminators": "allow", "final_record_without_terminator": "allow",
                    "blank_record_policy": "reject", "skip_initial_space": False,
                    "header": "first_logical_record", "strict_syntax": True,
                },
                "columns": {
                    "question_id": "qid", "question_text": "question",
                    "correct_option": "gold", "prediction": "answer",
                },
                "expected_columns": ["qid", "question", "choice_a", "choice_b", "answer", "gold", "score"],
                "ignored_columns": {"score": "producer aggregate only"},
                "null_values": ["", "NA"],
                "numeric_columns": [
                    {"source_column": "score", "value_type": "float", "null_allowed": True, "finite_only": True}
                ],
                "option_mapping": {
                    "mode": "ordered_columns", "ordered_columns": ["choice_a", "choice_b"],
                    "structured_column": None, "structured_label_key": None, "structured_text_key": None,
                },
                "extra_field_policy": "preserve_unmapped",
                "preserve_namespace": "producer",
                "source_run_id": None, "source_repository": None, "source_commit": None,
                "notes": {},
            }
        ],
        "datasets": [
            {
                "dataset_id": "dataset",
                "benchmark_name": "historical-benchmark",
                "split": "test",
                "reference_kind": "independent_input_snapshot",
                "trust_label": "producer-supplied",
                "source_ids": ["results"],
                "selection_source_id": "results",
                "expected_question_ids": ["q1", "q2"],
                "selection_seed": None,
                "selection_n_samples": None,
                "subject_filter": [],
                "selection_unknown_reasons": {
                    "selection_seed": "not recorded by producer",
                    "selection_n_samples": "not recorded by producer",
                },
                "columns": {
                    "question_id": "qid", "question_text": "question", "correct_option": "gold",
                    "choice_a": "choice_a", "choice_b": "choice_b",
                },
                "revision": None, "fingerprint": None, "derivation": {},
                "limitations": ["publisher revision was not recorded"],
                "native_compatibility_identity": None,
            }
        ],
        "models": [
            {
                "model_key": "model", "display_name": "historical-model", "backend": None,
                "provider": None, "revision": None, "effective_parameters": {},
                "unknown_reasons": {
                    "backend": "not recorded by producer", "provider": "not recorded by producer",
                    "revision": "not recorded by producer",
                },
                "native_compatibility_identity": None,
            }
        ],
        "methods": [
            {
                "method_key": "method", "name": "historical-direct", "effective_parameters": {},
                "implementation": None,
                "unknown_reasons": {"implementation": "not recorded by producer"},
                "native_compatibility_identity": None,
            }
        ],
        "prompts": [
            {
                "prompt_key": "prompt", "template_identity": None, "template_digest": None,
                "template_contents": None, "unknown_reason": "not recorded by producer",
                "native_compatibility_identity": None,
            }
        ],
        "conditions": [
            {
                "condition_key": "condition", "source_ids": ["results"], "dataset_id": "dataset",
                "model_key": "model", "method_key": "method", "prompt_key": "prompt", "seed": None,
                "calibration_identity": None, "preflight_identity": None, "protocol_settings": {},
                "generation_parameters": {},
                "unknown_reasons": {
                    "seed": "not recorded by producer",
                    "calibration_identity": "not recorded by producer",
                    "preflight_identity": "not recorded by producer",
                },
                "expected_question_ids": ["q1", "q2"],
                "evidence_status": "complete", "scope_disposition": "included", "executable": None,
                "qualifications": [], "limitations": [], "damaged_question_ids": [],
                "recoverable_question_ids": [],
                "result_origin": {
                    "derivation_origin": "external_import",
                    "default_prediction_origin": "external_historical_inference",
                    "per_question_prediction_origins": {},
                },
            }
        ],
        "authorizations": [],
        "overlays": [],
        "metrics": ["accuracy"],
        "provenance": {"producer_request_id": {"value": None, "reason": "not recorded by producer"}},
        "audit": {"source_path": str(tmp_path), "imported_at": "2026-07-18T00:00:00Z"},
    }


def _spec(tmp_path: Path, **kwargs):
    raw = _raw_spec(tmp_path, **kwargs)
    spec_path = tmp_path / "import.yaml"
    spec_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_import_spec(spec_path)


def test_build_import_plan_produces_a_valid_v3_manifest(tmp_path):
    spec = _spec(tmp_path)
    request = ImportRequest(spec=spec, run_id="run-1", workspace_root=tmp_path, strict=True)
    plan = build_import_plan(request)
    assert plan.manifest["schema_version"] == "choicebench.manifest.v3"
    assert len(plan.manifest["payload"]["semantic_conditions"]) == 1
    assert len(plan.manifest["payload"]["realizations"]) == 1
    (result,) = plan.realizations.values()
    assert result.computed_evidence_status == "complete"
    assert result.evaluable is True
    assert len(result.evaluable_rows) == 2


def test_build_import_plan_uses_a_supplied_expected_dataset_override(tmp_path):
    spec = _spec(tmp_path)
    (dataset,) = spec.datasets
    (source,) = spec.sources
    from choicebench.importing.dataset_reference import build_expected_dataset
    from choicebench.importing.evidence import open_verified_source

    opened = open_verified_source(source, containment_root=tmp_path)
    override = build_expected_dataset(dataset, {"results": opened})
    request = ImportRequest(
        spec=spec, run_id="run-1", workspace_root=tmp_path, strict=True,
        expected_datasets={"dataset": override},
    )
    plan = build_import_plan(request)
    assert plan.expected_datasets["dataset"] is override


def test_build_import_plan_rejects_an_incomplete_expected_dataset_override(tmp_path):
    spec = _spec(tmp_path)
    request = ImportRequest(
        spec=spec, run_id="run-1", workspace_root=tmp_path, strict=True,
        expected_datasets={},
    )
    with pytest.raises(ImportEngineError, match="missing dataset"):
        build_import_plan(request)


def test_build_import_plan_rejects_a_source_outside_the_workspace_by_default(tmp_path):
    external_root = tmp_path / "external"
    external_root.mkdir()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    spec = _spec(external_root)
    request = ImportRequest(spec=spec, run_id="run-1", workspace_root=workspace_root, strict=True)
    with pytest.raises(Exception, match="escapes containment root"):
        build_import_plan(request)


def test_build_import_plan_accepts_a_source_outside_the_workspace_with_an_explicit_containment_root(
    tmp_path,
):
    external_root = tmp_path / "external"
    external_root.mkdir()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    spec = _spec(external_root)
    request = ImportRequest(
        spec=spec, run_id="run-1", workspace_root=workspace_root, strict=True,
        source_containment_root=external_root,
    )
    plan = build_import_plan(request)
    assert len(plan.manifest["payload"]["realizations"]) == 1


def test_dry_run_writes_nothing(tmp_path):
    spec = _spec(tmp_path)
    request = ImportRequest(spec=spec, run_id="run-1", workspace_root=tmp_path, strict=True)
    report = execute_import(request, dry_run=True)
    assert report.import_state == "validated"
    assert report.wrote_artifacts is False
    assert not (tmp_path / "runs").exists()
    assert report.counts.evidence_status == {"complete": 1}
    assert report.counts.scope_disposition == {"included": 1}


def test_real_import_writes_manifest_state_and_result(tmp_path):
    spec = _spec(tmp_path)
    request = ImportRequest(spec=spec, run_id="run-1", workspace_root=tmp_path, strict=True)
    report = execute_import(request, dry_run=False)
    assert report.import_state == "imported"
    assert report.wrote_artifacts is True
    assert report.idempotent_noop is False
    run_dir = tmp_path / "runs" / "run-1"
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "run_state.json").is_file()
    (realization_id,) = report.realization_digests.keys()
    assert (run_dir / f"results/{realization_id}.csv").is_file()
    assert (run_dir / f"artifacts/imports/validation/{realization_id}.json").is_file()
    assert (run_dir / "artifacts/imports/evidence/index.json").is_file()


def test_real_import_is_idempotent_noop_on_repeat(tmp_path):
    spec = _spec(tmp_path)
    request = ImportRequest(spec=spec, run_id="run-1", workspace_root=tmp_path, strict=True)
    execute_import(request, dry_run=False)
    second = execute_import(request, dry_run=False)
    assert second.idempotent_noop is True
    assert second.wrote_artifacts is False


def test_verify_import_run_validates_the_published_graph(tmp_path):
    spec = _spec(tmp_path)
    request = ImportRequest(spec=spec, run_id="run-1", workspace_root=tmp_path, strict=True)
    execute_import(request, dry_run=False)
    verified = verify_import_run(tmp_path / "runs" / "run-1")
    (realization,) = verified.realizations.values()
    assert realization.result_sha256 is not None
    assert set(realization.rows_by_question_id) == {"q1", "q2"}
    assert realization.prediction_origins == {
        "q1": "external_historical_inference", "q2": "external_historical_inference",
    }


def test_execute_import_reports_failure_without_writing_on_invalid_declaration(tmp_path):
    spec = _spec(tmp_path, results_csv="qid,question,choice_a,choice_b,answer,gold,score\nq1,One?,x,y,a,a,0.9\n")
    request = ImportRequest(spec=spec, run_id="run-1", workspace_root=tmp_path, strict=True)
    report = execute_import(request, dry_run=False)
    assert report.import_state == "failed"
    assert report.wrote_artifacts is False
    assert not (tmp_path / "runs").exists()
    assert report.failures


def test_serialize_import_report_round_trips_to_plain_dict(tmp_path):
    spec = _spec(tmp_path)
    request = ImportRequest(spec=spec, run_id="run-1", workspace_root=tmp_path, strict=True)
    report = execute_import(request, dry_run=True)
    serialized = serialize_import_report(report)
    assert serialized["import_state"] == "validated"
    assert serialized["counts"]["evidence_status"] == {"complete": 1}
    assert serialized["failures"] == []
