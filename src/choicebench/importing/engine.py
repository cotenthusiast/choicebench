"""Orchestrate generic dry-run and real imports of external results.

Reduced scope: this implements the base (non-overlay) import path fully --
plan, dry-run validate, real staged publish, and idempotent re-verification
of an existing run. Applying an authorized repair/offline-transformation
overlay on top of an already-published base run (Task 10's derive_overlay,
merged with the base's retained rows into a new run) is the next increment;
every piece it needs (identity, validation, authorization, overlay
derivation, evidence storage, atomic transactions, and this module's own
verify_import_run) is already built and tested, but the merge-and-publish
orchestration itself is not yet wired up here.

Report schema is also reduced from the original plan: it carries the counts
and digests needed to confirm exact status/queue reproduction and reject
held/excluded work, not the full historical checksum-report/column-
disposition/source-classification breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from dataclasses import asdict
import json

import pandas as pd

from choicebench.identity import CANONICALIZATION_VERSION, canonicalize, integrity_digest
from choicebench.importing.csv_adapter import parse_csv_source
from choicebench.importing.dataset_reference import ExpectedDataset, build_expected_dataset
from choicebench.importing.evidence import (
    evidence_record,
    open_verified_source,
    validate_evidence_index,
    write_evidence_blob,
    write_evidence_index,
)
from choicebench.importing.identity import (
    ImportIdentityError,
    build_import_semantic_identity,
    importer_implementation_identity,
    make_lineage_component,
    make_realization,
    make_result_origin,
)
from choicebench.importing.overlays import VerifiedBaseRealization
from choicebench.importing.schema import (
    ImportConditionSpec,
    ImportSpec,
    validate_csv_dialect_identity,
    validate_numeric_columns_identity,
    validate_option_mapping_identity,
)
from choicebench.importing.transaction import ImportTransaction
from choicebench.importing.validation import (
    ImportValidationError,
    normalize_realization_rows,
    prepare_realization_validation_artifact,
    validate_realization_validation_artifact,
    validate_source_rows,
    write_realization_validation_artifact,
)
from choicebench.io.writers import (
    prepare_manifest_result,
    publish_manifest_result,
    validate_result_artifact,
)
from choicebench.infra.artifacts import atomic_write_json
from choicebench.manifest import (
    MANIFEST_FILENAME,
    PROTOCOL_V3_VERSION,
    RUN_STATE_FILENAME,
    initial_run_state_v3,
    make_manifest_v3,
    validate_manifest_v3,
    validate_run_state_v3,
)

_ADAPTER_STRICT_KEY = "strict"


class ImportEngineError(ValueError):
    """Raised when an import request or an existing run fails validation."""


def _external_import_adapter(*, strict: bool) -> None:
    """Identity marker for this engine's CSV-adaptation stage.

    Deliberately a trivial, side-effect-free function rather than binding
    directly to parse_csv_source/validate_source_rows: those reference many
    stdlib globals (hashlib.sha256, re, etc.) that are neither plain data nor
    inspectable user-level code, and Task 5's runtime-callable identity fails
    closed on exactly that shape rather than silently ignoring it. Binding
    here instead still captures "did this engine's own orchestration change"
    (implementation_identity hashes this whole file); the installed
    choicebench package version already in importer_implementation_identity's
    "package" field is the primary signal for "did the installed release
    (including csv_adapter.py/validation.py) change."
    """
    return None


def _external_import_validator() -> None:
    """Identity marker for this engine's row-validation stage. See
    _external_import_adapter for why this is a marker, not the real function."""
    return None


@dataclass(frozen=True)
class ImportRequest:
    spec: ImportSpec
    run_id: str
    workspace_root: Path
    strict: bool
    overlays: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ImportPlan:
    manifest: Mapping[str, Any]
    expected_datasets: Mapping[str, ExpectedDataset]
    realizations: Mapping[str, Any]
    evidence_records: tuple[Mapping[str, Any], ...]
    findings: tuple[Any, ...]


@dataclass(frozen=True)
class VerifiedImportRun:
    manifest: Mapping[str, Any]
    manifest_digest: str
    realizations: Mapping[str, VerifiedBaseRealization]


@dataclass(frozen=True)
class ImportCounts:
    conditions: Mapping[str, int]
    evidence_status: Mapping[str, int]
    scope_disposition: Mapping[str, int]


@dataclass(frozen=True)
class DefectReport:
    total_findings: int
    by_code: Mapping[str, int]
    findings_digest: str


@dataclass(frozen=True)
class ImportReport:
    schema_version: Literal["choicebench.import-report.v1"]
    import_state: Literal["validated", "imported", "failed"]
    wrote_artifacts: bool
    idempotent_noop: bool
    run_id: str
    experiment_id: str | None
    counts: ImportCounts
    defects: DefectReport
    condition_digests: Mapping[str, str]
    realization_digests: Mapping[str, str]
    failures: tuple[Mapping[str, Any], ...]


def _source_record(declaration, opened, *, notes_digest: str | None) -> dict[str, Any]:
    unknown_reasons = {
        field: "not declared in generic import specification"
        for field, value in (
            ("source_run_id", declaration.source_run_id),
            ("source_repository", declaration.source_repository),
            ("source_commit", declaration.source_commit),
        )
        if value is None
    }
    return {
        "source_id": declaration.source_id,
        "logical_path": declaration.logical_path,
        "classification": declaration.classification,
        "format": declaration.format,
        "format_version": declaration.format_version,
        "sha256": opened.sha256,
        "provenance": {
            "source_run_id": declaration.source_run_id,
            "source_repository": declaration.source_repository,
            "source_commit": declaration.source_commit,
            "notes_digest": notes_digest,
            "evidence_digest": opened.sha256,
            "unknown_reasons": unknown_reasons,
        },
    }


def _build_realization_for_condition(
    condition: ImportConditionSpec,
    *,
    spec: ImportSpec,
    dataset: ExpectedDataset,
    model,
    method,
    prompt,
    workspace_root: Path,
    strict: bool,
) -> tuple[dict[str, Any], Any, tuple[dict[str, Any], ...]]:
    """Returns (realization_record, RealizationValidation, evidence_records)."""
    sources_by_id = {source.source_id: source for source in spec.sources}
    semantic = build_import_semantic_identity(
        condition=condition, dataset=dataset, model=model, method=method, prompt=prompt
    )

    opened_by_source: dict[str, Any] = {}
    tables_by_source: dict[str, Any] = {}
    for source_id in condition.source_ids:
        declaration = sources_by_id[source_id]
        opened = open_verified_source(declaration, containment_root=workspace_root)
        opened_by_source[source_id] = opened
        tables_by_source[source_id] = parse_csv_source(opened, declaration, strict=strict)

    primary_source_id = condition.source_ids[0]
    primary_declaration = sources_by_id[primary_source_id]
    primary_table = tables_by_source[primary_source_id]
    validated = validate_source_rows(
        primary_table, source_id=primary_source_id, mapping=primary_declaration.columns,
        condition=condition, expected=dataset,
    )
    result = normalize_realization_rows(
        validated, condition=condition, expected=dataset, mapping=primary_declaration.columns,
    )

    source_records = []
    evidence_records: list[dict[str, Any]] = []
    all_source_digests = []
    for source_id in condition.source_ids:
        declaration = sources_by_id[source_id]
        opened = opened_by_source[source_id]
        notes_digest = integrity_digest(declaration.notes) if declaration.notes else None
        source_records.append(_source_record(declaration, opened, notes_digest=notes_digest))
        all_source_digests.append(opened.sha256)
        evidence_records.append(
            evidence_record(opened, references=sorted(validated.rows_by_question_id))
        )
    all_source_digests = tuple(sorted(set(all_source_digests)))

    lineage_components = []
    for row in result.evaluable_rows:
        qid = row["question_id"]
        raw_row = dict(validated.rows_by_question_id[qid].values)
        component = make_lineage_component(
            operation_type="external_import",
            question_id=qid,
            parent_digests=(),
            source_digests=all_source_digests,
            authorization_digest=None,
            implementation=_external_import_adapter,
            parameters={_ADAPTER_STRICT_KEY: strict},
            input_digest=integrity_digest({"question_id": qid, "raw_row": raw_row}),
            preownership_output_digest=integrity_digest({"question_id": qid, "row": row}),
            prediction_origin=row["prediction_origin"],
        )
        lineage_components.append(component)

    row_assignments = [
        (
            component["identity"]["question_id"],
            component["identity"]["prediction_origin"],
            component["lineage_id"],
        )
        for component in lineage_components
    ]
    result_origin = make_result_origin(derivation_origin="external_import", row_assignments=row_assignments)

    qualification_digest = integrity_digest(list(result.qualifications))
    limitation_digest = integrity_digest(list(result.limitations))
    defect_digest = integrity_digest(list(result.defect_question_ids))

    runtime_importer = importer_implementation_identity(
        adapter=_external_import_adapter, validator=_external_import_validator
    )

    realization_identity = {
        "import_spec_digest": integrity_digest(canonicalize(spec.provenance)),
        "sources": source_records,
        "expected_dataset": {
            "snapshot_digest": dataset.snapshot_digest,
            "question_set_digest": dataset.question_set_digest,
            "derivation_digest": dataset.derivation_digest,
            "question_ids": list(dataset.selected_question_ids),
        },
        "importer_implementation": runtime_importer,
        "parsing_policy": {
            "dialect": validate_csv_dialect_identity(asdict(primary_declaration.dialect)),
            "mapping": dict(primary_declaration.columns),
            "null_values": list(primary_declaration.null_values),
            "numeric_columns": validate_numeric_columns_identity(
                [asdict(item) for item in primary_declaration.numeric_columns]
            ),
            "option_mapping": validate_option_mapping_identity(
                asdict(primary_declaration.option_mapping)
            ),
            "extra_field_policy": primary_declaration.extra_field_policy,
        },
        "validation": {"findings_digest": validated.findings_digest},
        "evidence": {
            "evidence_status": result.computed_evidence_status,
            "qualification_digest": qualification_digest,
            "limitation_digest": limitation_digest,
            "defect_digest": defect_digest,
            "scope_disposition": condition.scope_disposition,
        },
        "lineage_components": lineage_components,
        "result_origin": result_origin,
        "parent_digests": {"realization_digests": [], "evidence_digests": [], "result_digests": []},
        "authorization_digest": None,
        "overlay": None,
    }

    lineage_runtime_callables = {
        component["lineage_id"]: _external_import_adapter for component in lineage_components
    }
    realization = make_realization(
        condition_id=semantic.condition["condition_id"],
        condition_digest=semantic.condition["condition_digest"],
        identity=realization_identity,
        fields={},
        adapter=_external_import_adapter,
        validator=_external_import_validator,
        expected_dataset=dataset,
        semantic_condition=semantic.condition,
        lineage_runtime_callables=lineage_runtime_callables,
    )
    return realization, semantic.condition, result, tuple(evidence_records)


def build_import_plan(request: ImportRequest) -> ImportPlan:
    """Resolve, adapt, validate, and identity-bind every declared condition.
    Performs every read/validation/identity step but writes nothing."""
    spec = request.spec
    sources_by_id = {source.source_id: source for source in spec.sources}
    expected_datasets: dict[str, ExpectedDataset] = {}
    for declaration in spec.datasets:
        opened = {
            source_id: open_verified_source(
                sources_by_id[source_id], containment_root=request.workspace_root
            )
            for source_id in declaration.source_ids
        }
        expected_datasets[declaration.dataset_id] = build_expected_dataset(declaration, opened)

    models_by_key = {model.model_key: model for model in spec.models}
    methods_by_key = {method.method_key: method for method in spec.methods}
    prompts_by_key = {prompt.prompt_key: prompt for prompt in spec.prompts}

    semantic_conditions: dict[str, dict[str, Any]] = {}
    realizations: dict[str, dict[str, Any]] = {}
    realization_validations: dict[str, Any] = {}
    all_evidence_records: list[dict[str, Any]] = []
    all_findings: list[Any] = []

    for condition in spec.conditions:
        dataset = expected_datasets[condition.dataset_id]
        model = models_by_key[condition.model_key]
        method = methods_by_key[condition.method_key]
        prompt = prompts_by_key[condition.prompt_key]
        realization, condition_record, result, evidence_records = _build_realization_for_condition(
            condition, spec=spec, dataset=dataset, model=model, method=method, prompt=prompt,
            workspace_root=request.workspace_root, strict=request.strict,
        )
        semantic_conditions[condition_record["condition_id"]] = condition_record
        realizations[realization["realization_id"]] = realization
        realization_validations[realization["realization_id"]] = result
        all_evidence_records.extend(evidence_records)
        all_findings.extend(result.findings)

    payload = {
        "protocol_version": PROTOCOL_V3_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "semantic_conditions": semantic_conditions,
        "realizations": realizations,
    }
    manifest = make_manifest_v3(
        payload,
        audit={
            "source_location": str(request.workspace_root),
        },
    )
    validate_manifest_v3(manifest)

    return ImportPlan(
        manifest=manifest,
        expected_datasets=expected_datasets,
        realizations=realization_validations,
        evidence_records=tuple(all_evidence_records),
        findings=tuple(all_findings),
    )


def _counts_for_plan(plan: ImportPlan) -> ImportCounts:
    conditions = {"declared": len(plan.manifest["payload"]["semantic_conditions"])}
    evidence_status: dict[str, int] = {}
    scope_disposition: dict[str, int] = {}
    for realization_id, result in plan.realizations.items():
        evidence_status[result.computed_evidence_status] = (
            evidence_status.get(result.computed_evidence_status, 0) + 1
        )
        realization = plan.manifest["payload"]["realizations"][realization_id]
        disposition = realization["identity"]["realization"]["evidence"]["scope_disposition"]
        scope_disposition[disposition] = scope_disposition.get(disposition, 0) + 1
    return ImportCounts(
        conditions=conditions, evidence_status=evidence_status, scope_disposition=scope_disposition,
    )


def _defects_for_plan(plan: ImportPlan) -> DefectReport:
    by_code: dict[str, int] = {}
    for finding in plan.findings:
        by_code[finding.code] = by_code.get(finding.code, 0) + 1
    findings_payload = [finding.to_json() for finding in plan.findings]
    return DefectReport(
        total_findings=len(plan.findings), by_code=by_code,
        findings_digest=integrity_digest(findings_payload),
    )


def execute_import(request: ImportRequest, *, dry_run: bool = False) -> ImportReport:
    try:
        plan = build_import_plan(request)
    except (ImportIdentityError, ImportValidationError, ValueError) as exc:
        return ImportReport(
            schema_version="choicebench.import-report.v1",
            import_state="failed",
            wrote_artifacts=False,
            idempotent_noop=False,
            run_id=request.run_id,
            experiment_id=None,
            counts=ImportCounts(conditions={}, evidence_status={}, scope_disposition={}),
            defects=DefectReport(total_findings=0, by_code={}, findings_digest=integrity_digest([])),
            condition_digests={},
            realization_digests={},
            failures=({"code": "IMPORT_PLAN_FAILED", "sanitized_message": str(exc)},),
        )

    counts = _counts_for_plan(plan)
    defects = _defects_for_plan(plan)
    condition_digests = {
        condition_id: record["condition_digest"]
        for condition_id, record in plan.manifest["payload"]["semantic_conditions"].items()
    }
    realization_digests = {
        realization_id: record["realization_digest"]
        for realization_id, record in plan.manifest["payload"]["realizations"].items()
    }

    if dry_run:
        return ImportReport(
            schema_version="choicebench.import-report.v1",
            import_state="validated",
            wrote_artifacts=False,
            idempotent_noop=False,
            run_id=request.run_id,
            experiment_id=plan.manifest["experiment_id"],
            counts=counts,
            defects=defects,
            condition_digests=condition_digests,
            realization_digests=realization_digests,
            failures=(),
        )

    runs_dir = Path(request.workspace_root) / "runs"
    idempotent_noop = [False]

    def _validator(published_root: Path) -> None:
        if published_root == runs_dir / request.run_id and (published_root / MANIFEST_FILENAME).is_file():
            verify_import_run(published_root, expected=plan)
            idempotent_noop[0] = True
            return
        _write_staged_run(published_root, request, plan)

    with ImportTransaction(runs_dir=runs_dir, run_id=request.run_id) as txn:
        final_path = txn.publish(_validator)

    return ImportReport(
        schema_version="choicebench.import-report.v1",
        import_state="imported",
        wrote_artifacts=not idempotent_noop[0],
        idempotent_noop=idempotent_noop[0],
        run_id=request.run_id,
        experiment_id=plan.manifest["experiment_id"],
        counts=counts,
        defects=defects,
        condition_digests=condition_digests,
        realization_digests=realization_digests,
        failures=(),
    )


def _write_staged_run(staged_run: Path, request: ImportRequest, plan: ImportPlan) -> None:
    staged_run.mkdir(parents=True, exist_ok=True)
    sources_by_id = {source.source_id: source for source in request.spec.sources}

    written_digests: set[str] = set()
    index_records: list[Mapping[str, Any]] = []
    for record in plan.evidence_records:
        if record["sha256"] in written_digests:
            continue
        written_digests.add(record["sha256"])
        declaration = sources_by_id[record["source_id"]]
        opened = open_verified_source(declaration, containment_root=request.workspace_root)
        index_records.append(write_evidence_blob(staged_run, opened, references=record["references"]))
    write_evidence_index(staged_run, index_records)

    atomic_write_json(staged_run / MANIFEST_FILENAME, plan.manifest)

    state = initial_run_state_v3(plan.manifest)
    for realization_id, realization in plan.manifest["payload"]["realizations"].items():
        result = plan.realizations[realization_id]
        realization_state = state["realizations"][realization_id]
        realization_state["status"] = "completed" if result.evaluable else "evidence_only"

        prepared_validation = prepare_realization_validation_artifact(
            result, realization=realization, evidence_records=plan.evidence_records
        )
        write_realization_validation_artifact(staged_run, prepared_validation)
        realization_state["validation_sha256"] = prepared_validation.file_sha256

        if result.evaluable:
            condition_id = realization["condition_id"]
            condition_record = plan.manifest["payload"]["semantic_conditions"][condition_id]
            benchmark = condition_record["identity"]["benchmark"]
            identity_columns = {
                "condition_id": condition_id,
                "realization_id": realization_id,
                "experiment_id": plan.manifest["experiment_id"],
                "dataset_artifact_id": benchmark["artifact_id"],
                "dataset_selection_id": benchmark["selection_id"],
                "model_id": condition_record["identity"]["model_id"],
                "method_id": condition_record["identity"]["method_id"],
                "prompt_id": condition_record["identity"]["prompt_id"],
                "benchmark_name": benchmark["name"],
                "benchmark_split": benchmark["split"],
            }
            lineage_ids_by_qid = {
                assignment["question_id"]: assignment["prediction_lineage_id"]
                for assignment in realization["identity"]["realization"]["result_origin"]["row_assignments"]
            }
            result_rows = [
                {
                    **identity_columns,
                    **row,
                    "prediction_lineage_id": lineage_ids_by_qid[row["question_id"]],
                }
                for row in result.evaluable_rows
            ]
            prepared_result = prepare_manifest_result(
                result_rows, manifest=plan.manifest, realization_id=realization_id
            )
            _, published_metadata, _ = publish_manifest_result(prepared_result, run_dir=staged_run)
            realization_state["result_artifact_id"] = published_metadata["result_artifact_id"]
            realization_state["result_artifact_digest"] = published_metadata["result_artifact_digest"]
            realization_state["result_sha256"] = published_metadata["file_sha256"]

    atomic_write_json(staged_run / RUN_STATE_FILENAME, canonicalize(state, redact_secrets=False))
    validate_manifest_v3(plan.manifest)


def verify_import_run(run_dir: Path, expected: ImportPlan | None = None) -> VerifiedImportRun:
    """The single shared full-graph verifier: validates manifest, state,
    validation artifacts, and results, and returns immutable trusted values
    only after the whole graph passes."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    validate_manifest_v3(manifest)
    state = json.loads((run_dir / RUN_STATE_FILENAME).read_text())
    validate_run_state_v3(state, manifest)

    if expected is not None and manifest["experiment_digest"] != expected.manifest["experiment_digest"]:
        raise ImportEngineError(
            f"Run {run_dir} belongs to a different experiment than expected."
        )

    evidence_index_path = run_dir / "artifacts" / "imports" / "evidence" / "index.json"
    evidence_index = json.loads(evidence_index_path.read_text())
    evidence_index_digest = evidence_index["evidence_index_digest"]
    validate_evidence_index(run_dir, evidence_index_digest)

    realizations: dict[str, VerifiedBaseRealization] = {}
    for realization_id, realization in manifest["payload"]["realizations"].items():
        realization_identity = realization["identity"]["realization"]
        realization_state = state["realizations"][realization_id]
        status = realization_state["status"]

        validation_path = run_dir / f"artifacts/imports/validation/{realization_id}.json"
        validate_realization_validation_artifact(
            validation_path, manifest=manifest, realization=realization,
            expected_sha256=realization_state["validation_sha256"],
        )

        result_sha256 = None
        rows_by_question_id: dict[str, Any] = {}
        prediction_origins: dict[str, str] = {}
        if status == "completed":
            result_path = run_dir / f"results/{realization_id}.csv"
            metadata = validate_result_artifact(result_path, manifest=manifest, realization=realization)
            if (
                metadata["result_artifact_id"] != realization_state["result_artifact_id"]
                or metadata["result_artifact_digest"] != realization_state["result_artifact_digest"]
            ):
                raise ImportEngineError(
                    f"Realization {realization_id!r} result artifact does not match its "
                    "recorded run-state identity."
                )
            result_sha256 = metadata["file_sha256"]
            frame = pd.read_csv(result_path, dtype={"question_id": "string"})
            for _, row in frame.iterrows():
                qid = str(row["question_id"])
                rows_by_question_id[qid] = row.to_dict()
                prediction_origins[qid] = str(row["prediction_origin"])

        evidence_digests = {
            source["source_id"]: source["sha256"] for source in realization_identity["sources"]
        }

        realizations[realization_id] = VerifiedBaseRealization(
            condition_id=realization["condition_id"],
            condition_digest=realization["condition_digest"],
            realization_id=realization_id,
            realization_digest=realization["realization_digest"],
            evidence_index_digest=evidence_index_digest,
            evidence_source_digests=evidence_digests,
            validation_artifact_sha256=realization_state["validation_sha256"],
            result_sha256=result_sha256,
            rows_by_question_id=rows_by_question_id,
            prediction_origins=prediction_origins,
        )

    return VerifiedImportRun(
        manifest=manifest, manifest_digest=manifest["experiment_digest"], realizations=realizations
    )


def serialize_import_report(report: ImportReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "import_state": report.import_state,
        "wrote_artifacts": report.wrote_artifacts,
        "idempotent_noop": report.idempotent_noop,
        "run_id": report.run_id,
        "experiment_id": report.experiment_id,
        "counts": {
            "conditions": dict(report.counts.conditions),
            "evidence_status": dict(report.counts.evidence_status),
            "scope_disposition": dict(report.counts.scope_disposition),
        },
        "defects": {
            "total_findings": report.defects.total_findings,
            "by_code": dict(report.defects.by_code),
            "findings_digest": report.defects.findings_digest,
        },
        "condition_digests": dict(report.condition_digests),
        "realization_digests": dict(report.realization_digests),
        "failures": [dict(failure) for failure in report.failures],
    }
