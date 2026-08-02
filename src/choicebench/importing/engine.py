"""Orchestrate generic dry-run and real imports of external results.

Reduced scope: this implements both the base (non-overlay) import path --
plan, dry-run validate, real staged publish, and idempotent re-verification
of an existing run -- and the overlay merge-into-new-run path: an authorized
repair/offline-transformation overlay (Task 10's derive_overlay) merged with
an already-published base run's retained rows into a new, separate,
immutable run (execute_overlay_import). The base run itself is never
mutated; only its verified values are read.

Report schema is also reduced from the original plan: it carries the counts
and digests needed to confirm exact status/queue reproduction and reject
held/excluded work, not the full historical checksum-report/column-
disposition/source-classification breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
from choicebench.importing.authorization import (
    authorization_for_condition,
    validate_authorization_bundle,
)
from choicebench.importing.overlays import VerifiedBaseRealization, derive_overlay
from choicebench.importing.schema import (
    AuthorizationSpec,
    ImportConditionSpec,
    ImportSpec,
    OverlaySpec,
    SourceArtifactSpec,
    validate_csv_dialect_identity,
    validate_numeric_columns_identity,
    validate_option_mapping_identity,
)
from choicebench.importing.transaction import ImportTransaction
from choicebench.importing.validation import (
    ImportValidationError,
    RealizationValidation,
    _expected_choice_letters,
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
    expected_datasets: Mapping[str, ExpectedDataset] | None = None
    """Pre-built, pre-verified ExpectedDataset objects keyed by dataset_id,
    used in place of rebuilding each spec.datasets declaration from its
    source_ids via the generic build_expected_dataset. Some profiles (e.g.
    Stage 1's ARC/MMLU trust chain) need dataset-specific revalidation and
    deduplication the generic path cannot reproduce; spec.datasets is still
    populated for documentation/identity purposes, but is not read here when
    this override is supplied."""
    source_containment_root: Path | None = None
    """Root every declared source path must resolve under, checked by
    open_verified_source. Defaults to workspace_root (the common case: a
    self-contained fixture where sources and the run output share a root).
    Set this separately when sources live under an independent, read-only
    root distinct from where runs are written (e.g. importing a real,
    immutable data freeze into an isolated temporary CHOICEBENCH_HOME)."""


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

    # Build the realization's row-level lineage/result-origin from the full
    # published canonical row set when the cell is structurally complete (this
    # is populated for evaluable AND for malformed-but-structurally-complete
    # in-scope cells); for an evaluable cell published_rows == evaluable_rows,
    # so this is a no-op there. A genuinely incomplete/out-of-scope cell has
    # neither set and produces an empty (evidence-only) result origin.
    rows_for_result = result.published_rows if result.published_rows else result.evaluable_rows
    lineage_components = []
    for row in rows_for_result:
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
    containment_root = request.source_containment_root or request.workspace_root
    if request.expected_datasets is not None:
        expected_datasets: dict[str, ExpectedDataset] = dict(request.expected_datasets)
        missing_datasets = sorted(
            {condition.dataset_id for condition in spec.conditions} - set(expected_datasets)
        )
        if missing_datasets:
            raise ImportEngineError(
                f"request.expected_datasets is missing dataset(s) {missing_datasets} "
                "referenced by spec.conditions."
            )
    else:
        expected_datasets = {}
        for declaration in spec.datasets:
            opened = {
                source_id: open_verified_source(
                    sources_by_id[source_id], containment_root=containment_root
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
            workspace_root=containment_root, strict=request.strict,
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
        opened = open_verified_source(
            declaration,
            containment_root=request.source_containment_root or request.workspace_root,
        )
        index_records.append(write_evidence_blob(staged_run, opened, references=record["references"]))
    write_evidence_index(staged_run, index_records)

    atomic_write_json(staged_run / MANIFEST_FILENAME, plan.manifest)

    state = initial_run_state_v3(plan.manifest)
    for realization_id, realization in plan.manifest["payload"]["realizations"].items():
        result = plan.realizations[realization_id]
        realization_state = state["realizations"][realization_id]
        # A structurally-complete, in-scope cell publishes its full canonical
        # result CSV even when not evaluable ("published_unscored"): the rows
        # are retained verbatim (some predictions may be unparseable) so an
        # authorized overlay can later replace specific rows. A genuinely
        # incomplete/out-of-scope cell stays evidence-only with no result CSV.
        publishes_result = bool(result.published_rows)
        if result.evaluable:
            realization_state["status"] = "completed"
        elif publishes_result:
            realization_state["status"] = "published_unscored"
        else:
            realization_state["status"] = "evidence_only"

        prepared_validation = prepare_realization_validation_artifact(
            result, realization=realization, evidence_records=plan.evidence_records
        )
        write_realization_validation_artifact(staged_run, prepared_validation)
        realization_state["validation_sha256"] = prepared_validation.file_sha256

        if publishes_result:
            rows_for_result = result.published_rows if result.published_rows else result.evaluable_rows
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
                for row in rows_for_result
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


def verify_import_run(
    run_dir: Path, expected: ImportPlan | SimpleNamespace | None = None
) -> VerifiedImportRun:
    """The single shared full-graph verifier: validates manifest, state,
    validation artifacts, and results, and returns immutable trusted values
    only after the whole graph passes. `expected` need only expose a
    `.manifest` mapping (an ImportPlan, or a bare SimpleNamespace(manifest=...)
    for the overlay path, which has no full ImportPlan of its own)."""
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
        # Both a scored ("completed") and an unscored-but-structurally-complete
        # ("published_unscored") realization publish a full result CSV; read
        # the retained row set from either so an overlay can retain rows.
        if status in ("completed", "published_unscored"):
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


# --- Overlay merge-into-new-run orchestration -------------------------------
#
# Scope boundary: requires an *overlayable* base realization -- one that
# published a full canonical result CSV whose row-identity is exactly the
# expected complete question set (every ID present exactly once), with its
# source hash verified by verify_import_run. Overlayability is a data-identity
# property, distinct from evaluability (scoreability): a
# malformed-but-structurally-complete base (all IDs present, but some rows
# carry an unparseable prediction) is NOT evaluable yet IS overlayable -- it
# has a full published row set ("published_unscored" status) from which the
# non-authorized rows are retained verbatim while the authorized rows are
# replaced. Only a genuinely broken base -- one with missing/duplicate/
# unexpected rows or no published result at all (evidence-only) -- is refused,
# with the same fail-closed behavior as before. The derived realization's
# evidence_status is recomputed from the actual merged row content (it is NOT
# forced to the overlay's declared status); the base's own freeze-declared
# historical status remains represented separately (see below).

def _recompute_overlay_evidence_status(
    rows_by_qid: Mapping[str, Mapping[str, Any]], *, expected: ExpectedDataset, base_qualified: bool
) -> str:
    """Recompute the derived realization's evidence_status from the ACTUAL
    merged row content, never forcing it to the overlay's declared status. Any
    row whose predicted_option is empty or is not one of that question's option
    letters keeps the derived cell 'malformed' (e.g. a repair that fixed its 3
    authorized rows but left other unrelated parse-missing rows in place); a
    fully-parseable merged set is 'complete', or 'qualified' when the base
    itself was qualified. The merged set is always structurally complete here
    (the overlayable gate guaranteed the base's full row-identity), so
    'partial'/'failed' cannot arise."""
    frame = expected.frame
    for qid, row in rows_by_qid.items():
        matches = frame[frame["question_id"].astype(str) == str(qid)]
        letters = _expected_choice_letters(matches.iloc[0].to_dict()) if not matches.empty else []
        pred = row.get("predicted_option")
        pred = None if pred is None else str(pred).strip().upper()
        if not pred or pred == "NAN" or pred not in letters:
            return "malformed"
    return "qualified" if base_qualified else "complete"


@dataclass(frozen=True)
class OverlayImportRequest:
    base_run_dir: Path
    base_realization_id: str
    authorization: AuthorizationSpec
    authorization_source: SourceArtifactSpec
    overlay: OverlaySpec
    overlay_source: SourceArtifactSpec
    overlay_mapping: Mapping[str, str]
    condition_digests: Mapping[str, str]
    expected_dataset: ExpectedDataset
    run_id: str
    workspace_root: Path
    source_containment_root: Path | None = None
    """Root every declared source path (authorization_source, overlay_source)
    must resolve under. Defaults to workspace_root; set separately when
    those sources live under an independent, read-only root distinct from
    where the derived run is written -- see ImportRequest.source_containment_root."""


def _merge_overlay_realization(
    request: OverlayImportRequest,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Returns (realization_record, semantic_condition_record, result_rows,
    overlay_evidence_record). The evidence record covers only the new
    overlay source -- base evidence remains immutably stored in the base
    run's own directory and is reachable via parent_digests."""
    verified = verify_import_run(request.base_run_dir)
    base = verified.realizations.get(request.base_realization_id)
    if base is None:
        raise ImportEngineError(f"Unknown base realization {request.base_realization_id!r}.")
    # Overlayable gate: the base must have a published canonical row set whose
    # question-ID identity is exactly the expected complete set. This is
    # independent of the base's evidence_status -- a malformed base that is
    # structurally complete IS overlayable; a base with no published result, or
    # with missing/duplicate/unexpected rows, is refused fail-closed.
    expected_ids = tuple(request.expected_dataset.selected_question_ids)
    base_row_ids = list(base.rows_by_question_id)
    if (
        base.result_sha256 is None
        or set(base_row_ids) != set(expected_ids)
        or len(base_row_ids) != len(expected_ids)
    ):
        raise ImportEngineError(
            f"Realization {request.base_realization_id!r} is not overlayable: it lacks a "
            "structurally complete published canonical row set (every expected question ID "
            "present exactly once). A malformed-but-structurally-complete base is "
            "overlayable, but an evidence-only base or one with missing/duplicate/unexpected "
            "rows cannot be overlaid."
        )

    base_manifest = verified.manifest
    base_realization_record = base_manifest["payload"]["realizations"][request.base_realization_id]
    base_realization_identity = base_realization_record["identity"]["realization"]
    condition_id = base_realization_record["condition_id"]
    semantic_condition = base_manifest["payload"]["semantic_conditions"][condition_id]

    containment_root = request.source_containment_root or request.workspace_root
    opened_auth_source = open_verified_source(
        request.authorization_source, containment_root=containment_root
    )
    bundle = validate_authorization_bundle(
        request.authorization, opened_source=opened_auth_source,
        condition_digests=request.condition_digests,
        expected={base.condition_digest: request.expected_dataset},
    )
    authorization = authorization_for_condition(bundle, condition_digest=base.condition_digest)

    opened_overlay_source = open_verified_source(
        request.overlay_source, containment_root=containment_root
    )
    overlay_table = parse_csv_source(opened_overlay_source, request.overlay_source, strict=True)

    derived = derive_overlay(
        base=base, overlay=request.overlay, authorization=authorization,
        overlay_table=overlay_table, overlay_mapping=request.overlay_mapping,
        expected=request.expected_dataset,
    )

    base_lineage_by_id = {
        component["lineage_id"]: component
        for component in base_realization_identity["lineage_components"]
    }
    base_row_assignments = base_realization_identity["result_origin"]["row_assignments"]
    retained_assignments = [
        assignment for assignment in base_row_assignments
        if assignment["question_id"] not in derived.replacement_question_ids
    ]
    retained_lineage_ids = {assignment["prediction_lineage_id"] for assignment in retained_assignments}

    # Realization identity requires row_assignments to be an ordered subset of
    # the expected dataset's question order (identity.py), not "retained then
    # replaced" -- reassemble in dataset order regardless of which side (base
    # or overlay) supplies each question. lineage_components must follow the
    # SAME deterministic order: canonicalize() hashes list order, so building
    # it from set iteration (as an earlier version of this function did) made
    # the realization/experiment digest nondeterministic across processes
    # whenever more than one row was retained.
    assignments_by_qid = {
        assignment["question_id"]: (
            assignment["question_id"], assignment["prediction_origin"], assignment["prediction_lineage_id"]
        )
        for assignment in retained_assignments
    }
    assignments_by_qid.update({
        component["identity"]["question_id"]: (
            component["identity"]["question_id"],
            component["identity"]["prediction_origin"],
            component["lineage_id"],
        )
        for component in derived.lineage_components
    })
    row_assignments = [
        assignments_by_qid[qid]
        for qid in request.expected_dataset.selected_question_ids
        if qid in assignments_by_qid
    ]

    lineage_components_by_id = {**base_lineage_by_id}
    lineage_components_by_id.update(
        {component["lineage_id"]: component for component in derived.lineage_components}
    )
    all_lineage_components = [
        lineage_components_by_id[lineage_id]
        for _question_id, _origin, lineage_id in row_assignments
    ]
    derivation_origin = derived.lineage_components[0]["identity"]["operation_type"]
    result_origin = make_result_origin(derivation_origin=derivation_origin, row_assignments=row_assignments)

    # Assemble the merged result rows now -- non-authorized base rows retained
    # verbatim (numeric-tolerant, value-identical), authorized rows replaced --
    # so the derived realization's evidence_status is recomputed from the
    # ACTUAL merged content below instead of being forced to the overlay's
    # declared status.
    _ROW_KEYS = {
        "question_id", "question_text", "correct_option", "choices_json",
        "prediction_origin", "predicted_option",
    }
    retained_rows_by_qid = {
        assignment["question_id"]: {
            key: value
            for key, value in base.rows_by_question_id[assignment["question_id"]].items()
            if key in _ROW_KEYS
        }
        for assignment in retained_assignments
    }
    derived_rows_by_qid = {row["question_id"]: dict(row) for row in derived.preownership_rows}
    rows_by_qid = {**retained_rows_by_qid, **derived_rows_by_qid}
    result_rows = [
        rows_by_qid[qid] for qid in request.expected_dataset.selected_question_ids if qid in rows_by_qid
    ]
    base_qualified = base_realization_identity["evidence"]["evidence_status"] == "qualified"
    recomputed_evidence_status = _recompute_overlay_evidence_status(
        rows_by_qid, expected=request.expected_dataset, base_qualified=base_qualified
    )

    replacement_digest = integrity_digest(
        canonicalize({
            "replacement_question_ids": list(derived.replacement_question_ids),
            "replacement_reasons": dict(request.overlay.replacement_reasons),
        })
    )
    overlay_dict: dict[str, Any] = {
        "source_sha256": overlay_table.source_sha256,
        "replacement_digest": replacement_digest,
        "transformation_input_digest": None,
        "preownership_output_digest": None,
        "implementation_digest": None,
    }
    if derivation_origin == "offline_transformation":
        # The lineage/overlay digests MUST be computed from the derived
        # components in the SAME canonical order the identity verifier uses
        # (identity.py filters the realization's own dataset-ordered
        # lineage_components by operation_type), NOT in derive_overlay's
        # sorted(replacement_ids) order. integrity_digest hashes list order, so
        # whenever the replaced IDs' dataset order differs from their sorted
        # order the two digests diverge and identity validation raised a
        # false-positive "digest conflicts with its lineage components".
        ordered_derived_components = [
            component
            for component in all_lineage_components
            if component["identity"]["operation_type"] == derivation_origin
        ]
        overlay_dict["transformation_input_digest"] = integrity_digest(
            [component["identity"]["input_digest"] for component in ordered_derived_components]
        )
        overlay_dict["preownership_output_digest"] = integrity_digest(
            [component["identity"]["preownership_output_digest"] for component in ordered_derived_components]
        )
        overlay_dict["implementation_digest"] = integrity_digest(
            ordered_derived_components[0]["identity"]["implementation"]
        )

    overlay_source_notes_digest = (
        integrity_digest(request.overlay_source.notes) if request.overlay_source.notes else None
    )
    source_records = [
        *base_realization_identity["sources"],
        _source_record(request.overlay_source, opened_overlay_source, notes_digest=overlay_source_notes_digest),
    ]

    realization_identity = {
        "import_spec_digest": base_realization_identity["import_spec_digest"],
        "sources": source_records,
        "expected_dataset": base_realization_identity["expected_dataset"],
        "importer_implementation": base_realization_identity["importer_implementation"],
        "parsing_policy": base_realization_identity["parsing_policy"],
        "validation": base_realization_identity["validation"],
        "evidence": {
            **base_realization_identity["evidence"],
            "evidence_status": recomputed_evidence_status,
        },
        "lineage_components": all_lineage_components,
        "result_origin": result_origin,
        "parent_digests": {
            "realization_digests": [base.realization_digest],
            "evidence_digests": [base.validation_artifact_sha256],
            "result_digests": [base.result_sha256] if base.result_sha256 else [],
        },
        "authorization_digest": authorization.bundle_digest,
        "overlay": overlay_dict,
    }
    realization = make_realization(
        condition_id=condition_id,
        condition_digest=base.condition_digest,
        identity=realization_identity,
        fields={},
        adapter=_external_import_adapter,
        validator=_external_import_validator,
        expected_dataset=request.expected_dataset,
        semantic_condition=semantic_condition,
        lineage_runtime_callables={lid: _external_import_adapter for lid in retained_lineage_ids},
    )

    overlay_evidence_record = evidence_record(
        opened_overlay_source, references=sorted(derived.replacement_question_ids)
    )
    return realization, semantic_condition, result_rows, overlay_evidence_record


def execute_overlay_import(request: OverlayImportRequest, *, dry_run: bool = False) -> ImportReport:
    try:
        realization, semantic_condition, result_rows, overlay_evidence_record = (
            _merge_overlay_realization(request)
        )
    except (ImportIdentityError, ImportEngineError, ValueError) as exc:
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
            failures=({"code": "OVERLAY_IMPORT_FAILED", "sanitized_message": str(exc)},),
        )

    condition_id = realization["condition_id"]
    payload = {
        "protocol_version": PROTOCOL_V3_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "semantic_conditions": {condition_id: semantic_condition},
        "realizations": {realization["realization_id"]: realization},
    }
    manifest = make_manifest_v3(payload, audit={"source_location": str(request.workspace_root)})
    validate_manifest_v3(manifest)

    evidence_status = realization["identity"]["realization"]["evidence"]["evidence_status"]
    scope_disposition = realization["identity"]["realization"]["evidence"]["scope_disposition"]
    counts = ImportCounts(
        conditions={"declared": 1},
        evidence_status={evidence_status: 1},
        scope_disposition={scope_disposition: 1},
    )
    condition_digests = {condition_id: semantic_condition["condition_digest"]}
    realization_digests = {realization["realization_id"]: realization["realization_digest"]}

    if dry_run:
        return ImportReport(
            schema_version="choicebench.import-report.v1",
            import_state="validated",
            wrote_artifacts=False,
            idempotent_noop=False,
            run_id=request.run_id,
            experiment_id=manifest["experiment_id"],
            counts=counts,
            defects=DefectReport(total_findings=0, by_code={}, findings_digest=integrity_digest([])),
            condition_digests=condition_digests,
            realization_digests=realization_digests,
            failures=(),
        )

    runs_dir = Path(request.workspace_root) / "runs"
    idempotent_noop = [False]

    def _validator(published_root: Path) -> None:
        if published_root == runs_dir / request.run_id and (published_root / MANIFEST_FILENAME).is_file():
            verify_import_run(published_root, expected=SimpleNamespace(manifest=manifest))
            idempotent_noop[0] = True
            return
        _write_overlay_staged_run(
            published_root, manifest, realization, result_rows,
            overlay_source=request.overlay_source,
            overlay_evidence_record=overlay_evidence_record,
            workspace_root=request.source_containment_root or request.workspace_root,
            declared_evidence_status=request.overlay.expected_evidence_status,
        )

    with ImportTransaction(runs_dir=runs_dir, run_id=request.run_id) as txn:
        txn.publish(_validator)

    return ImportReport(
        schema_version="choicebench.import-report.v1",
        import_state="imported",
        wrote_artifacts=not idempotent_noop[0],
        idempotent_noop=idempotent_noop[0],
        run_id=request.run_id,
        experiment_id=manifest["experiment_id"],
        counts=counts,
        defects=DefectReport(total_findings=0, by_code={}, findings_digest=integrity_digest([])),
        condition_digests=condition_digests,
        realization_digests=realization_digests,
        failures=(),
    )


def _write_overlay_staged_run(
    staged_run: Path, manifest: Mapping[str, Any], realization: Mapping[str, Any],
    result_rows: list[dict[str, Any]],
    *,
    overlay_source: SourceArtifactSpec,
    overlay_evidence_record: Mapping[str, Any],
    workspace_root: Path,
    declared_evidence_status: str,
) -> None:
    staged_run.mkdir(parents=True, exist_ok=True)

    opened_overlay_source = open_verified_source(overlay_source, containment_root=workspace_root)
    written_record = write_evidence_blob(
        staged_run, opened_overlay_source, references=overlay_evidence_record["references"]
    )
    write_evidence_index(staged_run, [written_record])

    atomic_write_json(staged_run / MANIFEST_FILENAME, manifest)

    realization_id = realization["realization_id"]
    state = initial_run_state_v3(manifest)
    evidence = realization["identity"]["realization"]["evidence"]
    recomputed_status = evidence["evidence_status"]
    evaluable = recomputed_status in ("complete", "qualified") and evidence["scope_disposition"] == "included"
    # A derived overlay run always publishes a full canonical result CSV; its
    # run status reflects the RECOMPUTED evidence_status ("completed" when the
    # merged content is evaluable, else "published_unscored"). The overlay's
    # declared/intended historical status is preserved separately here so it is
    # never conflated with the observed evidence_status.
    state["realizations"][realization_id]["status"] = "completed" if evaluable else "published_unscored"
    state["realizations"][realization_id]["declared_evidence_status"] = declared_evidence_status

    condition_id = realization["condition_id"]
    condition_record = manifest["payload"]["semantic_conditions"][condition_id]
    benchmark = condition_record["identity"]["benchmark"]
    identity_columns = {
        "condition_id": condition_id,
        "realization_id": realization_id,
        "experiment_id": manifest["experiment_id"],
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
    final_rows = [
        {**identity_columns, **row, "prediction_lineage_id": lineage_ids_by_qid[row["question_id"]]}
        for row in result_rows
    ]
    prepared_result = prepare_manifest_result(final_rows, manifest=manifest, realization_id=realization_id)
    _, published_metadata, _ = publish_manifest_result(prepared_result, run_dir=staged_run)
    state["realizations"][realization_id]["result_artifact_id"] = published_metadata["result_artifact_id"]
    state["realizations"][realization_id]["result_artifact_digest"] = published_metadata["result_artifact_digest"]
    state["realizations"][realization_id]["result_sha256"] = published_metadata["file_sha256"]

    merged_validation = RealizationValidation(
        evaluable_rows=tuple(result_rows) if evaluable else (),
        findings=(),
        validation_digest=integrity_digest({"evaluable_rows": canonicalize(result_rows)}),
        computed_evidence_status=recomputed_status,
        evaluable=evaluable,
        qualifications=(),
        limitations=(),
        defect_question_ids=(),
        published_rows=tuple(result_rows),
        structurally_complete=True,
    )
    prepared_validation = prepare_realization_validation_artifact(
        merged_validation, realization=realization, evidence_records=()
    )
    write_realization_validation_artifact(staged_run, prepared_validation)
    state["realizations"][realization_id]["validation_sha256"] = prepared_validation.file_sha256

    atomic_write_json(staged_run / RUN_STATE_FILENAME, canonicalize(state, redact_secrets=False))
    validate_manifest_v3(manifest)


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
