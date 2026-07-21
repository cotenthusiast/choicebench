"""Analysis-facing orchestration on top of choicebench.importing.

Builds and (against disposable, test-only CHOICEBENCH_HOME directories only)
executes the operations the final-paper analysis needs on top of the
generic importer at commit b9167c3e01f8a111ac92011bf62cc79774d38f3a:

  - importing the 102 historical records (thin re-export of choicebench's own
    stage1_paper_freeze profile -- no new logic, just wiring; see
    ``historical_import_spec``)
  - applying a 3-row inference-repair overlay (28 of these exist)
  - applying a 3-row offline-transformation overlay backed by Component 1's
    semantic-rematch engine (6 of these exist)
  - importing one new fourth-cell (visible_llm_matcher) realization as a
    brand-new condition, not an overlay (12 of these exist)

Per the final-paper-eval session's scope boundary, none of these are ever
executed against the real immutable bundle here -- only synthetic/disposable
fixtures in tests exercise this module. These are the exact functions the
later "final-boss" session calls against the authoritative bundle.

Format-version labels (methodology lock #14, schema_compatibility_matrix.json):
``format_version`` is a plain, non-empty ``str`` field on ``SourceArtifactSpec``
(schema.py), not a closed enum -- these three new values are additive, not a
schema change.

Fourth-cell PredictionOrigin decision (methodology lock #7, handoff decision 1):
the handoff package deliberately left the exact ``PredictionOrigin`` enum
value unresolved, recording only the concept ``genuine_model_inference`` and
explicitly ruling OUT ``external_historical_inference`` (that value names an
already-historical prediction, not a freshly-executed one). Of the two
remaining schema values, ``external_repair_inference`` is used elsewhere in
this codebase (``importing/overlays.py::_REPAIR_ORIGINS``) exclusively for
overlays that repair an existing base cell -- semantically wrong here, since
a fourth-cell realization is a brand-new condition, not a repair of anything.
``native_inference`` is therefore the closer fit: its role across this
importer is "this row's prediction came from a real, freshly-executed
inference call," which is exactly what the handoff verified for all 12
realizations (``stage2_calls_made=1000`` in every job log). This is a
documented judgment call, not asserted by the handoff -- flag it to the user
before it is ever applied against the authoritative bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from choicebench.importing.csv_adapter import parse_csv_source
from choicebench.importing.dataset_reference import ExpectedDataset
from choicebench.importing.engine import (
    ImportEngineError,
    ImportReport,
    ImportRequest,
    OverlayImportRequest,
    execute_import,
    execute_overlay_import,
)
from choicebench.importing.evidence import open_verified_source
from choicebench.importing.profiles.stage1_paper_freeze import (
    build_stage1_cell_authorization,
    build_stage1_expected_datasets,
    build_stage1_import_spec,
    translate_stage1_paper_freeze,
)
from choicebench.importing.schema import (
    AuthorizationSpec,
    CsvDialectSpec,
    ImportConditionSpec,
    ImportMethodSpec,
    ImportModelSpec,
    ImportPromptSpec,
    ImportSpec,
    OptionMappingSpec,
    OverlaySpec,
    ResultOriginSpec,
    SourceArtifactSpec,
)

INFERENCE_REPAIR_PATCH_FORMAT_VERSION = "choicebench.inference_repair_patch.v1"
EXTERNAL_RESULT_ROWS_FORMAT_VERSION = "choicebench.external_result_rows.v1"
OFFLINE_TRANSFORMATION_PATCH_FORMAT_VERSION = "choicebench.offline_transformation_patch.v1"

# Documented decision -- see module docstring. This constant is the single
# place that decision lives, so it can be reviewed/changed in one place.
FOURTH_CELL_PREDICTION_ORIGIN = "native_inference"

_REPAIR_PATCH_ROW_COUNT = 3
_OFFLINE_TRANSFORMATION_ROW_COUNT = 3
_DIAGNOSTICS_PATH_MARKER = "diagnostics_excluded_from_import"
_EXCLUDED_SCOPE_DISPOSITION = "excluded_from_paper_matrix"


class OrchestrationError(ValueError):
    """Raised when an orchestration-level fail-closed check rejects an input."""


# --- Fail-closed guards, shared by all three overlay/import paths ----------


def reject_diagnostic_paths(paths: Sequence[Path | str]) -> None:
    """Reject any path whose parts include the diagnostics directory marker.
    Diagnostic/canary artifacts must never enter any import (methodology
    lock #11)."""
    for path in paths:
        parts = PurePosixPath(str(path)).parts
        if _DIAGNOSTICS_PATH_MARKER in parts:
            raise OrchestrationError(
                f"Path {path!r} is under {_DIAGNOSTICS_PATH_MARKER!r} and must "
                "never enter an import."
            )


def reject_excluded_scope_disposition(
    scope_disposition: str, *, cell_id: str
) -> None:
    """Reject building any authorization/overlay/condition for a
    paper-scope-excluded cell (the two hosted-Qwen PriDe cells, methodology
    lock #10). Excluded cells are preserved read-only in the freeze; they
    must never receive an overlay or be treated as an importable condition
    for aggregation."""
    if scope_disposition == _EXCLUDED_SCOPE_DISPOSITION:
        raise OrchestrationError(
            f"Cell {cell_id!r} is paper-scope-excluded "
            f"({_EXCLUDED_SCOPE_DISPOSITION!r}) and must never receive an "
            "overlay or be imported as an aggregate-eligible condition."
        )


def require_exact_row_count(question_ids: Sequence[str], *, expected: int, where: str) -> None:
    """Fail closed if a patch/overlay artifact does not carry exactly
    ``expected`` rows, or carries duplicate question IDs. This is defense in
    depth: derive_overlay's own authorization-grants check would eventually
    catch an unauthorized ID, but a wrong row count with only authorized IDs
    (e.g. 2 of the 3, or a duplicate) would otherwise pass silently through
    to a less specific error."""
    if len(question_ids) != len(set(question_ids)):
        raise OrchestrationError(f"{where} contains duplicate question IDs.")
    if len(question_ids) != expected:
        raise OrchestrationError(
            f"{where} has {len(question_ids)} row(s); exactly {expected} are required."
        )


def require_exact_question_id_set(
    actual: Sequence[str], expected: Sequence[str], *, where: str
) -> None:
    """Fail closed if the row-level question ID set does not exactly match
    the authorized set (no missing, no unexpected, no extras)."""
    actual_set, expected_set = set(actual), set(expected)
    unexpected = sorted(actual_set - expected_set)
    missing = sorted(expected_set - actual_set)
    if unexpected or missing:
        raise OrchestrationError(
            f"{where}: unexpected question ID(s) {unexpected}, missing "
            f"question ID(s) {missing}."
        )


# --- 1. Historical import (102 records) -- thin wrapper ---------------------


@dataclass(frozen=True)
class HistoricalImportContext:
    freeze_root: Path
    import_spec: ImportSpec
    expected_datasets: Mapping[str, ExpectedDataset]


def build_historical_import_context(freeze_root: Path) -> HistoricalImportContext:
    """Build the full 102-cell historical ImportSpec by delegating entirely
    to choicebench's own verified stage1_paper_freeze profile -- no new
    parsing/validation logic here, only wiring. This is the function the
    final-boss session calls with the real immutable bundle's
    ``historical_freeze/`` root; this session never calls it with that real
    root, only with synthetic fixtures in tests."""
    freeze_root = Path(freeze_root)
    translation = translate_stage1_paper_freeze(freeze_root)
    expected_datasets = build_stage1_expected_datasets(freeze_root)
    import_spec = build_stage1_import_spec(freeze_root, translation, expected_datasets)
    return HistoricalImportContext(
        freeze_root=freeze_root, import_spec=import_spec, expected_datasets=expected_datasets
    )


def execute_historical_import(
    context: HistoricalImportContext, *, run_id: str, workspace_root: Path, dry_run: bool = False
) -> ImportReport:
    request = ImportRequest(
        spec=context.import_spec,
        run_id=run_id,
        workspace_root=Path(workspace_root),
        strict=True,
        expected_datasets=context.expected_datasets,
        source_containment_root=context.freeze_root,
    )
    return execute_import(request, dry_run=dry_run)


# --- 2. Inference-repair overlay (28 of these, 3 rows each) -----------------


def build_repair_overlay_request(
    *,
    freeze_root: Path,
    cell_id: str,
    condition_digest: str,
    base_run_dir: Path,
    base_realization_id: str,
    expected_dataset: ExpectedDataset,
    patch_source: SourceArtifactSpec,
    patch_question_reasons: Mapping[str, str],
    authorized_question_ids: Sequence[str],
    dataset_snapshot_digest: str,
    cell_evidence_sha256: str,
    prediction_origins: Mapping[str, str],
    scope_disposition: str,
    run_id: str,
    workspace_root: Path,
    source_containment_root: Path | None = None,
) -> OverlayImportRequest:
    """Build the OverlayImportRequest for one of the 28 three-row
    inference-repair patches. ``patch_source`` must be a SourceArtifactSpec
    for the 3-row patch CSV with
    ``format_version=INFERENCE_REPAIR_PATCH_FORMAT_VERSION`` and
    ``classification='repaired'``. ``prediction_origins`` maps each of the 3
    replaced question IDs to a PredictionOrigin (native_inference or
    external_repair_inference -- the only two values ``derive_overlay``
    accepts for an inference_repair overlay). ``authorized_question_ids``
    must be the cell's exact 3-ID authorized set from
    ``repair_overlay_manifest.csv``/``.json`` in the analysis handoff -- the
    caller must load it from that manifest, never from the patch file itself
    (a patch's own declared IDs are exactly what this check exists to
    distrust: a correct row COUNT with a swapped/wrong ID must still fail).

    ``source_containment_root`` must be a single directory both the
    authorization source (under ``freeze_root``) and ``patch_source``
    resolve under -- the engine opens both under one shared root per call.
    In the real bundle these live under sibling directories (the immutable
    bundle vs. the verification package's ``analysis_handoff/``), so this
    defaults to ``freeze_root`` only for the common case where the caller
    has already arranged for both to share it; pass the real common
    ancestor explicitly otherwise."""
    reject_excluded_scope_disposition(scope_disposition, cell_id=cell_id)
    reject_diagnostic_paths([patch_source.path, patch_source.logical_path])
    require_exact_row_count(
        list(patch_question_reasons), expected=_REPAIR_PATCH_ROW_COUNT,
        where=f"repair patch for {cell_id!r}",
    )
    require_exact_question_id_set(
        list(patch_question_reasons), authorized_question_ids,
        where=f"repair patch for {cell_id!r} vs. repair_overlay_manifest authorization",
    )
    if patch_source.format_version != INFERENCE_REPAIR_PATCH_FORMAT_VERSION:
        raise OrchestrationError(
            f"Repair patch for {cell_id!r} has format_version "
            f"{patch_source.format_version!r}; expected "
            f"{INFERENCE_REPAIR_PATCH_FORMAT_VERSION!r}."
        )
    if patch_source.classification != "repaired":
        raise OrchestrationError(
            f"Repair patch for {cell_id!r} has classification "
            f"{patch_source.classification!r}; expected 'repaired'."
        )

    authorization, authorization_source = build_stage1_cell_authorization(
        freeze_root, _ledger(freeze_root), cell_id=cell_id, condition_digest=condition_digest,
        authorization_type="inference_repair", question_reasons=patch_question_reasons,
        dataset_snapshot_digest=dataset_snapshot_digest, cell_evidence_sha256=cell_evidence_sha256,
    )

    overlay = OverlaySpec(
        overlay_id=f"ovl_{cell_id}",
        base_run_path=Path(base_run_dir),
        base_condition_digest=condition_digest,
        base_realization_id=base_realization_id,
        base_realization_digest="",  # filled by caller from a verified base
        base_evidence_digests=dict(authorization.input_evidence_digests),
        base_validation_artifact_sha256="",
        base_result_sha256=None,
        source_id=patch_source.source_id,
        authorization_id=authorization.authorization_id,
        replacement_reasons=dict(patch_question_reasons),
        result_origin=ResultOriginSpec(
            derivation_origin="repair_overlay",
            default_prediction_origin=None,
            per_question_prediction_origins=dict(prediction_origins),
        ),
        lineage_notes={"cell_id": cell_id},
        implementation={
            "qualified_name": "final_paper_analysis.import_orchestration.inference_repair_overlay",
            "source_digest": patch_source.expected_sha256,
        },
        input_digest="0" * 64,
        preownership_output_digest="0" * 64,
        expected_evidence_status="complete",
    )

    return OverlayImportRequest(
        base_run_dir=Path(base_run_dir),
        base_realization_id=base_realization_id,
        authorization=authorization,
        authorization_source=authorization_source,
        overlay=overlay,
        overlay_source=patch_source,
        overlay_mapping={"question_id": "question_id", "prediction": "parsed_choice"},
        condition_digests={cell_id: condition_digest},
        expected_dataset=expected_dataset,
        run_id=run_id,
        workspace_root=Path(workspace_root),
        source_containment_root=Path(source_containment_root or freeze_root),
    )


def _ledger(freeze_root: Path) -> Mapping[str, str]:
    """Load the freeze's checksums.sha256 ledger the same way
    stage1_paper_freeze.py does internally (that loader is private, so this
    is a small, deliberate re-implementation rather than importing a
    leading-underscore name from choicebench -- per this session's
    constraint to avoid depending on choicebench internals beyond its public
    API)."""
    ledger: dict[str, str] = {}
    path = Path(freeze_root) / "checksums" / "checksums.sha256"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            digest, _, relative_path = line.partition("  ")
            ledger[relative_path] = digest
    return ledger


# --- 3. Offline-transformation overlay (6 of these, 3 rows each) -----------


def build_offline_transformation_overlay_request(
    *,
    freeze_root: Path,
    cell_id: str,
    condition_digest: str,
    base_run_dir: Path,
    base_realization_id: str,
    expected_dataset: ExpectedDataset,
    overlay_source: SourceArtifactSpec,
    question_reasons: Mapping[str, str],
    authorized_question_ids: Sequence[str],
    dataset_snapshot_digest: str,
    cell_evidence_sha256: str,
    scope_disposition: str,
    run_id: str,
    workspace_root: Path,
    source_containment_root: Path | None = None,
) -> OverlayImportRequest:
    """Build the OverlayImportRequest for one of the 6 three-row offline
    semantic-rematch overlays. The overlay CSV must already have been
    produced by Component 1's ``semantic_rematch.match_text_to_options``
    (this function never runs the matcher itself -- see
    ``build_semantic_rematch_overlay_rows``). offline_transformation
    overlays retain each underlying response's own prediction origin
    (overlays.py::derive_overlay), so no per-question origin is declared
    here. ``authorized_question_ids`` must be the cell's exact 3-ID
    authorized set from ``semantic_rematch_manifest.csv``/``.json`` in the
    analysis handoff -- never derived from the overlay CSV's own rows (a
    correct row COUNT with a swapped/wrong ID must still fail closed)."""
    reject_excluded_scope_disposition(scope_disposition, cell_id=cell_id)
    reject_diagnostic_paths([overlay_source.path, overlay_source.logical_path])
    require_exact_row_count(
        list(question_reasons), expected=_OFFLINE_TRANSFORMATION_ROW_COUNT,
        where=f"offline transformation overlay for {cell_id!r}",
    )
    require_exact_question_id_set(
        list(question_reasons), authorized_question_ids,
        where=f"offline transformation overlay for {cell_id!r} vs. semantic_rematch_manifest authorization",
    )
    if overlay_source.format_version != OFFLINE_TRANSFORMATION_PATCH_FORMAT_VERSION:
        raise OrchestrationError(
            f"Offline transformation overlay for {cell_id!r} has format_version "
            f"{overlay_source.format_version!r}; expected "
            f"{OFFLINE_TRANSFORMATION_PATCH_FORMAT_VERSION!r}."
        )

    authorization, authorization_source = build_stage1_cell_authorization(
        freeze_root, _ledger(freeze_root), cell_id=cell_id, condition_digest=condition_digest,
        authorization_type="offline_transformation", question_reasons=question_reasons,
        dataset_snapshot_digest=dataset_snapshot_digest, cell_evidence_sha256=cell_evidence_sha256,
    )

    overlay = OverlaySpec(
        overlay_id=f"ovl_{cell_id}_semantic_rematch",
        base_run_path=Path(base_run_dir),
        base_condition_digest=condition_digest,
        base_realization_id=base_realization_id,
        base_realization_digest="",
        base_evidence_digests=dict(authorization.input_evidence_digests),
        base_validation_artifact_sha256="",
        base_result_sha256=None,
        source_id=overlay_source.source_id,
        authorization_id=authorization.authorization_id,
        replacement_reasons=dict(question_reasons),
        result_origin=ResultOriginSpec(
            derivation_origin="offline_transformation",
            default_prediction_origin=None,
            per_question_prediction_origins={},
        ),
        lineage_notes={"cell_id": cell_id},
        implementation={
            "qualified_name": (
                "final_paper_analysis.semantic_rematch.match_text_to_options"
            ),
            "source_digest": overlay_source.expected_sha256,
        },
        input_digest="0" * 64,
        preownership_output_digest="0" * 64,
        expected_evidence_status="recoverable",
    )

    return OverlayImportRequest(
        base_run_dir=Path(base_run_dir),
        base_realization_id=base_realization_id,
        authorization=authorization,
        authorization_source=authorization_source,
        overlay=overlay,
        overlay_source=overlay_source,
        overlay_mapping={"question_id": "question_id", "prediction": "parsed_choice"},
        condition_digests={cell_id: condition_digest},
        expected_dataset=expected_dataset,
        run_id=run_id,
        workspace_root=Path(workspace_root),
        source_containment_root=Path(source_containment_root or freeze_root),
    )


def build_semantic_rematch_overlay_rows(
    *, free_text_responses: Mapping[str, str], choices_by_question: Mapping[str, Mapping[str, str]]
) -> dict[str, str | None]:
    """Run Component 1's matcher over exactly the rows given (never the real
    bundle in this session) and return {question_id: parsed_choice_or_None}.
    Zero model *inference* calls in the LLM/API sense -- the only model
    invoked is the local, deterministic sentence-transformers embedding
    model, and only for rows that reach rule 3.

    CRITICAL CALLER OBLIGATION (methodology lock #6, verified decision):
    ``free_text_responses`` MUST be sourced from the ORIGINAL HISTORICAL
    ``two_stage_v1`` artifact's ``free_text_response`` field for all 6
    cells -- NEVER the repaired ``two_stage_v1`` artifact, even for the 4
    API models whose ARC ``two_stage_v1`` cell was itself one of the 28
    inference repairs. This function is source-agnostic by design (it takes
    plain strings, not a file path) and cannot enforce this itself; the
    final-boss session must verify the identity check the handoff already
    performed (free_text_response identical between historical and repaired
    files for all 4 API models x 3 audited IDs) before calling this."""
    from final_paper_analysis.semantic_rematch import match_text_to_options, options_for_question

    results: dict[str, str | None] = {}
    for question_id, free_text in free_text_responses.items():
        options = options_for_question(question_id, choices_by_question[question_id])
        match = match_text_to_options(free_text, options)
        results[question_id] = match.label
    return results


# --- 4. Fourth-cell import (12 new conditions, not overlays) ----------------


def compute_fourth_cell_evidence_status(rows: Sequence[Mapping[str, str]]) -> str:
    """Compute the true evidence_status for a fourth-cell realization from
    its own row content, instead of assuming 'complete'. Discovered against
    real data: 4 of the 12 real fourth-cell CSVs (gemini-2.5-flash x
    {arc_challenge, mmlu}; meta-llama-3.1-8b-instruct x {arc_challenge,
    mmlu}) have a nonzero number of rows with an empty ``parsed_choice`` --
    the Stage-2 LLM-matcher's own genuine parse misses (a real model
    response is present, no transport/execution error occurred; the same
    "legitimate parse failure with a stored raw response" category as the
    historical freeze's own undocumented parse_missing rows). All 12 real
    files are independently confirmed structurally complete (1000/1000
    unique question IDs, no duplicates), so the only two reachable outcomes
    here are 'malformed' (>=1 row lacks a usable prediction) or 'complete'
    (all rows have one) -- mirrors validation.py's own
    ``compute_evidence_status`` for the malformed/complete branches
    specifically (fourth cells never declare qualifications or
    recoverable_question_ids, so the qualified/recoverable branches never
    apply here)."""
    if any(not (row.get("predicted_option") or row.get("parsed_choice") or "").strip() for row in rows):
        return "malformed"
    return "complete"


def build_fourth_cell_condition(
    *,
    condition_key: str,
    dataset_id: str,
    model_key: str,
    expected_question_ids: Sequence[str],
    evidence_status: str = "complete",
) -> ImportConditionSpec:
    """Build the ImportConditionSpec for one of the 12 fourth-cell
    (visible_llm_matcher) realizations. This is a brand-new condition, not an
    overlay -- engine.py's base-import path (_build_realization_for_condition)
    always assigns derivation_origin='external_import' itself, matching the
    handoff's lineage_derivation_concept exactly with no ambiguity.
    prediction_origin uses FOURTH_CELL_PREDICTION_ORIGIN -- see module
    docstring for the documented reasoning; this is the one methodology-lock
    decision this session makes on the handoff's behalf and must be
    reconfirmed before the authoritative import.

    ``evidence_status`` must be the REAL cell's computed status (see
    ``compute_fourth_cell_evidence_status``) -- the 'complete' default is a
    convenience only for cells actually confirmed complete (e.g. tests
    constructing an all-parseable fixture); real callers must always compute
    and pass this explicitly rather than relying on the default, or a
    genuinely malformed real cell will hit the engine's declared-vs-computed
    mismatch check and fail closed (as discovered against the real
    gemini-2.5-flash/meta-llama-3.1-8b-instruct fourth-cell files)."""
    return ImportConditionSpec(
        condition_key=condition_key,
        source_ids=(condition_key,),
        dataset_id=dataset_id,
        model_key=model_key,
        method_key="visible_llm_matcher",
        prompt_key="visible_llm_matcher_v1",
        seed=42,
        calibration_identity=None,
        preflight_identity=None,
        protocol_settings={},
        generation_parameters={},
        unknown_reasons={
            "calibration_identity": "not recorded by the fourth-cell bundle",
            "preflight_identity": "not recorded by the fourth-cell bundle",
        },
        expected_question_ids=tuple(expected_question_ids),
        evidence_status=evidence_status,
        scope_disposition="included",
        executable=None,
        qualifications=(),
        limitations=(),
        damaged_question_ids=(),
        recoverable_question_ids=(),
        result_origin=ResultOriginSpec(
            derivation_origin="external_import",
            default_prediction_origin=FOURTH_CELL_PREDICTION_ORIGIN,
            per_question_prediction_origins={},
        ),
    )


def build_fourth_cell_source(
    *,
    condition_key: str,
    csv_path: Path,
    expected_sha256: str,
    expected_columns: Sequence[str],
    generation_commit: str,
    collection_backfill_commit: str,
) -> SourceArtifactSpec:
    """Build the SourceArtifactSpec for one fourth-cell realization CSV.
    Records both commits the handoff kept distinct (methodology lock #7):
    the experiment-generation commit and the collection-backfill HEAD --
    ``source_commit`` takes the generation commit, and both are also
    recorded in ``notes`` so neither is lost.

    The real fourth-cell CSVs carry a single structured ``choices_json``
    column (``[{"text": ..., "source_index": 0}, ...]``), not separate
    ``choice_a``/``choice_b``/``choice_c``/``choice_d`` columns -- confirmed
    against the real bundle files, not assumed. ``structured_label_key``
    points at ``source_index`` (an integer position, not a pre-computed
    letter); ``csv_adapter._validate_structured_choices`` letterizes it
    (0 -> A, 1 -> B, ...) -- see that function's docstring/comment for the
    positional-label extension this relies on."""
    reject_diagnostic_paths([csv_path])
    return SourceArtifactSpec(
        source_id=condition_key,
        path=Path(csv_path),
        logical_path=Path(csv_path).name,
        expected_sha256=expected_sha256,
        format="csv",
        format_version=EXTERNAL_RESULT_ROWS_FORMAT_VERSION,
        classification="derived",
        dialect=CsvDialectSpec(),
        columns={"question_id": "question_id", "correct_option": "correct_option", "prediction": "parsed_choice"},
        expected_columns=tuple(expected_columns),
        ignored_columns={},
        null_values=("",),
        numeric_columns=(),
        option_mapping=OptionMappingSpec(
            mode="structured_json",
            ordered_columns=(),
            structured_column="choices_json",
            structured_label_key="source_index",
            structured_text_key="text",
        ),
        extra_field_policy="preserve_unmapped",
        preserve_namespace="final_paper_analysis",
        source_run_id=None,
        source_repository="choicebench",
        source_commit=generation_commit,
        notes={
            "experiment_generation_commit": generation_commit,
            "collection_backfill_commit": collection_backfill_commit,
        },
    )


def build_fourth_cell_import_request(
    *,
    condition: ImportConditionSpec,
    source: SourceArtifactSpec,
    dataset_id: str,
    expected_dataset: ExpectedDataset,
    model_key: str,
    backend: str,
    run_id: str,
    workspace_root: Path,
    source_containment_root: Path,
) -> ImportRequest:
    """Build the full ImportRequest for one fourth-cell realization -- a
    single-condition ImportSpec sharing the same model/method/prompt/dataset
    plumbing the historical profile uses, but scoped to just this one new
    condition (base import, not an overlay). ``backend`` must be 'api' or
    'local' -- this distinction is load-bearing (methodology lock #8: local
    vs. API model identities stay separate) so it is a required argument,
    not left undeclared like the historical profile's undocumented fields."""
    spec = ImportSpec(
        schema_version="choicebench.import-spec.v1",
        import_name=f"Fourth cell: {condition.condition_key}",
        sources=(source,),
        datasets=(),
        models=(
            ImportModelSpec(
                model_key=model_key, display_name=model_key, backend=backend, provider=backend,
                revision=None, effective_parameters={},
                unknown_reasons={"revision": "not recorded by the fourth-cell bundle"},
                native_compatibility_identity=None,
            ),
        ),
        methods=(
            ImportMethodSpec(
                method_key="visible_llm_matcher", name="visible_llm_matcher",
                effective_parameters={}, implementation=None,
                unknown_reasons={"implementation": "not recorded by the fourth-cell bundle"},
                native_compatibility_identity=None,
            ),
        ),
        prompts=(
            ImportPromptSpec(
                prompt_key="visible_llm_matcher_v1", template_identity=None, template_digest=None,
                template_contents=None,
                unknown_reason=(
                    "Fourth-cell realizations reuse the historical option_matching.txt "
                    "Stage-2 prompt (methodology lock #7); no separate template identity "
                    "was captured by the handoff for this profile."
                ),
                native_compatibility_identity=None,
            ),
        ),
        conditions=(condition,),
        authorizations=(),
        overlays=(),
        metrics=["accuracy"],
        provenance={"producer_request_id": {"value": None, "reason": "fourth_cell_import"}},
        audit={"source_location": str(source_containment_root)},
    )
    return ImportRequest(
        spec=spec,
        run_id=run_id,
        workspace_root=Path(workspace_root),
        strict=True,
        expected_datasets={dataset_id: expected_dataset},
        source_containment_root=Path(source_containment_root),
    )
