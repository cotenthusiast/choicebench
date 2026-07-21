"""Tests for the import/overlay orchestration layer (spec section 2).

All fixtures are synthetic/disposable (a tiny fake Stage-1 freeze reusing the
main choicebench repo's own test helpers) -- per the session's scope
boundary, the real immutable bundle is never read or imported here. These
tests prove the orchestration functions work end-to-end against the real
generic importer/engine (no mocks of choicebench.importing itself), which is
exactly what the later "final-boss" session will call against the real
bundle.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from choicebench.importing.engine import execute_import, execute_overlay_import, verify_import_run
from choicebench.importing.evidence import open_verified_source
from choicebench.importing.identity import build_import_semantic_identity
from choicebench.importing.profiles.stage1_paper_freeze import (
    build_stage1_expected_datasets,
    build_stage1_import_spec,
    translate_stage1_paper_freeze,
)
from choicebench.importing.schema import CsvDialectSpec, OptionMappingSpec, SourceArtifactSpec

from final_paper_analysis.import_orchestration import (
    EXTERNAL_RESULT_ROWS_FORMAT_VERSION,
    FOURTH_CELL_PREDICTION_ORIGIN,
    INFERENCE_REPAIR_PATCH_FORMAT_VERSION,
    OFFLINE_TRANSFORMATION_PATCH_FORMAT_VERSION,
    OrchestrationError,
    build_fourth_cell_condition,
    build_offline_transformation_overlay_request,
    build_repair_overlay_request,
    build_semantic_rematch_overlay_rows,
    execute_historical_import,
    reject_diagnostic_paths,
    reject_excluded_scope_disposition,
    require_exact_question_id_set,
    require_exact_row_count,
)
from tests.importing.test_stage1_import_spec import _arc_result_rows, _build_full_freeze, _result_row
from tests.importing.test_stage1_profile import _cell, _default_arc_normalized_rows


# --- Fail-closed guard unit tests (no importer objects needed) --------------


def test_reject_diagnostic_paths_rejects_any_path_under_the_marker():
    with pytest.raises(OrchestrationError, match="diagnostics_excluded_from_import"):
        reject_diagnostic_paths(
            ["bundle/diagnostics_excluded_from_import/fourth_cell_canary.csv"]
        )


def test_reject_diagnostic_paths_allows_authoritative_paths():
    reject_diagnostic_paths(["bundle/provenance/logs/canary_qwen_two_stage_v2_9399740.log"])


def test_reject_excluded_scope_disposition_rejects_excluded_from_paper_matrix():
    with pytest.raises(OrchestrationError, match="paper-scope-excluded"):
        reject_excluded_scope_disposition("excluded_from_paper_matrix", cell_id="x")


def test_reject_excluded_scope_disposition_allows_included():
    reject_excluded_scope_disposition("included", cell_id="x")


def test_require_exact_row_count_rejects_wrong_count():
    with pytest.raises(OrchestrationError, match="exactly 3"):
        require_exact_row_count(["q1", "q2"], expected=3, where="test")


def test_require_exact_row_count_rejects_duplicates():
    with pytest.raises(OrchestrationError, match="duplicate"):
        require_exact_row_count(["q1", "q1", "q2"], expected=3, where="test")


def test_require_exact_row_count_accepts_exact_match():
    require_exact_row_count(["q1", "q2", "q3"], expected=3, where="test")


def test_require_exact_question_id_set_rejects_unexpected_and_missing():
    with pytest.raises(OrchestrationError, match="unexpected question ID.*missing question ID"):
        require_exact_question_id_set(["q1", "q9"], ["q1", "q2"], where="test")


def test_require_exact_question_id_set_accepts_exact_match_any_order():
    require_exact_question_id_set(["q2", "q1"], ["q1", "q2"], where="test")


# --- Semantic-rematch overlay row generation (Component 1 wiring) -----------


def test_build_semantic_rematch_overlay_rows_produces_parsed_choices():
    free_text = {"q1": "the second option", "q2": "totally unrelated nonsense output"}
    choices = {
        "q1": {"A": "the first option", "B": "the second option", "C": "the third option"},
        "q2": {"A": "cats", "B": "dogs", "C": "birds"},
    }
    results = build_semantic_rematch_overlay_rows(
        free_text_responses=free_text, choices_by_question=choices
    )
    assert results["q1"] == "B"
    # q2's answer shares no exact/containment overlap and is semantically far
    # from all three options -- expect it to fail to match (None), not be
    # coerced into a guess.
    assert results["q2"] is None


def test_build_semantic_rematch_overlay_rows_restricts_three_option_questions():
    from final_paper_analysis.semantic_rematch import THREE_OPTION_QUESTION_IDS

    qid = next(iter(THREE_OPTION_QUESTION_IDS))
    free_text = {qid: "third"}
    choices = {qid: {"A": "first", "B": "second", "C": "third", "D": "third"}}
    results = build_semantic_rematch_overlay_rows(
        free_text_responses=free_text, choices_by_question=choices
    )
    # "third" exact-matches C (dict iteration order A,B,C before D would have
    # picked C anyway) -- the real point is D is never a candidate at all,
    # verified directly in test_semantic_rematch.py. Here we just confirm
    # the wiring reaches the right label.
    assert results[qid] == "C"


# --- Fourth-cell condition construction -------------------------------------


def test_build_fourth_cell_condition_uses_external_import_and_documented_prediction_origin():
    condition = build_fourth_cell_condition(
        condition_key="fourth_cell__gpt-4-1-mini__arc_challenge",
        dataset_id="arc_challenge",
        model_key="gpt-4-1-mini",
        expected_question_ids=("q1", "q2"),
    )
    assert condition.result_origin.derivation_origin == "external_import"
    assert condition.result_origin.default_prediction_origin == FOURTH_CELL_PREDICTION_ORIGIN
    assert FOURTH_CELL_PREDICTION_ORIGIN == "native_inference"
    assert condition.method_key == "visible_llm_matcher"
    assert condition.scope_disposition == "included"


def _choices_json_row(question_id, correct_option, choices, parsed_choice):
    """Real-shaped fourth-cell row: a single structured choices_json column
    (source_index positional, no explicit letter label) instead of separate
    choice_a/b/c/d columns -- matches the actual bundle files exactly, not a
    simplified fixture."""
    choices_json = json.dumps(
        [{"text": text, "source_index": i} for i, text in enumerate(choices)]
    )
    return {
        "question_id": question_id, "correct_option": correct_option,
        "choices_json": choices_json, "parsed_choice": parsed_choice,
    }


def test_fourth_cell_import_executes_end_to_end_against_the_real_engine(tmp_path):
    from final_paper_analysis.import_orchestration import (
        build_fourth_cell_import_request,
        build_fourth_cell_source,
    )

    ctx = _historical_run(tmp_path)
    all_ids = ctx["arc_ids"]
    fourth_cell_rows = [
        _choices_json_row(row["question_id"], row["correct_option"],
                           [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]],
                           row["correct_option"])
        for row in _arc_result_rows(all_correct=True)
    ]

    import csv

    csv_path = tmp_path / "fourth_cell.csv"
    fieldnames = ["question_id", "correct_option", "choices_json", "parsed_choice"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fourth_cell_rows)
    expected_sha256 = sha256(csv_path.read_bytes()).hexdigest()

    condition_key = "fourth_cell__gpt-4-1-mini__arc_challenge"
    condition = build_fourth_cell_condition(
        condition_key=condition_key, dataset_id="arc_challenge",
        model_key="gpt-4-1-mini", expected_question_ids=all_ids,
    )
    source = build_fourth_cell_source(
        condition_key=condition_key, csv_path=csv_path, expected_sha256=expected_sha256,
        expected_columns=fieldnames, generation_commit="b9454aa",
        collection_backfill_commit="02e5437c3e91f9a13ccf3c8bb357106e6a16e210",
    )
    request = build_fourth_cell_import_request(
        condition=condition, source=source, dataset_id="arc_challenge",
        expected_dataset=ctx["expected_dataset"], model_key="gpt-4-1-mini", backend="api",
        run_id="fourth-cell-run", workspace_root=tmp_path / "fourth_cell_workspace",
        source_containment_root=tmp_path,
    )

    report = execute_import(request, dry_run=False)
    assert report.import_state == "imported", report.failures

    run_dir = tmp_path / "fourth_cell_workspace" / "runs" / "fourth-cell-run"
    verified = verify_import_run(run_dir)
    (realization,) = verified.realizations.values()
    assert set(realization.rows_by_question_id) == set(all_ids)
    correct_by_qid = {row["question_id"]: row["correct_option"] for row in fourth_cell_rows}
    for qid in all_ids:
        assert realization.rows_by_question_id[qid]["predicted_option"] == correct_by_qid[qid]
        assert realization.prediction_origins[qid] == FOURTH_CELL_PREDICTION_ORIGIN


def test_fourth_cell_import_with_genuine_parse_misses_declares_malformed(tmp_path):
    """Regression: real fourth-cell CSVs are not all fully parseable (found
    against the real bundle: gemini-2.5-flash and meta-llama-3.1-8b-instruct
    both have genuine Stage-2 LLM-matcher parse misses on some rows -- a real
    model response present, no error, just unparseable). The condition's
    declared evidence_status must be computed from the actual data
    (compute_fourth_cell_evidence_status), not hardcoded to 'complete', or
    the engine's declared-vs-computed mismatch check fails the whole import."""
    from final_paper_analysis.import_orchestration import (
        build_fourth_cell_import_request,
        build_fourth_cell_source,
        compute_fourth_cell_evidence_status,
    )

    ctx = _historical_run(tmp_path)
    all_ids = ctx["arc_ids"]
    base_rows = _arc_result_rows(all_correct=True)
    # blank out one row's parsed_choice -- a genuine, benign parse miss,
    # unrelated to the other 3 rows, same as the real gemini/meta-llama data.
    fourth_cell_rows = [
        _choices_json_row(
            row["question_id"], row["correct_option"],
            [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]],
            "" if row["question_id"] == all_ids[0] else row["correct_option"],
        )
        for row in base_rows
    ]

    computed_status = compute_fourth_cell_evidence_status(fourth_cell_rows)
    assert computed_status == "malformed"

    import csv

    csv_path = tmp_path / "fourth_cell_malformed.csv"
    fieldnames = ["question_id", "correct_option", "choices_json", "parsed_choice"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fourth_cell_rows)
    expected_sha256 = sha256(csv_path.read_bytes()).hexdigest()

    condition_key = "fourth_cell__gpt-4-1-mini__arc_challenge__malformed"
    condition = build_fourth_cell_condition(
        condition_key=condition_key, dataset_id="arc_challenge",
        model_key="gpt-4-1-mini", expected_question_ids=all_ids,
        evidence_status=computed_status,
    )
    source = build_fourth_cell_source(
        condition_key=condition_key, csv_path=csv_path, expected_sha256=expected_sha256,
        expected_columns=fieldnames, generation_commit="b9454aa",
        collection_backfill_commit="02e5437c3e91f9a13ccf3c8bb357106e6a16e210",
    )
    request = build_fourth_cell_import_request(
        condition=condition, source=source, dataset_id="arc_challenge",
        expected_dataset=ctx["expected_dataset"], model_key="gpt-4-1-mini", backend="api",
        run_id="fourth-cell-malformed-run", workspace_root=tmp_path / "fourth_cell_workspace_malformed",
        source_containment_root=tmp_path,
    )
    report = execute_import(request, dry_run=False)
    assert report.import_state == "imported", report.failures

    run_dir = tmp_path / "fourth_cell_workspace_malformed" / "runs" / "fourth-cell-malformed-run"
    verified = verify_import_run(run_dir)
    (realization,) = verified.realizations.values()
    assert realization.rows_by_question_id[all_ids[0]]["predicted_option"] is None
    for qid in all_ids[1:]:
        assert realization.rows_by_question_id[qid]["predicted_option"] is not None
    evidence = verified.manifest["payload"]["realizations"][realization.realization_id]["identity"]["realization"]["evidence"]
    assert evidence["evidence_status"] == "malformed"


def test_fourth_cell_source_rejects_choices_json_missing_source_index(tmp_path):
    """Regression: a structured choice object with neither a string label nor
    a valid positional source_index must still fail closed, not silently
    treat it as option-less."""
    ctx = _historical_run(tmp_path)
    all_ids = ctx["arc_ids"]
    rows = [
        {
            "question_id": qid, "correct_option": "A",
            "choices_json": json.dumps([{"text": "x"}, {"text": "y"}]),  # no source_index/label
            "parsed_choice": "A",
        }
        for qid in all_ids
    ]
    import csv

    from final_paper_analysis.import_orchestration import (
        build_fourth_cell_import_request,
        build_fourth_cell_source,
    )

    csv_path = tmp_path / "fourth_cell_bad.csv"
    fieldnames = ["question_id", "correct_option", "choices_json", "parsed_choice"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    expected_sha256 = sha256(csv_path.read_bytes()).hexdigest()

    condition_key = "fourth_cell__gpt-4-1-mini__arc_challenge__bad"
    condition = build_fourth_cell_condition(
        condition_key=condition_key, dataset_id="arc_challenge",
        model_key="gpt-4-1-mini", expected_question_ids=all_ids,
    )
    source = build_fourth_cell_source(
        condition_key=condition_key, csv_path=csv_path, expected_sha256=expected_sha256,
        expected_columns=fieldnames, generation_commit="b9454aa",
        collection_backfill_commit="02e5437c3e91f9a13ccf3c8bb357106e6a16e210",
    )
    request = build_fourth_cell_import_request(
        condition=condition, source=source, dataset_id="arc_challenge",
        expected_dataset=ctx["expected_dataset"], model_key="gpt-4-1-mini", backend="api",
        run_id="fourth-cell-run-bad", workspace_root=tmp_path / "fourth_cell_workspace_bad",
        source_containment_root=tmp_path,
    )
    report = execute_import(request, dry_run=False)
    assert report.import_state == "failed"
    assert any("structured choices" in f.get("sanitized_message", "") for f in report.failures)


# --- Historical import: thin wrapper works end-to-end -----------------------


def test_execute_historical_import_imports_all_synthetic_cells(tmp_path):
    from final_paper_analysis.import_orchestration import build_historical_import_context

    arc_ids = [row["question_id"] for row in _default_arc_normalized_rows()]
    clean_cell_id = "cbp__model-a__arc_challenge__baseline"
    # See _historical_run's comment: _offline_authority() requires at least
    # one recoverable cell to exist in any synthetic freeze fixture.
    recoverable_cell_id = "cbp__model-a__arc_challenge__semantic_matching_v1"
    cells = [
        _cell(clean_cell_id, "baseline", "canonical_complete"),
        _cell(
            recoverable_cell_id, "semantic_matching_v1",
            "recoverable_from_existing_artifacts", recoverable=(arc_ids[0],),
        ),
    ]
    for cell in cells:
        cell["benchmark"] = "arc_challenge"
        cell["expected_question_count"] = len(arc_ids)
    rows = {
        clean_cell_id: _arc_result_rows(all_correct=True),
        recoverable_cell_id: _arc_result_rows(all_correct=False),
    }
    freeze_root = _build_full_freeze(tmp_path / "freeze_root", cells=cells, cell_result_rows=rows)

    context = build_historical_import_context(freeze_root)
    assert len(context.import_spec.conditions) == 2

    report = execute_historical_import(
        context, run_id="historical-run", workspace_root=tmp_path / "workspace"
    )
    assert report.import_state == "imported", report.failures
    assert report.wrote_artifacts is True


# --- Repair overlay: full end-to-end against the real engine ----------------


def _repair_patch_source(tmp_path: Path, *, rows: list[dict]) -> tuple[SourceArtifactSpec, dict[str, str]]:
    import csv

    patch_path = tmp_path / "patch_3rows.csv"
    fieldnames = ["question_id", "correct_option", "choice_a", "choice_b", "choice_c", "choice_d", "parsed_choice"]
    with patch_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    data = patch_path.read_bytes()
    source = SourceArtifactSpec(
        source_id="repair-patch",
        path=patch_path,
        logical_path="patch_3rows.csv",
        expected_sha256=sha256(data).hexdigest(),
        format="csv",
        format_version=INFERENCE_REPAIR_PATCH_FORMAT_VERSION,
        classification="repaired",
        dialect=CsvDialectSpec(),
        columns={"question_id": "question_id", "prediction": "parsed_choice"},
        expected_columns=tuple(fieldnames),
        ignored_columns={},
        null_values=("",),
        numeric_columns=(),
        option_mapping=OptionMappingSpec(
            mode="ordered_columns",
            ordered_columns=("choice_a", "choice_b", "choice_c", "choice_d"),
            structured_column=None, structured_label_key=None, structured_text_key=None,
        ),
        extra_field_policy="preserve_unmapped",
        preserve_namespace="final_paper_analysis",
        source_run_id=None, source_repository="two-stage-prompting",
        source_commit="b8e784f3eb5d2a727a97eb675140b383a34584fa",
        notes={},
    )
    question_reasons = {row["question_id"]: "malformed prediction requires inference repair" for row in rows}
    return source, question_reasons


def _historical_run(tmp_path: Path):
    arc_ids = [row["question_id"] for row in _default_arc_normalized_rows()]
    cell_id = "cbp__model-a__arc_challenge__two_stage_v1"
    # _offline_authority() (stage1_paper_freeze.py) requires that if ANY cell
    # declares recoverable_question_ids, all such cells agree on the exact
    # same set -- with zero recoverable cells it still requires exactly one
    # distinct set, so a synthetic freeze must always carry at least one
    # recoverable cell (matching the real freeze's own 6-cell/18-pair
    # invariant), even when the test target is a different cell.
    recoverable_cell_id = "cbp__model-a__arc_challenge__semantic_matching_v1"
    cells = [
        _cell(cell_id, "two_stage_v1", "canonical_complete"),
        _cell(
            recoverable_cell_id, "semantic_matching_v1",
            "recoverable_from_existing_artifacts", recoverable=(arc_ids[0],),
        ),
    ]
    for cell in cells:
        cell["benchmark"] = "arc_challenge"
        cell["expected_question_count"] = len(arc_ids)
    rows = {
        cell_id: _arc_result_rows(all_correct=True),
        recoverable_cell_id: _arc_result_rows(all_correct=False),
    }
    freeze_root = _build_full_freeze(tmp_path / "freeze_root", cells=cells, cell_result_rows=rows)

    translation = translate_stage1_paper_freeze(freeze_root)
    expected_datasets = build_stage1_expected_datasets(freeze_root)
    spec = build_stage1_import_spec(freeze_root, translation, expected_datasets)

    models_by_key = {m.model_key: m for m in spec.models}
    methods_by_key = {m.method_key: m for m in spec.methods}
    prompts_by_key = {p.prompt_key: p for p in spec.prompts}
    conditions_by_key = {c.condition_key: c for c in spec.conditions}
    condition = conditions_by_key[cell_id]
    dataset = expected_datasets["arc_challenge"]
    semantic = build_import_semantic_identity(
        condition=condition, dataset=dataset,
        model=models_by_key[condition.model_key], method=methods_by_key[condition.method_key],
        prompt=prompts_by_key[condition.prompt_key],
    )
    condition_digest = semantic.condition["condition_digest"]

    from choicebench.importing.engine import ImportRequest

    workspace_root = tmp_path / "workspace"
    request = ImportRequest(
        spec=spec, run_id="base-run", workspace_root=workspace_root, strict=True,
        expected_datasets=expected_datasets, source_containment_root=freeze_root,
    )
    execute_import(request, dry_run=False)
    base_run_dir = workspace_root / "runs" / "base-run"
    verified = verify_import_run(base_run_dir)
    (base,) = [
        realization for realization in verified.realizations.values()
        if realization.condition_digest == condition_digest
    ]

    sources_by_id = {s.source_id: s for s in spec.sources}
    cell_source = sources_by_id[cell_id]
    return {
        "freeze_root": freeze_root, "workspace_root": workspace_root, "base_run_dir": base_run_dir,
        "base": base, "condition_digest": condition_digest, "expected_dataset": dataset,
        "arc_ids": arc_ids, "cell_id": cell_id, "cell_source_sha256": cell_source.expected_sha256,
    }


def test_build_repair_overlay_request_executes_a_real_three_row_repair(tmp_path):
    ctx = _historical_run(tmp_path)
    repaired_ids = ctx["arc_ids"][:3]
    patch_rows = [
        _result_row(qid, "A", ["a1", "b1", "c1", "d1"], "A") for qid in repaired_ids
    ]
    patch_source, question_reasons = _repair_patch_source(tmp_path, rows=patch_rows)

    overlay_request = build_repair_overlay_request(
        freeze_root=ctx["freeze_root"], cell_id=ctx["cell_id"], condition_digest=ctx["condition_digest"],
        base_run_dir=ctx["base_run_dir"], base_realization_id=ctx["base"].realization_id,
        expected_dataset=ctx["expected_dataset"], patch_source=patch_source,
        patch_question_reasons=question_reasons, authorized_question_ids=repaired_ids,
        dataset_snapshot_digest=ctx["expected_dataset"].snapshot_digest,
        cell_evidence_sha256=ctx["cell_source_sha256"],
        prediction_origins={qid: "native_inference" for qid in repaired_ids},
        scope_disposition="included", run_id="repair-run", workspace_root=ctx["workspace_root"],
        source_containment_root=tmp_path,
    )
    # OverlaySpec.base_realization_digest/base_validation_artifact_sha256 are
    # populated from the caller's verified base -- fill them in here exactly
    # as the final-boss session's real caller would.
    from dataclasses import replace

    overlay_request = replace(
        overlay_request,
        overlay=replace(
            overlay_request.overlay,
            base_realization_digest=ctx["base"].realization_digest,
            base_validation_artifact_sha256=ctx["base"].validation_artifact_sha256,
            base_result_sha256=ctx["base"].result_sha256,
        ),
    )

    report = execute_overlay_import(overlay_request, dry_run=False)
    assert report.import_state == "imported", report.failures

    overlay_run_dir = ctx["workspace_root"] / "runs" / "repair-run"
    verified = verify_import_run(overlay_run_dir)
    (derived,) = verified.realizations.values()
    for qid in repaired_ids:
        assert derived.rows_by_question_id[qid]["predicted_option"] == "A"
        assert derived.prediction_origins[qid] == "native_inference"
    # the untouched 4th row keeps its historical origin.
    other_id = ctx["arc_ids"][3]
    assert derived.prediction_origins[other_id] == "external_historical_inference"


def test_build_repair_overlay_request_rejects_excluded_pride_cell(tmp_path):
    ctx = _historical_run(tmp_path)
    repaired_id = ctx["arc_ids"][0]
    patch_rows = [_result_row(repaired_id, "A", ["a1", "b1", "c1", "d1"], "A")]
    patch_source, question_reasons = _repair_patch_source(tmp_path, rows=patch_rows)

    with pytest.raises(OrchestrationError, match="paper-scope-excluded"):
        build_repair_overlay_request(
            freeze_root=ctx["freeze_root"], cell_id=ctx["cell_id"], condition_digest=ctx["condition_digest"],
            base_run_dir=ctx["base_run_dir"], base_realization_id=ctx["base"].realization_id,
            expected_dataset=ctx["expected_dataset"], patch_source=patch_source,
            patch_question_reasons=question_reasons, authorized_question_ids=[repaired_id],
            dataset_snapshot_digest=ctx["expected_dataset"].snapshot_digest,
            cell_evidence_sha256=ctx["cell_source_sha256"],
            prediction_origins={repaired_id: "native_inference"},
            scope_disposition="excluded_from_paper_matrix", run_id="repair-run",
            workspace_root=ctx["workspace_root"],
        )


def test_build_repair_overlay_request_rejects_wrong_row_count(tmp_path):
    ctx = _historical_run(tmp_path)
    patch_rows = [
        _result_row(ctx["arc_ids"][0], "A", ["a1", "b1", "c1", "d1"], "A"),
        _result_row(ctx["arc_ids"][1], "B", ["a2", "b2", "c2", "d2"], "B"),
    ]
    patch_source, question_reasons = _repair_patch_source(tmp_path, rows=patch_rows)

    with pytest.raises(OrchestrationError, match="exactly 3"):
        build_repair_overlay_request(
            freeze_root=ctx["freeze_root"], cell_id=ctx["cell_id"], condition_digest=ctx["condition_digest"],
            base_run_dir=ctx["base_run_dir"], base_realization_id=ctx["base"].realization_id,
            expected_dataset=ctx["expected_dataset"], patch_source=patch_source,
            patch_question_reasons=question_reasons, authorized_question_ids=list(question_reasons),
            dataset_snapshot_digest=ctx["expected_dataset"].snapshot_digest,
            cell_evidence_sha256=ctx["cell_source_sha256"],
            prediction_origins={qid: "native_inference" for qid in question_reasons},
            scope_disposition="included", run_id="repair-run", workspace_root=ctx["workspace_root"],
        )


def test_build_repair_overlay_request_rejects_correct_count_wrong_id_swap(tmp_path):
    """Fix A regression: a patch with the right row COUNT (3) but one ID
    swapped for an unauthorized one must fail closed, not silently pass.
    build_repair_overlay_request must cross-check the patch's actual question
    IDs against the caller-supplied authorized set (in the real pipeline,
    loaded from repair_overlay_manifest.csv) -- it must not just trust
    whatever IDs the patch file itself declares."""
    ctx = _historical_run(tmp_path)
    # patch declares arc_ids[0], arc_ids[1], arc_ids[3] (3 rows, right count)
    # but the manifest's real authorized set is arc_ids[0], arc_ids[1], arc_ids[2].
    swapped_ids = [ctx["arc_ids"][0], ctx["arc_ids"][1], ctx["arc_ids"][3]]
    authorized_ids = [ctx["arc_ids"][0], ctx["arc_ids"][1], ctx["arc_ids"][2]]
    patch_rows = [
        _result_row(qid, "A", ["a1", "b1", "c1", "d1"], "A") for qid in swapped_ids
    ]
    patch_source, question_reasons = _repair_patch_source(tmp_path, rows=patch_rows)

    with pytest.raises(OrchestrationError, match="unexpected question ID.*missing question ID"):
        build_repair_overlay_request(
            freeze_root=ctx["freeze_root"], cell_id=ctx["cell_id"], condition_digest=ctx["condition_digest"],
            base_run_dir=ctx["base_run_dir"], base_realization_id=ctx["base"].realization_id,
            expected_dataset=ctx["expected_dataset"], patch_source=patch_source,
            patch_question_reasons=question_reasons, authorized_question_ids=authorized_ids,
            dataset_snapshot_digest=ctx["expected_dataset"].snapshot_digest,
            cell_evidence_sha256=ctx["cell_source_sha256"],
            prediction_origins={qid: "native_inference" for qid in swapped_ids},
            scope_disposition="included", run_id="repair-run", workspace_root=ctx["workspace_root"],
        )


def test_build_repair_overlay_request_rejects_wrong_format_version(tmp_path):
    ctx = _historical_run(tmp_path)
    repaired_ids = ctx["arc_ids"][:3]
    patch_rows = [
        _result_row(qid, "A", ["a1", "b1", "c1", "d1"], "A") for qid in repaired_ids
    ]
    patch_source, question_reasons = _repair_patch_source(tmp_path, rows=patch_rows)
    from dataclasses import replace

    wrong_source = replace(patch_source, format_version="some.other.v1")

    with pytest.raises(OrchestrationError, match="format_version"):
        build_repair_overlay_request(
            freeze_root=ctx["freeze_root"], cell_id=ctx["cell_id"], condition_digest=ctx["condition_digest"],
            base_run_dir=ctx["base_run_dir"], base_realization_id=ctx["base"].realization_id,
            expected_dataset=ctx["expected_dataset"], patch_source=wrong_source,
            patch_question_reasons=question_reasons, authorized_question_ids=repaired_ids,
            dataset_snapshot_digest=ctx["expected_dataset"].snapshot_digest,
            cell_evidence_sha256=ctx["cell_source_sha256"],
            prediction_origins={qid: "native_inference" for qid in repaired_ids},
            scope_disposition="included", run_id="repair-run", workspace_root=ctx["workspace_root"],
        )


def test_build_repair_overlay_request_rejects_diagnostic_path(tmp_path):
    ctx = _historical_run(tmp_path)
    repaired_id = ctx["arc_ids"][0]
    patch_rows = [_result_row(repaired_id, "A", ["a1", "b1", "c1", "d1"], "A")]
    patch_source, question_reasons = _repair_patch_source(tmp_path, rows=patch_rows)
    from dataclasses import replace

    diagnostic_source = replace(
        patch_source, logical_path="diagnostics_excluded_from_import/patch_3rows.csv"
    )

    with pytest.raises(OrchestrationError, match="diagnostics_excluded_from_import"):
        build_repair_overlay_request(
            freeze_root=ctx["freeze_root"], cell_id=ctx["cell_id"], condition_digest=ctx["condition_digest"],
            base_run_dir=ctx["base_run_dir"], base_realization_id=ctx["base"].realization_id,
            expected_dataset=ctx["expected_dataset"], patch_source=diagnostic_source,
            patch_question_reasons=question_reasons, authorized_question_ids=[repaired_id],
            dataset_snapshot_digest=ctx["expected_dataset"].snapshot_digest,
            cell_evidence_sha256=ctx["cell_source_sha256"],
            prediction_origins={repaired_id: "native_inference"},
            scope_disposition="included", run_id="repair-run", workspace_root=ctx["workspace_root"],
        )


# --- Offline-transformation overlay: full end-to-end ------------------------


def test_build_offline_transformation_overlay_request_executes_a_real_semantic_rematch(tmp_path):
    ctx = _historical_run(tmp_path)
    # arc_ids[2] (arc_q3) has an empty choice_d in the shared fixture, which
    # is irrelevant to this generic 3-row test -- skip it and use 3 other
    # questions so all 4 options are populated.
    rematch_ids = [ctx["arc_ids"][0], ctx["arc_ids"][1], ctx["arc_ids"][3]]

    # Component 1 produces the parsed choice deterministically, offline.
    free_text_by_question = {qid: "a1" for qid in rematch_ids}
    choices_by_question = {
        qid: {"A": "a1", "B": "b1", "C": "c1", "D": "d1"} for qid in rematch_ids
    }
    parsed = build_semantic_rematch_overlay_rows(
        free_text_responses=free_text_by_question, choices_by_question=choices_by_question
    )
    assert all(parsed[qid] == "A" for qid in rematch_ids)

    overlay_rows = [
        _result_row(qid, "A", ["a1", "b1", "c1", "d1"], parsed[qid]) for qid in rematch_ids
    ]
    import csv

    overlay_path = tmp_path / "semantic_overlay_3rows.csv"
    fieldnames = ["question_id", "correct_option", "choice_a", "choice_b", "choice_c", "choice_d", "parsed_choice"]
    with overlay_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overlay_rows)
    data = overlay_path.read_bytes()
    overlay_source = SourceArtifactSpec(
        source_id="semantic-overlay", path=overlay_path, logical_path="semantic_overlay_3rows.csv",
        expected_sha256=sha256(data).hexdigest(), format="csv",
        format_version=OFFLINE_TRANSFORMATION_PATCH_FORMAT_VERSION, classification="derived",
        dialect=CsvDialectSpec(), columns={"question_id": "question_id", "prediction": "parsed_choice"},
        expected_columns=tuple(fieldnames), ignored_columns={}, null_values=("",), numeric_columns=(),
        option_mapping=OptionMappingSpec(
            mode="ordered_columns", ordered_columns=("choice_a", "choice_b", "choice_c", "choice_d"),
            structured_column=None, structured_label_key=None, structured_text_key=None,
        ),
        extra_field_policy="preserve_unmapped", preserve_namespace="final_paper_analysis",
        source_run_id=None, source_repository="two-stage-prompting",
        source_commit="b8e784f3eb5d2a727a97eb675140b383a34584fa", notes={},
    )
    question_reasons = {qid: "option-hidden offline semantic rematch" for qid in rematch_ids}

    overlay_request = build_offline_transformation_overlay_request(
        freeze_root=ctx["freeze_root"], cell_id=ctx["cell_id"], condition_digest=ctx["condition_digest"],
        base_run_dir=ctx["base_run_dir"], base_realization_id=ctx["base"].realization_id,
        expected_dataset=ctx["expected_dataset"], overlay_source=overlay_source,
        question_reasons=question_reasons, authorized_question_ids=rematch_ids,
        dataset_snapshot_digest=ctx["expected_dataset"].snapshot_digest,
        cell_evidence_sha256=ctx["cell_source_sha256"], scope_disposition="included",
        run_id="offline-run", workspace_root=ctx["workspace_root"],
        source_containment_root=tmp_path,
    )
    from dataclasses import replace

    overlay_request = replace(
        overlay_request,
        overlay=replace(
            overlay_request.overlay,
            base_realization_digest=ctx["base"].realization_digest,
            base_validation_artifact_sha256=ctx["base"].validation_artifact_sha256,
            base_result_sha256=ctx["base"].result_sha256,
        ),
    )

    report = execute_overlay_import(overlay_request, dry_run=False)
    assert report.import_state == "imported", report.failures
    overlay_run_dir = ctx["workspace_root"] / "runs" / "offline-run"
    verified = verify_import_run(overlay_run_dir)
    (derived,) = verified.realizations.values()
    for qid in rematch_ids:
        assert derived.rows_by_question_id[qid]["predicted_option"] == "A"
        # offline_transformation retains the underlying response's own origin.
        assert derived.prediction_origins[qid] == "external_historical_inference"


def test_build_offline_transformation_overlay_request_rejects_correct_count_wrong_id_swap(tmp_path):
    """Fix A regression, offline_transformation side: correct row count (3)
    with one ID swapped for an unauthorized one must fail closed against the
    caller-supplied authorized set (in the real pipeline,
    semantic_rematch_manifest.csv), not whatever IDs the overlay CSV itself
    declares."""
    ctx = _historical_run(tmp_path)
    swapped_ids = [ctx["arc_ids"][0], ctx["arc_ids"][1], ctx["arc_ids"][3]]
    authorized_ids = [ctx["arc_ids"][0], ctx["arc_ids"][1], ctx["arc_ids"][2]]

    free_text_by_question = {qid: "a1" for qid in swapped_ids}
    choices_by_question = {
        qid: {"A": "a1", "B": "b1", "C": "c1", "D": "d1"} for qid in swapped_ids
    }
    parsed = build_semantic_rematch_overlay_rows(
        free_text_responses=free_text_by_question, choices_by_question=choices_by_question
    )
    overlay_rows = [
        _result_row(qid, "A", ["a1", "b1", "c1", "d1"], parsed[qid]) for qid in swapped_ids
    ]
    import csv

    overlay_path = tmp_path / "semantic_overlay_swapped.csv"
    fieldnames = ["question_id", "correct_option", "choice_a", "choice_b", "choice_c", "choice_d", "parsed_choice"]
    with overlay_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overlay_rows)
    data = overlay_path.read_bytes()
    overlay_source = SourceArtifactSpec(
        source_id="semantic-overlay-swapped", path=overlay_path, logical_path="semantic_overlay_swapped.csv",
        expected_sha256=sha256(data).hexdigest(), format="csv",
        format_version=OFFLINE_TRANSFORMATION_PATCH_FORMAT_VERSION, classification="derived",
        dialect=CsvDialectSpec(), columns={"question_id": "question_id", "prediction": "parsed_choice"},
        expected_columns=tuple(fieldnames), ignored_columns={}, null_values=("",), numeric_columns=(),
        option_mapping=OptionMappingSpec(
            mode="ordered_columns", ordered_columns=("choice_a", "choice_b", "choice_c", "choice_d"),
            structured_column=None, structured_label_key=None, structured_text_key=None,
        ),
        extra_field_policy="preserve_unmapped", preserve_namespace="final_paper_analysis",
        source_run_id=None, source_repository="two-stage-prompting",
        source_commit="b8e784f3eb5d2a727a97eb675140b383a34584fa", notes={},
    )
    question_reasons = {qid: "option-hidden offline semantic rematch" for qid in swapped_ids}

    with pytest.raises(OrchestrationError, match="unexpected question ID.*missing question ID"):
        build_offline_transformation_overlay_request(
            freeze_root=ctx["freeze_root"], cell_id=ctx["cell_id"], condition_digest=ctx["condition_digest"],
            base_run_dir=ctx["base_run_dir"], base_realization_id=ctx["base"].realization_id,
            expected_dataset=ctx["expected_dataset"], overlay_source=overlay_source,
            question_reasons=question_reasons, authorized_question_ids=authorized_ids,
            dataset_snapshot_digest=ctx["expected_dataset"].snapshot_digest,
            cell_evidence_sha256=ctx["cell_source_sha256"], scope_disposition="included",
            run_id="offline-run", workspace_root=ctx["workspace_root"],
            source_containment_root=tmp_path,
        )
