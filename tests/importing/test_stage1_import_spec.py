"""Tests for full Stage 1 ImportSpec/AuthorizationSpec assembly (Task 18).

Uses a small synthetic freeze fixture combining test_stage1_profile.py's
cell-manifest/queue fixtures with its ARC/MMLU dataset fixtures, plus real
per-cell canonical result CSVs -- never real historical data. Column names,
the 6-column common schema (question_id/correct_option/parsed_choice/
choice_a..d), and the evidence-status reconciliation behavior below were all
verified directly against the real, read-only Stage 1 freeze at
/home/cotenthusiast/Projects/model-generalization/paper_data_freeze during
development (never committed): of 102 real cells, every one reconciles to
either "malformed" (any MISSING_PREDICTION/INVALID_PREDICTION/
CORRECT_OPTION_MISMATCH finding) or "complete"/"qualified" -- the freeze's
own STATUS_MAP-mapped status disagrees with the independently-computed
status for a majority of cells (e.g. 20 of 28 "malformed_requires_inference"
cells have zero row-level defects and reconcile to "complete"; 28 of 57
"canonical_complete" cells have some MISSING_PREDICTION rows the freeze
tolerates but ChoiceBench's generic validator does not, and reconcile to
"malformed"), confirming the reconciliation-by-probe design is load-bearing,
not a formality.
"""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path

import pytest

from choicebench.importing.authorization import validate_authorization_bundle
from choicebench.importing.evidence import open_verified_source
from choicebench.importing.profiles.stage1_paper_freeze import (
    Stage1ProfileError,
    build_stage1_cell_authorization,
    build_stage1_expected_datasets,
    build_stage1_import_spec,
    translate_stage1_paper_freeze,
)
from tests.importing.test_stage1_profile import (
    _build_dataset_freeze,
    _cell,
    _default_arc_normalized_rows,
    _default_mmlu_normalized_rows,
    _write,
)

_STAGE1_RESULT_COLUMNS = [
    "question_id", "correct_option", "choice_a", "choice_b", "choice_c", "choice_d",
    "parsed_choice",
]


def _result_row(question_id, correct_option, choices, parsed_choice):
    row = {"question_id": question_id, "correct_option": correct_option, "parsed_choice": parsed_choice}
    for letter, text in zip("abcd", choices):
        row[f"choice_{letter}"] = text
    return row


def _write_result_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_STAGE1_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _arc_result_rows(*, all_correct: bool = True):
    normalized = _default_arc_normalized_rows()
    rows = []
    for row in normalized:
        choices = [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]]
        parsed = row["correct_option"] if all_correct else "A"
        rows.append(_result_row(row["question_id"], row["correct_option"], choices, parsed))
    return rows


def _build_full_freeze(
    tmp_path: Path,
    *,
    cells,
    cell_result_rows: dict[str, list[dict]],
    approved_rows=(),
    held_rows=(),
    excluded_rows=(),
    rerun_candidate_rows=(),
):
    freeze_root = _build_dataset_freeze(tmp_path)

    manifest_json_path = freeze_root / "manifests" / "canonical_results_manifest.json"
    _write(manifest_json_path, json.dumps(cells))
    _write(freeze_root / "manifests" / "canonical_results_manifest.csv", "cell_id\n")
    _write(freeze_root / "manifests" / "cell_status_matrix.csv", "cell_id\n")
    _write(freeze_root / "manifests" / "expected_matrix.csv", "cell_id\n")

    def _write_queue_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("cell_id\n", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    _write_queue_csv(freeze_root / "manifests" / "approved_rerun_queue.csv", list(approved_rows))
    _write_queue_csv(freeze_root / "manifests" / "held_or_declined_reruns.csv", list(held_rows))
    _write_queue_csv(freeze_root / "manifests" / "paper_scope_excluded_reruns.csv", list(excluded_rows))
    _write_queue_csv(freeze_root / "manifests" / "rerun_queue.csv", list(rerun_candidate_rows))
    _write(freeze_root / "reports" / "canonical_freeze_report.md", "# report\n")

    for cell in cells:
        rows = cell_result_rows[cell["cell_id"]]
        _write_result_csv(freeze_root / cell["canonical_path"], rows)
        digest = sha256((freeze_root / cell["canonical_path"]).read_bytes()).hexdigest()
        cell["canonical_sha256"] = digest
    _write(manifest_json_path, json.dumps(cells))

    relative_paths = [
        "manifests/canonical_results_manifest.json",
        "manifests/canonical_results_manifest.csv",
        "manifests/cell_status_matrix.csv",
        "manifests/expected_matrix.csv",
        "manifests/approved_rerun_queue.csv",
        "manifests/held_or_declined_reruns.csv",
        "manifests/paper_scope_excluded_reruns.csv",
        "manifests/rerun_queue.csv",
        "reports/canonical_freeze_report.md",
        *[cell["canonical_path"] for cell in cells],
    ]
    ledger_path = freeze_root / "checksums" / "checksums.sha256"
    existing_lines = ledger_path.read_text(encoding="utf-8").splitlines()
    all_lines = list(existing_lines)
    for relative_path in relative_paths:
        data = (freeze_root / relative_path).read_bytes()
        digest = sha256(data).hexdigest()
        all_lines.append(f"{digest}  {relative_path}")
    _write(ledger_path, "\n".join(all_lines) + "\n")
    return freeze_root


def _default_cells_and_rows():
    arc_ids = [row["question_id"] for row in _default_arc_normalized_rows()]
    clean_cell_id = "cbp__model-a__arc_challenge__baseline"
    recoverable_cell_id = "cbp__model-a__arc_challenge__semantic_matching_v1"
    malformed_cell_id = "cbp__model-a__arc_challenge__two_stage_v1"

    cells = [
        _cell(clean_cell_id, "baseline", "canonical_complete"),
        _cell(
            recoverable_cell_id, "semantic_matching_v1", "recoverable_from_existing_artifacts",
            recoverable=(arc_ids[0],),
        ),
        _cell(malformed_cell_id, "two_stage_v1", "malformed_requires_inference"),
    ]
    for cell in cells:
        cell["benchmark"] = "arc_challenge"
        cell["expected_question_count"] = len(arc_ids)

    rows = {
        clean_cell_id: _arc_result_rows(all_correct=True),
        # recoverable: rows are all present/parseable (just semantically
        # "wrong" per the freeze's own domain knowledge) -- reconciles to
        # "complete", matching the real freeze's 6 recoverable cells.
        recoverable_cell_id: _arc_result_rows(all_correct=False),
        malformed_cell_id: [
            *_arc_result_rows(all_correct=True)[:-1],
            _result_row(arc_ids[-1], "B", ["a4", "b4", "c4", "d4"], ""),  # missing prediction
        ],
    }
    return cells, rows, arc_ids


def test_build_stage1_import_spec_assembles_sources_conditions_and_children(tmp_path):
    cells, rows, arc_ids = _default_cells_and_rows()
    freeze_root = _build_full_freeze(tmp_path, cells=cells, cell_result_rows=rows)
    translation = translate_stage1_paper_freeze(freeze_root)
    expected_datasets = build_stage1_expected_datasets(freeze_root)

    spec = build_stage1_import_spec(freeze_root, translation, expected_datasets)
    assert len(spec.conditions) == 3
    assert len(spec.sources) == 3 + 2  # 3 cells + arc/mmlu normalized reference sources
    assert {model.model_key for model in spec.models} == {"model-a"}
    assert {method.method_key for method in spec.methods} == {
        "baseline", "semantic_matching_v1", "two_stage_v1",
    }
    assert len(spec.prompts) == 1


def test_build_stage1_import_spec_reconciles_recoverable_cell_to_complete(tmp_path):
    """The freeze declares this cell 'recoverable_from_existing_artifacts', but
    every row is present with a valid (if semantically wrong) parsed_choice --
    ChoiceBench's generic validator independently computes 'complete', and
    build_stage1_import_spec must declare that, not the freeze's own label,
    or the real import would fail its own declaration-mismatch check."""
    cells, rows, arc_ids = _default_cells_and_rows()
    freeze_root = _build_full_freeze(tmp_path, cells=cells, cell_result_rows=rows)
    translation = translate_stage1_paper_freeze(freeze_root)
    expected_datasets = build_stage1_expected_datasets(freeze_root)

    spec = build_stage1_import_spec(freeze_root, translation, expected_datasets)
    by_key = {c.condition_key: c for c in spec.conditions}
    recoverable = by_key["cbp__model-a__arc_challenge__semantic_matching_v1"]
    assert recoverable.evidence_status == "complete"
    assert recoverable.recoverable_question_ids == (arc_ids[0],)


def test_build_stage1_import_spec_reconciles_malformed_cell_with_missing_prediction(tmp_path):
    cells, rows, arc_ids = _default_cells_and_rows()
    freeze_root = _build_full_freeze(tmp_path, cells=cells, cell_result_rows=rows)
    translation = translate_stage1_paper_freeze(freeze_root)
    expected_datasets = build_stage1_expected_datasets(freeze_root)

    spec = build_stage1_import_spec(freeze_root, translation, expected_datasets)
    by_key = {c.condition_key: c for c in spec.conditions}
    malformed = by_key["cbp__model-a__arc_challenge__two_stage_v1"]
    assert malformed.evidence_status == "malformed"


def test_build_stage1_import_spec_keeps_a_clean_complete_cell_complete(tmp_path):
    cells, rows, arc_ids = _default_cells_and_rows()
    freeze_root = _build_full_freeze(tmp_path, cells=cells, cell_result_rows=rows)
    translation = translate_stage1_paper_freeze(freeze_root)
    expected_datasets = build_stage1_expected_datasets(freeze_root)

    spec = build_stage1_import_spec(freeze_root, translation, expected_datasets)
    by_key = {c.condition_key: c for c in spec.conditions}
    clean = by_key["cbp__model-a__arc_challenge__baseline"]
    assert clean.evidence_status == "complete"
    assert clean.scope_disposition == "included"


def test_build_stage1_import_spec_marks_excluded_from_paper_matrix(tmp_path):
    cells, rows, arc_ids = _default_cells_and_rows()
    excluded_cell_id = "cbp__model-a__arc_challenge__pride"
    excluded_cell = _cell(excluded_cell_id, "pride", "excluded_from_paper_matrix")
    excluded_cell["benchmark"] = "arc_challenge"
    excluded_cell["expected_question_count"] = len(arc_ids)
    cells = [*cells, excluded_cell]
    rows[excluded_cell_id] = _arc_result_rows(all_correct=True)

    freeze_root = _build_full_freeze(tmp_path, cells=cells, cell_result_rows=rows)
    translation = translate_stage1_paper_freeze(freeze_root)
    expected_datasets = build_stage1_expected_datasets(freeze_root)

    spec = build_stage1_import_spec(freeze_root, translation, expected_datasets)
    by_key = {c.condition_key: c for c in spec.conditions}
    excluded = by_key[excluded_cell_id]
    assert excluded.scope_disposition == "excluded_from_paper_matrix"
    assert excluded.evidence_status == "complete"


def test_build_stage1_import_spec_rejects_canonical_sha256_ledger_disagreement(tmp_path):
    cells, rows, arc_ids = _default_cells_and_rows()
    freeze_root = _build_full_freeze(tmp_path, cells=cells, cell_result_rows=rows)
    expected_datasets = build_stage1_expected_datasets(freeze_root)

    # Tamper the manifest's own recorded canonical_sha256 for one cell (a
    # forged claim about that cell's own result file), while re-checksumming
    # the manifest.json file itself so translate_stage1_paper_freeze's own
    # whole-file checksum check still passes -- isolating the per-cell
    # cross-check build_stage1_import_spec performs against the untouched
    # ledger entry for that cell's actual canonical CSV.
    manifest_path = freeze_root / "manifests" / "canonical_results_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[0]["canonical_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ledger_path = freeze_root / "checksums" / "checksums.sha256"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    new_manifest_digest = sha256(manifest_path.read_bytes()).hexdigest()
    updated_lines = [
        f"{new_manifest_digest}  manifests/canonical_results_manifest.json"
        if line.endswith("manifests/canonical_results_manifest.json") else line
        for line in lines
    ]
    ledger_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    translation = translate_stage1_paper_freeze(freeze_root)
    with pytest.raises(Stage1ProfileError, match="disagrees with the checksum ledger"):
        build_stage1_import_spec(freeze_root, translation, expected_datasets)


def test_build_stage1_cell_authorization_produces_a_bundle_the_generic_validator_accepts(tmp_path):
    cells, rows, arc_ids = _default_cells_and_rows()
    freeze_root = _build_full_freeze(tmp_path, cells=cells, cell_result_rows=rows)
    translation = translate_stage1_paper_freeze(freeze_root)
    expected_datasets = build_stage1_expected_datasets(freeze_root)
    from choicebench.importing.profiles.stage1_paper_freeze import _load_checksum_ledger

    ledger = _load_checksum_ledger(freeze_root)
    spec = build_stage1_import_spec(freeze_root, translation, expected_datasets)
    by_key = {c.condition_key: c for c in spec.conditions}
    cell_id = "cbp__model-a__arc_challenge__semantic_matching_v1"
    condition = by_key[cell_id]
    dataset = expected_datasets["arc_challenge"]

    from choicebench.importing.identity import build_import_semantic_identity

    models_by_key = {m.model_key: m for m in spec.models}
    methods_by_key = {m.method_key: m for m in spec.methods}
    prompts_by_key = {p.prompt_key: p for p in spec.prompts}
    semantic = build_import_semantic_identity(
        condition=condition, dataset=dataset,
        model=models_by_key[condition.model_key], method=methods_by_key[condition.method_key],
        prompt=prompts_by_key[condition.prompt_key],
    )
    condition_digest = semantic.condition["condition_digest"]

    sources_by_id = {s.source_id: s for s in spec.sources}
    cell_source = sources_by_id[cell_id]

    authorization, auth_source = build_stage1_cell_authorization(
        freeze_root, ledger,
        cell_id=cell_id, condition_digest=condition_digest,
        authorization_type="offline_transformation",
        question_reasons={arc_ids[0]: "option-hidden semantic rematch"},
        dataset_snapshot_digest=dataset.snapshot_digest,
        cell_evidence_sha256=cell_source.expected_sha256,
    )
    opened = open_verified_source(auth_source, containment_root=freeze_root)
    bundle = validate_authorization_bundle(
        authorization, opened_source=opened,
        condition_digests={cell_id: condition_digest},
        expected={condition_digest: dataset},
    )
    assert bundle.authorization_type == "offline_transformation"
    assert bundle.grants[condition_digest] == {arc_ids[0]: "option-hidden semantic rematch"}
    assert bundle.executable is False
