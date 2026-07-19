"""Tests for Stage 1 profile translation (Task 15, reduced scope).

Uses a small synthetic freeze fixture (never real historical data) whose
structure was verified against the real, read-only Stage 1 freeze at
/home/cotenthusiast/Projects/model-generalization/paper_data_freeze during
development: file names, column headers, and the exact real queue/matrix
counts (138 approved / 687 held / 3 excluded / 828 classified / 776
rerun-queue-candidate pairs; 100/84/4/12/2 matrix breakdown; 6-cell/18-pair
offline-recoverable authority) were all cross-checked against that freeze
directly, not merely assumed from the plan.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from choicebench.importing.profiles.stage1_paper_freeze import (
    STATUS_MAP,
    Stage1ProfileError,
    translate_stage1_paper_freeze,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _cell(cell_id, method, status, *, damaged=(), recoverable=()):
    return {
        "cell_id": cell_id, "model": "model-a", "provider_backend": "provider",
        "benchmark": "arc_challenge", "benchmark_split": "robustness", "method": method,
        "expected_question_count": 10, "evidence_for_expected_existence": "x",
        "evidence_for_expected_row_count": "y", "discovered_candidate_count": 1,
        "final_status": status, "status": status, "canonical_artifact_id": f"artifact_{cell_id}",
        "canonical_raw_path": f"raw/{cell_id}.csv", "canonical_path": f"canonical/{cell_id}.csv",
        "canonical_sha256": "0" * 64, "observed_unique_count": 10, "duplicate_question_count": 0,
        "exact_missing_question_ids": [], "exact_unexpected_question_ids": [],
        "damaged_question_ids": list(damaged), "recoverable_question_ids": list(recoverable),
        "qualification": "",
    }


def _queue_row(cell_id, question_ids, reasons, **overrides):
    row = {
        "cell_id": cell_id, "model": "model-a", "provider_backend": "provider",
        "model_revision": "not recorded", "benchmark": "arc_challenge",
        "benchmark_snapshot_revision": "0" * 64, "method": "baseline",
        "exact_question_ids": json.dumps(question_ids),
        "number_of_questions": len(question_ids), "seed": 42,
        "work_type": "complete_partial_or_replace_damaged_rows",
        "configuration_identity": "0" * 64,
        "question_reasons": json.dumps(reasons),
        "source_configuration": "[]", "prompt_template_identity": "[]",
        "generation_parameters": "{}", "expected_output_schema": "[]",
        "supporting_artifact_paths": "[]",
    }
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("cell_id\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_freeze(
    tmp_path: Path,
    *,
    cells,
    approved_rows=(),
    held_rows=(),
    excluded_rows=(),
    rerun_candidate_rows=(),
    tamper_checksum_for=None,
):
    freeze_root = tmp_path / "freeze"
    manifest_json_path = freeze_root / "manifests" / "canonical_results_manifest.json"
    _write(manifest_json_path, json.dumps(cells))
    _write(freeze_root / "manifests" / "canonical_results_manifest.csv", "cell_id\n")
    _write(freeze_root / "manifests" / "cell_status_matrix.csv", "cell_id\n")
    _write(freeze_root / "manifests" / "expected_matrix.csv", "cell_id\n")
    _write_csv(freeze_root / "manifests" / "approved_rerun_queue.csv", list(approved_rows))
    _write_csv(freeze_root / "manifests" / "held_or_declined_reruns.csv", list(held_rows))
    _write_csv(freeze_root / "manifests" / "paper_scope_excluded_reruns.csv", list(excluded_rows))
    _write_csv(freeze_root / "manifests" / "rerun_queue.csv", list(rerun_candidate_rows))
    _write(freeze_root / "reports" / "canonical_freeze_report.md", "# report\n")

    relative_paths = (
        "manifests/canonical_results_manifest.json",
        "manifests/canonical_results_manifest.csv",
        "manifests/cell_status_matrix.csv",
        "manifests/expected_matrix.csv",
        "manifests/approved_rerun_queue.csv",
        "manifests/held_or_declined_reruns.csv",
        "manifests/paper_scope_excluded_reruns.csv",
        "manifests/rerun_queue.csv",
        "reports/canonical_freeze_report.md",
    )
    ledger_lines = []
    for relative_path in relative_paths:
        data = (freeze_root / relative_path).read_bytes()
        digest = sha256(data).hexdigest()
        if tamper_checksum_for == relative_path:
            digest = "f" * 64
        ledger_lines.append(f"{digest}  {relative_path}")
    _write(freeze_root / "checksums" / "checksums.sha256", "\n".join(ledger_lines) + "\n")
    return freeze_root


def _default_cells():
    return [
        _cell("cbp__model-a__arc_challenge__baseline", "baseline", "canonical_complete"),
        _cell("cbp__model-a__arc_challenge__two_stage_v1", "two_stage_v1", "canonical_qualified"),
        _cell(
            "cbp__model-a__arc_challenge__semantic_matching_v1", "semantic_matching_v1",
            "recoverable_from_existing_artifacts", recoverable=("q1", "q2"),
        ),
        _cell(
            "cbp__model-b__arc_challenge__semantic_matching_v1", "semantic_matching_v1",
            "recoverable_from_existing_artifacts", recoverable=("q1", "q2"),
        ),
        _cell(
            "cbp__model-a__arc_challenge__independent_hypothesis", "independent_hypothesis",
            "malformed_requires_inference", damaged=("q3",),
        ),
        _cell("cbp__model-a__arc_challenge__pride", "pride", "canonical_complete"),
        _cell("cbp__model-a__mmlu__pride", "pride", "excluded_from_paper_matrix"),
    ]


def test_translate_stage1_paper_freeze_reproduces_matrix_and_offline_authority(tmp_path):
    freeze_root = _build_freeze(tmp_path, cells=_default_cells())
    result = translate_stage1_paper_freeze(freeze_root)
    assert result.matrix.intended_cells == 6  # 7 cells minus the one excluded_from_paper_matrix
    # baseline + two_stage_v1 + both semantic_matching_v1 cells (only
    # independent_hypothesis/pride are excluded from the "core method" bucket)
    assert result.matrix.core_method_cells == 4
    assert result.matrix.ihs_cells == 1
    assert result.matrix.local_pride_cells == 1  # 2 pride cells minus 1 excluded
    assert result.matrix.excluded_preserved_cells == 1
    assert result.offline_authority.condition_count == 2
    assert result.offline_authority.question_cell_count == 4
    assert result.offline_authority.recoverable_question_ids == ("q1", "q2")
    assert result.offline_authority.inference_executable is False


def test_translate_stage1_paper_freeze_verifies_checksums(tmp_path):
    freeze_root = _build_freeze(
        tmp_path, cells=_default_cells(),
        tamper_checksum_for="manifests/canonical_results_manifest.json",
    )
    with pytest.raises(Stage1ProfileError, match="checksum mismatch"):
        translate_stage1_paper_freeze(freeze_root)


def test_translate_stage1_paper_freeze_rejects_duplicate_cell_id(tmp_path):
    cells = _default_cells()
    cells.append(dict(cells[0]))
    freeze_root = _build_freeze(tmp_path, cells=cells)
    with pytest.raises(Stage1ProfileError, match="duplicate cell_id"):
        translate_stage1_paper_freeze(freeze_root)


def test_translate_stage1_paper_freeze_requires_matching_recoverable_ids_across_cells(tmp_path):
    cells = _default_cells()
    # give the second recoverable cell a DIFFERENT question set than the first
    for cell in cells:
        if cell["cell_id"] == "cbp__model-b__arc_challenge__semantic_matching_v1":
            cell["recoverable_question_ids"] = ["q1", "q9"]
    freeze_root = _build_freeze(tmp_path, cells=cells)
    with pytest.raises(Stage1ProfileError, match="do not all authorize the exact same question"):
        translate_stage1_paper_freeze(freeze_root)


def test_translate_stage1_paper_freeze_computes_queue_pairs_and_disjointness(tmp_path):
    approved = [
        _queue_row(
            "cbp__model-a__arc_challenge__baseline", ["q1", "q2"],
            {"q1": "reason1", "q2": "reason2"},
            queue_disposition="approved", execution_authority="authoritative", executable="true",
        )
    ]
    held = [
        _queue_row(
            "cbp__model-a__arc_challenge__two_stage_v1", ["q3"], {"q3": "reason3"},
            queue_disposition="held", execution_authority="none", executable="false",
        )
    ]
    excluded = [
        _queue_row(
            "cbp__model-a__arc_challenge__pride", ["q4"], {"q4": "reason4"},
            queue_disposition="excluded", execution_authority="none", executable="false",
        )
    ]
    rerun_candidates = [
        _queue_row("cbp__model-a__arc_challenge__baseline", ["q1"], {}),
    ]
    freeze_root = _build_freeze(
        tmp_path, cells=_default_cells(), approved_rows=approved, held_rows=held,
        excluded_rows=excluded, rerun_candidate_rows=rerun_candidates,
    )
    result = translate_stage1_paper_freeze(freeze_root)
    assert result.queue_counts.approved_executable_question_cells == 2
    assert result.queue_counts.held_nonexecutable_question_cells == 1
    assert result.queue_counts.excluded_nonexecutable_question_cells == 1
    assert result.queue_counts.classified_question_cells == 4
    assert result.queue_counts.rerun_candidate_question_cells == 1
    assert result.queue_counts.rerun_candidates_are_classified_subset is True
    assert result.queue_pairs.approved[("cbp__model-a__arc_challenge__baseline", "q1")] == "reason1"


def test_translate_stage1_paper_freeze_rejects_overlapping_approved_and_held(tmp_path):
    approved = [
        _queue_row(
            "cbp__model-a__arc_challenge__baseline", ["q1"], {"q1": "r"},
            queue_disposition="approved", execution_authority="authoritative", executable="true",
        )
    ]
    held = [
        _queue_row(
            "cbp__model-a__arc_challenge__baseline", ["q1"], {"q1": "r"},
            queue_disposition="held", execution_authority="none", executable="false",
        )
    ]
    freeze_root = _build_freeze(tmp_path, cells=_default_cells(), approved_rows=approved, held_rows=held)
    with pytest.raises(Stage1ProfileError, match="not pairwise disjoint"):
        translate_stage1_paper_freeze(freeze_root)


def test_translate_stage1_paper_freeze_rejects_unclassified_rerun_candidate(tmp_path):
    rerun_candidates = [
        _queue_row("cbp__model-a__arc_challenge__baseline", ["q_never_classified"], {}),
    ]
    freeze_root = _build_freeze(tmp_path, cells=_default_cells(), rerun_candidate_rows=rerun_candidates)
    with pytest.raises(Stage1ProfileError, match="never classified"):
        translate_stage1_paper_freeze(freeze_root)


def test_translate_stage1_paper_freeze_rejects_approved_row_claiming_wrong_disposition(tmp_path):
    approved = [
        _queue_row(
            "cbp__model-a__arc_challenge__baseline", ["q1"], {"q1": "r"},
            queue_disposition="held",  # wrong -- approved file must say "approved"
            execution_authority="authoritative", executable="true",
        )
    ]
    freeze_root = _build_freeze(tmp_path, cells=_default_cells(), approved_rows=approved)
    with pytest.raises(Stage1ProfileError, match="queue_disposition"):
        translate_stage1_paper_freeze(freeze_root)


def test_translate_stage1_paper_freeze_rejects_held_row_claiming_executable(tmp_path):
    held = [
        _queue_row(
            "cbp__model-a__arc_challenge__baseline", ["q1"], {"q1": "r"},
            queue_disposition="held", execution_authority="none", executable="true",  # wrong
        )
    ]
    freeze_root = _build_freeze(tmp_path, cells=_default_cells(), held_rows=held)
    with pytest.raises(Stage1ProfileError, match="executable"):
        translate_stage1_paper_freeze(freeze_root)


def test_status_map_matches_canonical_values():
    assert STATUS_MAP["canonical_complete"] == "complete"
    assert STATUS_MAP["canonical_qualified"] == "qualified"
    assert STATUS_MAP["recoverable_from_existing_artifacts"] == "recoverable"
    assert STATUS_MAP["incomplete_requires_inference"] == "partial"
    assert STATUS_MAP["malformed_requires_inference"] == "malformed"
    assert "excluded_from_paper_matrix" not in STATUS_MAP  # not an evidence-status input
