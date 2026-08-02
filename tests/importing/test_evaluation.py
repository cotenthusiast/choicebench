"""Tests for the reduced-scope v3 reader/evaluator (Task 13). Reuses
tests/importing/test_engine.py's ImportSpec fixture builder rather than
re-deriving a full spec here."""

from __future__ import annotations

import pytest

from choicebench.io.readers import (
    ResultSetError,
    build_evaluation_report_for_result_set,
    read_manifest_result_set,
)
from choicebench.importing.engine import ImportRequest, execute_import
from tests.importing.test_engine import _spec


def _imported_run(tmp_path, **spec_kwargs):
    spec = _spec(tmp_path, **spec_kwargs)
    request = ImportRequest(spec=spec, run_id="run-1", workspace_root=tmp_path, strict=True)
    execute_import(request, dry_run=False)
    return tmp_path / "runs" / "run-1"


def test_read_manifest_result_set_auto_selects_the_single_eligible_realization(tmp_path):
    run_dir = _imported_run(tmp_path)
    result_set = read_manifest_result_set(run_dir)
    assert result_set.selection.policy == "single_evaluable_per_condition"
    assert len(result_set.selection.realization_ids) == 1
    assert set(result_set.rows["question_id"].astype(str)) == {"q1", "q2"}


def test_read_manifest_result_set_rejects_unknown_realization_id(tmp_path):
    run_dir = _imported_run(tmp_path)
    with pytest.raises(ResultSetError, match="Unknown realization"):
        read_manifest_result_set(run_dir, realization_ids=("real_" + "0" * 16,))


def test_read_manifest_result_set_explicit_selection(tmp_path):
    run_dir = _imported_run(tmp_path)
    auto = read_manifest_result_set(run_dir)
    (realization_id,) = auto.selection.realization_ids
    explicit = read_manifest_result_set(run_dir, realization_ids=(realization_id,))
    assert explicit.selection.policy == "explicit"
    assert explicit.selection.realization_ids == (realization_id,)


def test_build_evaluation_report_accounts_for_every_realization(tmp_path):
    run_dir = _imported_run(tmp_path)
    result_set = read_manifest_result_set(run_dir)
    report = build_evaluation_report_for_result_set("run-1", result_set, reparse=False)
    assert report["schema_version"] == "choicebench.evaluation.v2"
    (condition_id,) = report["conditions"]
    (realization_id,) = report["conditions"][condition_id]["realization_ids"]
    entry = report["realizations"][realization_id]
    assert entry["selected"] is True
    assert entry["evidence_status"] == "complete"
    assert entry["scope_disposition"] == "included"
    assert entry["metrics"]["n"] == 2
    assert entry["metrics"]["accuracy"] == 1.0  # both rows' answer matches gold


def test_build_evaluation_report_scores_incorrect_predictions(tmp_path):
    run_dir = _imported_run(
        tmp_path,
        results_csv="qid,question,choice_a,choice_b,answer,gold,score\nq1,One?,x,y,b,a,0.9\nq2,Two?,m,n,b,b,0.7\n",
    )
    result_set = read_manifest_result_set(run_dir)
    report = build_evaluation_report_for_result_set("run-1", result_set, reparse=False)
    (realization_id,) = result_set.selection.realization_ids
    assert report["realizations"][realization_id]["metrics"]["accuracy"] == 0.5
