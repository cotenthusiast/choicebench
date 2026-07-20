"""Tests for final outputs + claims ledger (spec section 10)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from final_paper_analysis.outputs import (
    ClaimsLedgerEntry,
    OutputsError,
    compute_sha256sums,
    verify_all_required_outputs_present,
    write_analysis_manifest,
    write_canonical_rows,
    write_claims_ledger,
    write_csv_table,
    write_sha256sums,
)


def _entry(**overrides) -> ClaimsLedgerEntry:
    defaults = dict(
        claim_id="acc_gpt41mini_arc_baseline",
        source_file="cell_metrics.csv",
        row_or_cell_identity="gpt-4-1-mini|arc_challenge|baseline",
        analysis_function="metrics.strict_accuracy@v1",
        analysis_version="1.0.0",
        input_artifact_hashes={"canonical_rows.parquet": "a" * 64},
        metric_definition="correct / total rows, missing counts as incorrect",
        denominator="all rows in the cell",
        ci_or_test_method="clopper_pearson_95",
        value=0.62,
    )
    defaults.update(overrides)
    return ClaimsLedgerEntry(**defaults)


def test_write_canonical_rows_sorts_deterministically(tmp_path):
    frame = pd.DataFrame({
        "question_id": ["q2", "q1", "q1"],
        "model_key": ["m", "m", "m"],
        "backend": ["api", "api", "api"],
        "method_key": ["baseline", "cyclic", "baseline"],
    })
    path1 = write_canonical_rows(frame, tmp_path / "out1")
    path2 = write_canonical_rows(frame.sample(frac=1, random_state=0), tmp_path / "out2")
    read1 = pd.read_parquet(path1)
    read2 = pd.read_parquet(path2)
    pd.testing.assert_frame_equal(read1, read2)


def test_write_csv_table_deterministic_column_and_row_order(tmp_path):
    rows = [{"b": 2, "a": 1}, {"b": 1, "a": 2}]
    path = write_csv_table(rows, tmp_path, "test_table.csv")
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["a", "b"]
    assert list(frame["a"]) == [1, 2]  # sorted by full-row content


def test_compute_sha256sums_matches_direct_hash(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("world", encoding="utf-8")
    digests = compute_sha256sums(tmp_path)
    import hashlib

    assert digests["a.txt"] == hashlib.sha256(b"hello").hexdigest()
    assert digests["sub/b.txt"] == hashlib.sha256(b"world").hexdigest()


def test_write_sha256sums_excludes_itself_and_matches_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    path = write_sha256sums(tmp_path)
    content = path.read_text(encoding="utf-8")
    assert "SHA256SUMS" not in content
    assert "a.txt" in content


def test_write_analysis_manifest_sorted_keys(tmp_path):
    path = write_analysis_manifest({"z": 1, "a": 2}, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"z"')
    assert json.loads(text) == {"z": 1, "a": 2}


# --- Claims ledger: fail-closed on missing fields ---------------------------


def test_write_claims_ledger_writes_a_valid_entry(tmp_path):
    path = write_claims_ledger([_entry()], tmp_path)
    frame = pd.read_csv(path)
    assert len(frame) == 1
    assert frame.iloc[0]["claim_id"] == "acc_gpt41mini_arc_baseline"
    assert frame.iloc[0]["value"] == pytest.approx(0.62)


@pytest.mark.parametrize("field", [
    "source_file", "row_or_cell_identity", "analysis_function", "analysis_version",
    "metric_definition", "denominator", "ci_or_test_method",
])
def test_write_claims_ledger_rejects_blank_required_field(tmp_path, field):
    entry = _entry(**{field: "   "})
    with pytest.raises(OutputsError, match=field):
        write_claims_ledger([entry], tmp_path)


def test_write_claims_ledger_rejects_empty_input_hashes(tmp_path):
    entry = _entry(input_artifact_hashes={})
    with pytest.raises(OutputsError, match="input_artifact_hashes"):
        write_claims_ledger([entry], tmp_path)


def test_write_claims_ledger_rejects_duplicate_claim_id(tmp_path):
    with pytest.raises(OutputsError, match="Duplicate claim_id"):
        write_claims_ledger([_entry(), _entry()], tmp_path)


def test_write_claims_ledger_rejects_blank_claim_id(tmp_path):
    with pytest.raises(OutputsError, match="claim_id"):
        write_claims_ledger([_entry(claim_id="  ")], tmp_path)


# --- Required-output completeness check -------------------------------------


def test_verify_all_required_outputs_present_fails_closed_when_missing(tmp_path):
    with pytest.raises(OutputsError, match="Missing required"):
        verify_all_required_outputs_present(tmp_path)


def test_verify_all_required_outputs_present_passes_when_complete(tmp_path):
    from final_paper_analysis.outputs import REQUIRED_FLAT_OUTPUT_FILES, REQUIRED_OUTPUT_DIRECTORIES

    for name in REQUIRED_FLAT_OUTPUT_FILES:
        (tmp_path / name).write_text("x", encoding="utf-8")
    for name in REQUIRED_OUTPUT_DIRECTORIES:
        (tmp_path / name).mkdir()
    verify_all_required_outputs_present(tmp_path)  # must not raise
