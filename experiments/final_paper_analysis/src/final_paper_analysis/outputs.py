"""Final outputs + claims ledger (spec section 10).

Deterministic writers for every required final-paper-analysis artifact.
This session performs no authoritative import (per its scope boundary), so
these writers are proven here against small synthetic tables; the
final-boss session wires them to the real per-cell results produced by
Components 3-9. Determinism means: given the same logical input, the same
bytes are written every time (sorted keys/columns/rows, no wall-clock or
process-order-dependent content) -- not necessarily byte-identical parquet
across different pyarrow versions, which is outside this code's control.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

REQUIRED_FLAT_OUTPUT_FILES = (
    "canonical_rows.parquet",
    "cell_metrics.csv",
    "accuracy_clopper_pearson.csv",
    "bias_metrics.csv",
    "failure_and_coverage_metrics.csv",
    "paired_method_deltas.csv",
    "paired_tests.csv",
    "two_by_two_factorial_effects.csv",
    "strategy_consistency_summary.csv",
    "sensitivity_analyses.csv",
    "flip_rate_metrics.csv",
    "analysis_manifest.json",
    "paper_claims_ledger.csv",
)
REQUIRED_OUTPUT_DIRECTORIES = ("final_paper_tables", "final_paper_figures")


class OutputsError(ValueError):
    """Raised when a final-output artifact is written with invalid or
    incomplete data."""


def write_canonical_rows(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Write canonical_rows.parquet, sorted by a fixed key so row order is
    deterministic regardless of upstream construction order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sort_columns = [c for c in ("question_id", "model_key", "backend", "method_key") if c in frame.columns]
    ordered = frame.sort_values(sort_columns).reset_index(drop=True) if sort_columns else frame
    path = output_dir / "canonical_rows.parquet"
    ordered.to_parquet(path, index=False)
    return path


def write_csv_table(rows: Sequence[Mapping[str, Any]], output_dir: Path, filename: str) -> Path:
    """Write a deterministic CSV: columns sorted, rows sorted by their own
    string representation (a fixed, reproducible tiebreak when the caller
    hasn't already imposed a meaningful order)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    if not frame.empty:
        frame = frame[sorted(frame.columns)]
        frame = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    path = output_dir / filename
    frame.to_csv(path, index=False)
    return path


def compute_sha256sums(output_dir: Path, *, exclude: frozenset[str] = frozenset({"SHA256SUMS"})) -> dict[str, str]:
    """{relative_posix_path: sha256_hex} for every file under output_dir,
    sorted by relative path."""
    output_dir = Path(output_dir)
    digests: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(output_dir).as_posix()
        if relative in exclude:
            continue
        digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(digests.items()))


def write_sha256sums(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    digests = compute_sha256sums(output_dir)
    lines = [f"{digest}  {relative_path}" for relative_path, digest in digests.items()]
    path = output_dir / "SHA256SUMS"
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


_REQUIRED_LEDGER_TEXT_FIELDS = (
    "source_file", "row_or_cell_identity", "analysis_function", "analysis_version",
    "metric_definition", "denominator", "ci_or_test_method",
)


@dataclass(frozen=True)
class ClaimsLedgerEntry:
    """Maps one publishable number to everything needed to audit it."""

    claim_id: str
    source_file: str
    row_or_cell_identity: str
    analysis_function: str
    analysis_version: str
    input_artifact_hashes: Mapping[str, str]
    metric_definition: str
    denominator: str
    ci_or_test_method: str
    value: float


def _validate_ledger_entry(entry: ClaimsLedgerEntry) -> None:
    if not entry.claim_id.strip():
        raise OutputsError("ClaimsLedgerEntry.claim_id must be non-empty.")
    for field in _REQUIRED_LEDGER_TEXT_FIELDS:
        value = getattr(entry, field)
        if not isinstance(value, str) or not value.strip():
            raise OutputsError(f"ClaimsLedgerEntry.{field} must be a non-empty string (claim_id={entry.claim_id!r}).")
    if not entry.input_artifact_hashes:
        raise OutputsError(
            f"ClaimsLedgerEntry.input_artifact_hashes must be non-empty (claim_id={entry.claim_id!r})."
        )


def write_claims_ledger(entries: Sequence[ClaimsLedgerEntry], output_dir: Path) -> Path:
    """Validate every entry (fails closed on any missing/blank required
    field -- a claim with no documented metric definition, denominator, or
    CI method must never be silently written) and write
    paper_claims_ledger.csv, sorted by claim_id for determinism. Fails
    closed on a duplicate claim_id."""
    seen_ids: set[str] = set()
    for entry in entries:
        _validate_ledger_entry(entry)
        if entry.claim_id in seen_ids:
            raise OutputsError(f"Duplicate claim_id in claims ledger: {entry.claim_id!r}.")
        seen_ids.add(entry.claim_id)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in sorted(entries, key=lambda e: e.claim_id):
        row = asdict(entry)
        row["input_artifact_hashes"] = json.dumps(dict(entry.input_artifact_hashes), sort_keys=True)
        rows.append(row)
    frame = pd.DataFrame(rows)
    path = output_dir / "paper_claims_ledger.csv"
    frame.to_csv(path, index=False)
    return path


def write_analysis_manifest(metadata: Mapping[str, Any], output_dir: Path) -> Path:
    """Write analysis_manifest.json with sorted keys -- ``metadata`` must
    not contain a wall-clock timestamp or other run-to-run-varying content
    the caller wants byte-reproducibility for; pass an explicit,
    caller-supplied version/seed record instead."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "analysis_manifest.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_all_required_outputs_present(output_dir: Path) -> None:
    """Fail closed if any required flat file or directory is missing from
    output_dir."""
    output_dir = Path(output_dir)
    missing_files = [name for name in REQUIRED_FLAT_OUTPUT_FILES if not (output_dir / name).is_file()]
    missing_dirs = [name for name in REQUIRED_OUTPUT_DIRECTORIES if not (output_dir / name).is_dir()]
    if missing_files or missing_dirs:
        raise OutputsError(
            f"Missing required output file(s) {missing_files} and/or directory(ies) {missing_dirs}."
        )
