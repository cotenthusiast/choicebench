"""Translate the read-only Stage 1 paper-freeze into verified queue/status/
matrix accounting.

Reduced scope: this verifies checksums, parses the checksum-covered
canonical manifest and the four queue files, explodes them into
(cell_id, question_id) pairs, and reproduces the queue/status/matrix counts
the research objective needs to confirm (exact 138/687/3 approved/held/
excluded pairs, all pairwise disjoint; rerun_queue.csv's 776 rerun-candidate
pairs verified as a subset of the 828 classified pairs, not a fourth
authority tier -- rerun_queue.csv carries no queue_disposition/
execution_authority/executable columns at all, unlike the three
classification files; the 100-cell paper matrix breakdown; and the
6-cell/18-pair offline recoverable authority). It does NOT yet build
ImportSpec/AuthorizationSpec
objects ready to feed into the generic engine -- that requires the ARC/MMLU
ExpectedDataset trust chain (to validate authorization grants against real
selected question IDs, i.e. Task 16/Unit J) plus full per-cell
SourceArtifactSpec construction across the freeze's distinct method CSV
header schemas, neither of which is done here. translate_stage1_paper_freeze
is the verified foundation both of those build on: checksum-verified cell
records and verified, pairwise-disjoint, correctly-typed queue pairs.

This module is the only place in the importer allowed to know Stage 1
paper-specific facts (cell ID format, queue file names, method names). No
generic importer module imports from here.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import csv
import io
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from choicebench.importing.csv_adapter import OpenedSource
from choicebench.importing.dataset_reference import ExpectedDataset, build_expected_dataset
from choicebench.importing.schema import DatasetReferenceSpec

STATUS_MAP = {
    "canonical_complete": "complete",
    "canonical_qualified": "qualified",
    "recoverable_from_existing_artifacts": "recoverable",
    "incomplete_requires_inference": "partial",
    "malformed_requires_inference": "malformed",
}
_EXCLUDED_STATUS = "excluded_from_paper_matrix"
_PAPER_SCOPE_EXEMPT_METHOD = "pride"
_IHS_METHOD = "independent_hypothesis"

_VERIFIED_RELATIVE_PATHS = (
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


class Stage1ProfileError(ValueError):
    """Raised when the Stage 1 freeze fails checksum or consistency validation."""


def _load_checksum_ledger(freeze_root: Path) -> dict[str, str]:
    ledger: dict[str, str] = {}
    path = freeze_root / "checksums" / "checksums.sha256"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            digest, _, relative_path = line.partition("  ")
            ledger[relative_path] = digest
    return ledger


def _verify_checksummed_file(freeze_root: Path, ledger: Mapping[str, str], relative_path: str) -> bytes:
    path = freeze_root / relative_path
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise Stage1ProfileError(f"{relative_path} is unreadable: {exc}") from exc
    expected = ledger.get(relative_path)
    if expected is None:
        raise Stage1ProfileError(f"{relative_path} has no checksum ledger entry.")
    actual = sha256(data).hexdigest()
    if actual != expected:
        raise Stage1ProfileError(
            f"{relative_path} checksum mismatch: expected {expected}, found {actual}."
        )
    return data


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _explode_question_pairs(
    rows: list[dict[str, str]], *, source_label: str
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], str]]:
    pairs: dict[tuple[str, str], dict[str, str]] = {}
    reasons: dict[tuple[str, str], str] = {}
    for row in rows:
        cell_id = row["cell_id"]
        question_ids = json.loads(row["exact_question_ids"])
        question_reasons = json.loads(row["question_reasons"]) if row.get("question_reasons") else {}
        for question_id in question_ids:
            pair = (cell_id, question_id)
            if pair in pairs:
                raise Stage1ProfileError(
                    f"{source_label} declares duplicate (cell_id, question_id) pair {pair}."
                )
            pairs[pair] = row
            reasons[pair] = question_reasons.get(question_id, "")
    return pairs, reasons


def _require_all(rows: list[dict[str, str]], *, field: str, expected: str, source_label: str) -> None:
    bad = [row for row in rows if row.get(field) != expected]
    if bad:
        raise Stage1ProfileError(
            f"{source_label} has {len(bad)} row(s) with {field} != {expected!r}."
        )


def _require_all_bool(rows: list[dict[str, str]], *, field: str, expected: bool, source_label: str) -> None:
    expected_text = "true" if expected else "false"
    bad = [row for row in rows if row.get(field, "").strip().lower() != expected_text]
    if bad:
        raise Stage1ProfileError(
            f"{source_label} has {len(bad)} row(s) with {field} != {expected_text!r}."
        )


@dataclass(frozen=True)
class Stage1MatrixCounts:
    intended_cells: int
    core_method_cells: int
    local_pride_cells: int
    ihs_cells: int
    excluded_preserved_cells: int


@dataclass(frozen=True)
class Stage1QueueCounts:
    approved_executable_question_cells: int
    held_nonexecutable_question_cells: int
    excluded_nonexecutable_question_cells: int
    rerun_candidate_question_cells: int
    classified_question_cells: int
    pairwise_disjoint: bool
    approved_authoritative_executable: bool
    held_executable: bool
    excluded_executable: bool
    rerun_candidates_are_classified_subset: bool


@dataclass(frozen=True)
class Stage1QueuePairs:
    approved: Mapping[tuple[str, str], str]
    held: Mapping[tuple[str, str], str]
    excluded: Mapping[tuple[str, str], str]
    rerun_candidates: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class Stage1OfflineAuthority:
    condition_count: int
    question_cell_count: int
    recoverable_question_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    inference_executable: bool


@dataclass(frozen=True)
class Stage1Translation:
    cell_records: Mapping[str, Mapping[str, Any]]
    matrix: Stage1MatrixCounts
    queue_counts: Stage1QueueCounts
    queue_pairs: Stage1QueuePairs
    offline_authority: Stage1OfflineAuthority
    verified_relative_paths: tuple[str, ...]


def _matrix_counts(cell_records: Mapping[str, Mapping[str, Any]]) -> Stage1MatrixCounts:
    method_counts: dict[str, int] = {}
    excluded_cells = 0
    for record in cell_records.values():
        method_counts[record["method"]] = method_counts.get(record["method"], 0) + 1
        if record["final_status"] == _EXCLUDED_STATUS:
            excluded_cells += 1
    ihs_cells = method_counts.get(_IHS_METHOD, 0)
    pride_cells_total = method_counts.get(_PAPER_SCOPE_EXEMPT_METHOD, 0)
    local_pride_cells = pride_cells_total - excluded_cells
    core_method_cells = sum(
        count
        for method, count in method_counts.items()
        if method not in {_IHS_METHOD, _PAPER_SCOPE_EXEMPT_METHOD}
    )
    return Stage1MatrixCounts(
        intended_cells=core_method_cells + ihs_cells + local_pride_cells,
        core_method_cells=core_method_cells,
        local_pride_cells=local_pride_cells,
        ihs_cells=ihs_cells,
        excluded_preserved_cells=excluded_cells,
    )


def _offline_authority(cell_records: Mapping[str, Mapping[str, Any]]) -> Stage1OfflineAuthority:
    recoverable_cells = {
        cell_id: tuple(sorted(record["recoverable_question_ids"]))
        for cell_id, record in cell_records.items()
        if record["recoverable_question_ids"]
    }
    distinct_id_sets = set(recoverable_cells.values())
    if len(distinct_id_sets) != 1:
        raise Stage1ProfileError(
            "Recoverable cells do not all authorize the exact same question IDs: "
            f"{sorted(distinct_id_sets)}."
        )
    (recoverable_question_ids,) = distinct_id_sets
    cell_ids = tuple(sorted(recoverable_cells))
    return Stage1OfflineAuthority(
        condition_count=len(cell_ids),
        question_cell_count=len(cell_ids) * len(recoverable_question_ids),
        recoverable_question_ids=recoverable_question_ids,
        cell_ids=cell_ids,
        inference_executable=False,
    )


def translate_stage1_paper_freeze(freeze_root: Path) -> Stage1Translation:
    freeze_root = Path(freeze_root)
    ledger = _load_checksum_ledger(freeze_root)
    for relative_path in _VERIFIED_RELATIVE_PATHS:
        _verify_checksummed_file(freeze_root, ledger, relative_path)

    manifest_json = json.loads(
        (freeze_root / "manifests" / "canonical_results_manifest.json").read_text()
    )
    cell_records: dict[str, Mapping[str, Any]] = {}
    for record in manifest_json:
        cell_id = record["cell_id"]
        if cell_id in cell_records:
            raise Stage1ProfileError(f"Canonical manifest declares duplicate cell_id {cell_id!r}.")
        if record["status"] not in STATUS_MAP and record["final_status"] != _EXCLUDED_STATUS:
            raise Stage1ProfileError(
                f"Cell {cell_id!r} has an unrecognized status {record['status']!r}."
            )
        cell_records[cell_id] = record

    matrix = _matrix_counts(cell_records)
    offline_authority = _offline_authority(cell_records)

    approved_rows = _read_csv_rows(freeze_root / "manifests" / "approved_rerun_queue.csv")
    held_rows = _read_csv_rows(freeze_root / "manifests" / "held_or_declined_reruns.csv")
    excluded_rows = _read_csv_rows(freeze_root / "manifests" / "paper_scope_excluded_reruns.csv")
    # rerun_queue.csv carries no queue_disposition/execution_authority/executable
    # columns at all (unlike the three files above): it is the technical rerun
    # working-queue, not an authority classification. Verified below to be a
    # subset of the classified pairs, never treated as its own authority tier.
    rerun_candidate_rows = _read_csv_rows(freeze_root / "manifests" / "rerun_queue.csv")

    _require_all(approved_rows, field="queue_disposition", expected="approved", source_label="approved_rerun_queue.csv")
    _require_all(approved_rows, field="execution_authority", expected="authoritative", source_label="approved_rerun_queue.csv")
    _require_all_bool(approved_rows, field="executable", expected=True, source_label="approved_rerun_queue.csv")
    _require_all_bool(held_rows, field="executable", expected=False, source_label="held_or_declined_reruns.csv")
    _require_all_bool(excluded_rows, field="executable", expected=False, source_label="paper_scope_excluded_reruns.csv")

    approved_pairs, approved_reasons = _explode_question_pairs(approved_rows, source_label="approved_rerun_queue.csv")
    held_pairs, held_reasons = _explode_question_pairs(held_rows, source_label="held_or_declined_reruns.csv")
    excluded_pairs, excluded_reasons = _explode_question_pairs(excluded_rows, source_label="paper_scope_excluded_reruns.csv")
    rerun_candidate_pairs, _ = _explode_question_pairs(rerun_candidate_rows, source_label="rerun_queue.csv")

    classified = set(approved_pairs) | set(held_pairs) | set(excluded_pairs)
    total_classified = len(approved_pairs) + len(held_pairs) + len(excluded_pairs)
    if len(classified) != total_classified:
        raise Stage1ProfileError(
            "approved/held/excluded queues are not pairwise disjoint: "
            f"{total_classified} declared pairs but only {len(classified)} distinct."
        )
    unclassified_candidates = set(rerun_candidate_pairs) - classified
    if unclassified_candidates:
        raise Stage1ProfileError(
            f"{len(unclassified_candidates)} rerun-queue candidate pair(s) were never "
            "classified as approved, held, or excluded."
        )

    queue_counts = Stage1QueueCounts(
        approved_executable_question_cells=len(approved_pairs),
        held_nonexecutable_question_cells=len(held_pairs),
        excluded_nonexecutable_question_cells=len(excluded_pairs),
        rerun_candidate_question_cells=len(rerun_candidate_pairs),
        classified_question_cells=len(classified),
        pairwise_disjoint=True,
        approved_authoritative_executable=True,
        held_executable=False,
        excluded_executable=False,
        rerun_candidates_are_classified_subset=True,
    )
    queue_pairs = Stage1QueuePairs(
        approved=approved_reasons, held=held_reasons, excluded=excluded_reasons,
        rerun_candidates=frozenset(rerun_candidate_pairs),
    )

    return Stage1Translation(
        cell_records=cell_records,
        matrix=matrix,
        queue_counts=queue_counts,
        queue_pairs=queue_pairs,
        offline_authority=offline_authority,
        verified_relative_paths=_VERIFIED_RELATIVE_PATHS,
    )


# --- ARC/MMLU expected-dataset trust chain (Task 16) -----------------------
#
# Reduced scope, verified directly against the real freeze: this recomputes
# question_text/correct_option/correct_answer_text/choices from each raw row
# and compares them fieldwise against the archived normalized row at the
# SAME position -- raw and normalized files have identical row counts and
# the same row order, so a positional join is sufficient and avoids needing
# to reproduce the archived normalizer's own question_id hash algorithm.
# This is a ChoiceBench revalidation of internal freeze consistency, not
# proof that the archived normalizer (or the raw upstream publisher) was
# authentic -- recorded as a limitation on the returned ExpectedDataset.

_DATASET_RELATIVE_PATHS = {
    "arc_challenge": {
        "raw": "raw/local_model_generalization/data/raw/arc_challenge_raw.csv",
        "normalized": "raw/local_model_generalization/data/processed/arc_challenge_normalized.csv",
        "ids": "raw/local_model_generalization/data/splits/arc_challenge/robustness_ids.json",
        "metadata": "raw/local_model_generalization/data/splits/arc_challenge/robustness_metadata.json",
    },
    "mmlu": {
        "raw": "raw/local_model_generalization/data/raw/mmlu_raw.csv",
        "normalized": "raw/local_model_generalization/data/processed/mmlu_normalized.csv",
        "ids": "raw/local_model_generalization/data/splits/benchmark/robustness_ids.json",
        "metadata": "raw/local_model_generalization/data/splits/benchmark/robustness_metadata.json",
    },
}


_ARC_MAX_OPTIONS = 4


def _recompute_arc_fields(raw_row: Mapping[str, str]) -> dict[str, str]:
    """The archived normalizer caps ARC options at _ARC_MAX_OPTIONS, silently
    dropping any raw option beyond it -- verified against the real freeze:
    3 raw ARC questions have a 5th option, and all 3 have their answerKey
    within the first _ARC_MAX_OPTIONS, so truncation never drops the correct
    option. If a future/other freeze ever did drop the correct option, the
    index check below still fails closed rather than silently truncating it.
    """
    try:
        parsed = json.loads(raw_row["choices"])
        texts = list(parsed["text"])
        labels = list(parsed["label"])
        index = labels.index(raw_row["answerKey"])
    except (KeyError, ValueError, TypeError) as exc:
        raise Stage1ProfileError(
            f"ARC raw row {raw_row.get('id')!r} has malformed choices/answerKey: {exc}."
        ) from exc
    if index >= _ARC_MAX_OPTIONS:
        raise Stage1ProfileError(
            f"ARC raw row {raw_row.get('id')!r} answerKey index {index} falls outside the "
            f"archived normalizer's first {_ARC_MAX_OPTIONS} options; truncation would drop "
            "the correct option, so this cannot be safely revalidated."
        )
    texts = texts[:_ARC_MAX_OPTIONS]
    fields = {
        "question_text": raw_row["question"],
        "correct_option": chr(ord("A") + index),
        "correct_answer_text": texts[index],
    }
    for offset in range(_ARC_MAX_OPTIONS):
        fields[f"choice_{chr(ord('a') + offset)}"] = texts[offset] if offset < len(texts) else ""
    return fields


def _recompute_mmlu_fields(raw_row: Mapping[str, str]) -> dict[str, str]:
    try:
        choices = list(ast.literal_eval(raw_row["choices"]))
        index = int(raw_row["answer"])
    except (SyntaxError, ValueError, TypeError) as exc:
        raise Stage1ProfileError(
            f"MMLU raw row has malformed choices/answer: {exc}."
        ) from exc
    if not (0 <= index < len(choices)):
        raise Stage1ProfileError(
            f"MMLU raw row answer index {index} is out of range for choices {choices!r}."
        )
    fields = {
        "question_text": raw_row["question"],
        "correct_option": chr(ord("A") + index),
        "correct_answer_text": str(choices[index]),
    }
    for offset, text in enumerate(choices):
        fields[f"choice_{chr(ord('a') + offset)}"] = str(text)
    return fields


def _revalidate_positional(
    raw_rows: list[dict[str, str]],
    normalized_rows: list[dict[str, str]],
    *,
    recompute: Callable[[Mapping[str, str]], dict[str, str]],
    benchmark_name: str,
) -> None:
    if len(raw_rows) != len(normalized_rows):
        raise Stage1ProfileError(
            f"{benchmark_name}: raw ({len(raw_rows)}) and normalized ({len(normalized_rows)}) "
            "row counts disagree; this revalidation assumes positional 1:1 correspondence."
        )
    for index, (raw_row, normalized_row) in enumerate(zip(raw_rows, normalized_rows)):
        recomputed = recompute(raw_row)
        for field, expected_value in recomputed.items():
            actual_value = (normalized_row.get(field) or "").strip()
            if actual_value != expected_value.strip():
                raise Stage1ProfileError(
                    f"{benchmark_name} row {index} ({normalized_row.get('question_id')!r}): "
                    f"recomputed {field}={expected_value!r} does not match the archived "
                    f"normalized value {actual_value!r} (ChoiceBench revalidation failure)."
                )


def _drop_duplicate_questions(
    normalized_rows: list[dict[str, str]], *, benchmark_name: str
) -> list[dict[str, str]]:
    """pandas.DataFrame.drop_duplicates(subset="question_id", keep="first")
    semantics -- but only after requiring every duplicate occurrence to agree
    on all parsed fields; a disagreeing duplicate fails closed rather than
    silently picking one."""
    seen: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for row in normalized_rows:
        question_id = row["question_id"]
        if question_id in seen:
            if row != seen[question_id]:
                raise Stage1ProfileError(
                    f"{benchmark_name}: duplicate question_id {question_id!r} occurs with "
                    "disagreeing field values; cannot apply keep-first deduplication."
                )
            continue
        seen[question_id] = row
        order.append(question_id)
    return [seen[question_id] for question_id in order]


def _build_expected_dataset_for_benchmark(
    freeze_root: Path,
    ledger: Mapping[str, str],
    *,
    benchmark_name: str,
    raw_recompute: Callable[[Mapping[str, str]], dict[str, str]],
) -> ExpectedDataset:
    paths = _DATASET_RELATIVE_PATHS[benchmark_name]
    raw_bytes = _verify_checksummed_file(freeze_root, ledger, paths["raw"])
    normalized_bytes = _verify_checksummed_file(freeze_root, ledger, paths["normalized"])
    ids_bytes = _verify_checksummed_file(freeze_root, ledger, paths["ids"])
    metadata_bytes = _verify_checksummed_file(freeze_root, ledger, paths["metadata"])

    raw_rows = list(csv.DictReader(io.StringIO(raw_bytes.decode("utf-8"))))
    normalized_rows = list(csv.DictReader(io.StringIO(normalized_bytes.decode("utf-8"))))
    _revalidate_positional(
        raw_rows, normalized_rows, recompute=raw_recompute, benchmark_name=benchmark_name
    )

    deduplicated_rows = _drop_duplicate_questions(normalized_rows, benchmark_name=benchmark_name)
    selected_ids = json.loads(ids_bytes)
    metadata = json.loads(metadata_bytes)

    available_ids = {row["question_id"] for row in deduplicated_rows}
    missing = [question_id for question_id in selected_ids if question_id not in available_ids]
    if missing:
        raise Stage1ProfileError(
            f"{benchmark_name}: {len(missing)} selected question ID(s) are absent from the "
            f"normalized/deduplicated source, e.g. {missing[:3]}."
        )

    fieldnames = list(deduplicated_rows[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(deduplicated_rows)
    deduplicated_bytes = buffer.getvalue().encode("utf-8")
    opened_source = OpenedSource(
        source_id="normalized",
        audit_path=freeze_root / paths["normalized"],
        logical_path=paths["normalized"],
        data=deduplicated_bytes,
        sha256=sha256(deduplicated_bytes).hexdigest(),
    )

    choice_columns = {
        key: key for key in fieldnames if key.startswith("choice_") and len(key) == 8
    }
    declaration = DatasetReferenceSpec(
        dataset_id=benchmark_name,
        benchmark_name=benchmark_name,
        split="robustness",
        reference_kind="independent_input_snapshot",
        trust_label="checksum_verified_freeze_internal",
        source_ids=("normalized",),
        selection_source_id="normalized",
        expected_question_ids=tuple(selected_ids),
        selection_seed=metadata.get("seed"),
        selection_n_samples=metadata.get("actual_size"),
        subject_filter=(),
        selection_unknown_reasons={},
        columns={
            "question_id": "question_id",
            "question_text": "question_text",
            "correct_option": "correct_option",
            **choice_columns,
        },
        revision=None,
        fingerprint=None,
        derivation={"source_role": f"{benchmark_name}_stage1_freeze_normalized"},
        limitations=(
            "Stage 1 freeze: field content independently revalidated against the raw "
            "source by ChoiceBench; this proves internal freeze consistency, not that "
            "the archived normalizer or the upstream publisher was authentic.",
        ),
        native_compatibility_identity=None,
    )
    return build_expected_dataset(declaration, {"normalized": opened_source})


def build_stage1_expected_datasets(freeze_root: Path) -> dict[str, ExpectedDataset]:
    freeze_root = Path(freeze_root)
    ledger = _load_checksum_ledger(freeze_root)
    return {
        "arc_challenge": _build_expected_dataset_for_benchmark(
            freeze_root, ledger, benchmark_name="arc_challenge", raw_recompute=_recompute_arc_fields,
        ),
        "mmlu": _build_expected_dataset_for_benchmark(
            freeze_root, ledger, benchmark_name="mmlu", raw_recompute=_recompute_mmlu_fields,
        ),
    }
