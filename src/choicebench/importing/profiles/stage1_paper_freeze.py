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
from dataclasses import dataclass, replace
from hashlib import sha256
import csv
import io
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from choicebench.identity import short_id
from choicebench.importing.csv_adapter import OpenedSource, parse_csv_source
from choicebench.importing.dataset_reference import ExpectedDataset, build_expected_dataset
from choicebench.importing.evidence import open_verified_source
from choicebench.importing.validation import compute_evidence_status, validate_source_rows
from choicebench.importing.schema import (
    AuthorizationSpec,
    CsvDialectSpec,
    DatasetReferenceSpec,
    ImportConditionSpec,
    ImportMethodSpec,
    ImportModelSpec,
    ImportPromptSpec,
    ImportSpec,
    OptionMappingSpec,
    ResultOriginSpec,
    SourceArtifactSpec,
)

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


# --- Full ImportSpec assembly (Task 16/Unit K remainder) --------------------
#
# Reduced scope, verified directly against the real freeze: all 102 cells
# have observed_unique_count==1000 with zero missing/unexpected/duplicate
# question IDs, so compute_evidence_status's "partial"/"recoverable"
# branches (which need missing_ids) never trigger here -- every cell
# reconciles to either "malformed" or "complete"/"qualified". The freeze's
# own STATUS_MAP-mapped status is a scientific/methodology classification,
# not always what ChoiceBench's generic row validator independently
# computes: the 6 "recoverable" cells and the "mmlu independent_hypothesis"
# "incomplete" cell have ZERO row-level defects (their defect -- option-
# hiding, or a protocol-invalid option-A score after MAX_TOKENS termination
# -- is invisible to generic validation, documented only via
# damaged_question_ids/qualification narrative). So every condition's
# declared evidence_status is reconciled via an actual probe
# (validate_source_rows + compute_evidence_status), never trusted blindly
# from STATUS_MAP. qualifications=({"reason": ...},) is pre-populated
# whenever record["qualification"] is non-empty, so a cell with a
# documented caveat but zero generic defects reconciles to "qualified"
# rather than bare "complete".
#
# question_id/correct_option/prediction are the only source->identity
# column mappings (question_text is deliberately NOT mapped), avoiding
# noise from formatting differences between each run's own recorded
# question_text and Unit J's normalized ExpectedDataset text --
# correct_option is the meaningful invariant (a categorical answer key,
# not prone to whitespace/formatting variance).
#
# Real queue reconciliation (verified directly against the freeze): 102
# cells = 62 needing no repair (57 complete + 5 qualified) + 6 recoverable
# (offline authority, 18 pairs, all arc_challenge/semantic_matching_v1) +
# 32 malformed/incomplete, of which 31 cells are fully approved (138 pairs
# total) and exactly 1 cell
# (cbp__gemini-2-5-flash__arc_challenge__independent_hypothesis) is fully
# HELD (687 questions, supervisor-stopped) + 2 excluded_from_paper_matrix
# cells (both "pride"/qwen-2.5-7b-instruct-turbo), one of which also
# carries 3 explicitly-excluded damaged questions in
# paper_scope_excluded_reruns.csv. Held and excluded pairs must NOT
# receive any authorization/overlay.

_STAGE1_COLUMNS = {
    "question_id": "question_id",
    "correct_option": "correct_option",
    "prediction": "parsed_choice",
}
_STAGE1_OPTION_COLUMNS = ("choice_a", "choice_b", "choice_c", "choice_d")
_STAGE1_PROMPT_KEY = "stage1_freeze_v1"


def _stage1_prompt() -> ImportPromptSpec:
    return ImportPromptSpec(
        prompt_key=_STAGE1_PROMPT_KEY,
        template_identity=None,
        template_digest=None,
        template_contents=None,
        unknown_reason=(
            "Stage 1 freeze: each canonical CSV row preserves its own literal rendered "
            "'prompt' text, but no single stable prompt-template identity/digest was "
            "recorded across methods at the paper's authoring time; method_key already "
            "captures the distinct prompting strategy."
        ),
        native_compatibility_identity=None,
    )


def _stage1_model(model_name: str, provider_backend: str) -> ImportModelSpec:
    return ImportModelSpec(
        model_key=model_name,
        display_name=model_name,
        backend=provider_backend,
        provider=provider_backend,
        revision=None,
        effective_parameters={},
        unknown_reasons={"revision": "not recorded by paper_data_freeze canonical manifest"},
        native_compatibility_identity=None,
    )


def _stage1_method(method_name: str) -> ImportMethodSpec:
    return ImportMethodSpec(
        method_key=method_name,
        name=method_name,
        effective_parameters={},
        implementation=None,
        unknown_reasons={"implementation": "not recorded by paper_data_freeze canonical manifest"},
        native_compatibility_identity=None,
    )


def _stage1_cell_source(
    freeze_root: Path, ledger: Mapping[str, str], cell_id: str, record: Mapping[str, Any]
) -> SourceArtifactSpec:
    relative_path = record["canonical_path"]
    ledger_digest = ledger.get(relative_path)
    if ledger_digest is None:
        raise Stage1ProfileError(f"{relative_path} has no checksum ledger entry.")
    if ledger_digest != record["canonical_sha256"]:
        raise Stage1ProfileError(
            f"Cell {cell_id!r} canonical_sha256 disagrees with the checksum ledger for "
            f"{relative_path}."
        )
    path = freeze_root / relative_path
    with path.open(newline="", encoding="utf-8") as handle:
        header = tuple(next(csv.reader(handle)))
    required = set(_STAGE1_COLUMNS.values()) | set(_STAGE1_OPTION_COLUMNS)
    missing = required - set(header)
    if missing:
        raise Stage1ProfileError(
            f"Cell {cell_id!r} canonical CSV is missing column(s) {sorted(missing)}."
        )
    return SourceArtifactSpec(
        source_id=cell_id,
        path=path,
        logical_path=relative_path,
        expected_sha256=record["canonical_sha256"],
        format="csv",
        format_version="paper_data_freeze.v1",
        classification="raw",
        dialect=CsvDialectSpec(),
        columns=dict(_STAGE1_COLUMNS),
        expected_columns=header,
        ignored_columns={},
        null_values=("",),
        numeric_columns=(),
        option_mapping=OptionMappingSpec(
            mode="ordered_columns",
            ordered_columns=_STAGE1_OPTION_COLUMNS,
            structured_column=None,
            structured_label_key=None,
            structured_text_key=None,
        ),
        extra_field_policy="preserve_unmapped",
        preserve_namespace="stage1_paper_freeze",
        source_run_id=None,
        source_repository=None,
        source_commit=None,
        notes={
            "cell_id": cell_id,
            "declared_status": record["status"],
            "final_status": record["final_status"],
            "qualification": record.get("qualification") or "",
        },
    )


def _stage1_normalized_dataset_source(
    freeze_root: Path, ledger: Mapping[str, str], *, benchmark_name: str, source_id: str
) -> SourceArtifactSpec:
    """Documentation-only source declaration for the ARC/MMLU normalized
    reference file: real path/checksum, but never opened by the generic
    engine, since ImportRequest.expected_datasets overrides the per-source
    ExpectedDataset rebuild with build_stage1_expected_datasets' own
    revalidation/dedup chain (needed for MMLU's real duplicate question_ids,
    which the generic rebuild would reject)."""
    relative_path = _DATASET_RELATIVE_PATHS[benchmark_name]["normalized"]
    expected_sha256 = ledger.get(relative_path)
    if expected_sha256 is None:
        raise Stage1ProfileError(f"{relative_path} has no checksum ledger entry.")
    return SourceArtifactSpec(
        source_id=source_id,
        path=freeze_root / relative_path,
        logical_path=relative_path,
        expected_sha256=expected_sha256,
        format="csv",
        format_version="paper_data_freeze.v1",
        classification="canonical",
        dialect=CsvDialectSpec(),
        columns={},
        expected_columns=(),
        ignored_columns={},
        null_values=(),
        numeric_columns=(),
        option_mapping=OptionMappingSpec(
            mode="ordered_columns", ordered_columns=(), structured_column=None,
            structured_label_key=None, structured_text_key=None,
        ),
        extra_field_policy="preserve_unmapped",
        preserve_namespace="stage1_paper_freeze",
        source_run_id=None, source_repository=None, source_commit=None,
        notes={"role": f"{benchmark_name}_expected_dataset_reference"},
    )


def _reconcile_evidence_status(
    condition: ImportConditionSpec,
    *,
    source: SourceArtifactSpec,
    freeze_root: Path,
    dataset: ExpectedDataset,
) -> ImportConditionSpec:
    opened = open_verified_source(source, containment_root=freeze_root)
    table = parse_csv_source(opened, source, strict=True)
    validated = validate_source_rows(
        table, source_id=source.source_id, mapping=source.columns, condition=condition, expected=dataset
    )
    computed, _ = compute_evidence_status(validated, condition=condition)
    if computed != condition.evidence_status:
        condition = replace(condition, evidence_status=computed)
    return condition


def build_stage1_import_spec(
    freeze_root: Path,
    translation: Stage1Translation,
    expected_datasets: Mapping[str, ExpectedDataset],
) -> ImportSpec:
    freeze_root = Path(freeze_root)
    ledger = _load_checksum_ledger(freeze_root)

    dataset_source_ids = {
        "arc_challenge": "arc_challenge_normalized_reference",
        "mmlu": "mmlu_normalized_reference",
    }
    sources: list[SourceArtifactSpec] = [
        _stage1_normalized_dataset_source(
            freeze_root, ledger, benchmark_name=benchmark_name, source_id=source_id
        )
        for benchmark_name, source_id in dataset_source_ids.items()
    ]

    conditions: list[ImportConditionSpec] = []
    models_by_key: dict[str, ImportModelSpec] = {}
    methods_by_key: dict[str, ImportMethodSpec] = {}
    prompt = _stage1_prompt()

    for cell_id in sorted(translation.cell_records):
        record = translation.cell_records[cell_id]
        source = _stage1_cell_source(freeze_root, ledger, cell_id, record)
        sources.append(source)

        model_name = record["model"]
        if model_name not in models_by_key:
            models_by_key[model_name] = _stage1_model(model_name, record["provider_backend"])
        method_name = record["method"]
        if method_name not in methods_by_key:
            methods_by_key[method_name] = _stage1_method(method_name)

        benchmark = record["benchmark"]
        dataset = expected_datasets[benchmark]
        scope_disposition = (
            "excluded_from_paper_matrix" if record["final_status"] == _EXCLUDED_STATUS else "included"
        )
        qualification_text = record.get("qualification") or ""
        qualifications = ({"reason": qualification_text},) if qualification_text else ()

        condition = ImportConditionSpec(
            condition_key=cell_id,
            source_ids=(cell_id,),
            dataset_id=benchmark,
            model_key=model_name,
            method_key=method_name,
            prompt_key=prompt.prompt_key,
            seed=42,
            calibration_identity=None,
            preflight_identity=None,
            protocol_settings={},
            generation_parameters={"temperature": 0.0},
            unknown_reasons={
                "calibration_identity": "not recorded by paper_data_freeze canonical manifest",
                "preflight_identity": "not recorded by paper_data_freeze canonical manifest",
            },
            expected_question_ids=dataset.selected_question_ids,
            evidence_status=STATUS_MAP.get(record["status"], "complete"),
            scope_disposition=scope_disposition,
            executable=None,
            qualifications=qualifications,
            limitations=(),
            damaged_question_ids=tuple(record["damaged_question_ids"]),
            recoverable_question_ids=tuple(record["recoverable_question_ids"]),
            result_origin=ResultOriginSpec(
                derivation_origin="external_import",
                default_prediction_origin="external_historical_inference",
                per_question_prediction_origins={},
            ),
        )
        condition = _reconcile_evidence_status(
            condition, source=source, freeze_root=freeze_root, dataset=dataset
        )
        conditions.append(condition)

    datasets = tuple(
        DatasetReferenceSpec(
            dataset_id=benchmark_name,
            benchmark_name=benchmark_name,
            split="robustness",
            reference_kind="independent_input_snapshot",
            trust_label="checksum_verified_freeze_internal",
            source_ids=(dataset_source_ids[benchmark_name],),
            selection_source_id=dataset_source_ids[benchmark_name],
            expected_question_ids=expected_datasets[benchmark_name].selected_question_ids,
            selection_seed=None,
            selection_n_samples=None,
            subject_filter=(),
            selection_unknown_reasons={},
            columns={},
            revision=None,
            fingerprint=None,
            derivation={},
            limitations=(),
            native_compatibility_identity=None,
        )
        for benchmark_name in ("arc_challenge", "mmlu")
    )
    # This DatasetReferenceSpec is documentation-only -- the real
    # ExpectedDataset comes from ImportRequest.expected_datasets, built by
    # build_stage1_expected_datasets' own revalidation/dedup chain, which
    # build_import_plan's generic per-source rebuild cannot reproduce for
    # MMLU's real duplicate question_ids (build_import_plan skips its own
    # per-declaration rebuild entirely whenever expected_datasets is
    # supplied, so this declaration's source_ids are never actually opened).

    return ImportSpec(
        schema_version="choicebench.import-spec.v1",
        import_name="Stage 1 paper data freeze",
        sources=tuple(sources),
        datasets=datasets,
        models=tuple(models_by_key.values()),
        methods=tuple(methods_by_key.values()),
        prompts=(prompt,),
        conditions=tuple(conditions),
        authorizations=(),
        overlays=(),
        metrics=["accuracy"],
        provenance={"producer_request_id": {"value": None, "reason": "paper_data_freeze import"}},
        audit={"source_location": str(freeze_root)},
    )


# --- Per-cell authorization bundle -------------------------------------
#
# IMPORTANT, verified by reading derive_overlay's own cross-check
# (overlays.py): a ValidatedAuthorization's input_evidence_digests is the
# WHOLE bundle's input_evidence_digests, unsliced by
# authorization_for_condition -- and derive_overlay requires
# overlay.base_evidence_digests to match it key-for-key AND every one of
# those keys to also appear in the base realization's OWN
# evidence_source_digests. Since every Stage 1 cell has exactly one
# source, a bundle whose input_evidence_digests spans more than one cell
# can never satisfy that check for any single cell's overlay. So each
# AuthorizationSpec here is deliberately scoped to exactly ONE cell/
# condition (one grants key, one evidence-digest key) -- NOT one bundle
# for all 31 approved cells or all 6 recoverable cells at once.

def build_stage1_cell_authorization(
    freeze_root: Path,
    ledger: Mapping[str, str],
    *,
    cell_id: str,
    condition_digest: str,
    authorization_type: str,
    question_reasons: Mapping[str, str],
    dataset_snapshot_digest: str,
    cell_evidence_sha256: str,
) -> tuple[AuthorizationSpec, SourceArtifactSpec]:
    if authorization_type == "inference_repair":
        relative_path = "manifests/approved_rerun_queue.csv"
        executable = True
    elif authorization_type == "offline_transformation":
        relative_path = "manifests/canonical_results_manifest.csv"
        executable = False
    else:
        raise Stage1ProfileError(f"Unsupported authorization_type {authorization_type!r}.")

    expected_sha256 = ledger.get(relative_path)
    if expected_sha256 is None:
        raise Stage1ProfileError(f"{relative_path} has no checksum ledger entry.")
    source_id = f"{cell_id}__{authorization_type}_authorization_source"
    source = SourceArtifactSpec(
        source_id=source_id,
        path=freeze_root / relative_path,
        logical_path=relative_path,
        expected_sha256=expected_sha256,
        format="csv",
        format_version="paper_data_freeze.v1",
        classification="canonical",
        dialect=CsvDialectSpec(),
        columns={},
        expected_columns=(),
        ignored_columns={},
        null_values=(),
        numeric_columns=(),
        option_mapping=OptionMappingSpec(
            mode="ordered_columns", ordered_columns=(), structured_column=None,
            structured_label_key=None, structured_text_key=None,
        ),
        extra_field_policy="preserve_unmapped",
        preserve_namespace="stage1_paper_freeze",
        source_run_id=None, source_repository=None, source_commit=None,
        notes={"role": f"{authorization_type}_authorization_evidence", "cell_id": cell_id},
    )

    purpose = (
        "Stage 1 approved rerun repair" if authorization_type == "inference_repair"
        else "Stage 1 offline-recoverable semantic rematch"
    )
    payload = {
        "schema_version": "choicebench.authorization-bundle.v1",
        "authorization_type": authorization_type,
        "grants": {condition_digest: dict(question_reasons)},
        "authority": "paper_data_freeze.manifests",
        "purpose": purpose,
        "executable": executable,
        "source_sha256": expected_sha256,
        "input_evidence_digests": {cell_id: cell_evidence_sha256},
        "expected_snapshot_digests": {condition_digest: dataset_snapshot_digest},
    }
    authorization_id = short_id("auth", payload)
    authorization = AuthorizationSpec(
        authorization_id=authorization_id,
        authorization_type=authorization_type,
        source_id=source_id,
        condition_question_reasons={condition_digest: dict(question_reasons)},
        authority="paper_data_freeze.manifests",
        purpose=purpose,
        executable=executable,
        input_evidence_digests={cell_id: cell_evidence_sha256},
        expected_snapshot_digests={condition_digest: dataset_snapshot_digest},
    )
    return authorization, source
