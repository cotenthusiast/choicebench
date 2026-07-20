# experiments/flip_rate_traces/trace_schema.py
#
# Stable per-permutation trace record for the flip-rate diagnostic. One row
# per (question, permutation) — the one thing the historical protocol
# discarded (see historical_protocol.py). Everything else about the
# question/permutation/vote is also carried per row so the majority vote and
# every flip-rate metric can be recomputed offline from these rows alone,
# with no dependency on the runner's in-memory state.

from __future__ import annotations

TRACE_SCHEMA_VERSION = "flip_rate_traces.trace.v1"

# Column order for CSV serialization. Keep in sync with build_trace_row().
TRACE_COLUMNS = [
    "schema_version",
    "run_id",
    "cell_id",             # e.g. cbp__gpt-4-1-mini__arc_challenge__cyclic_generation_majority_traced_v1
    "question_id",
    "benchmark",
    "split_name",
    "subject",
    "model_name",
    "provider",
    "n_options",           # real option count for this question (3 or 4)
    "permutation_index",   # 0..n_options-1; 0 = canonical/unrotated order
    "n_permutations",      # == n_options, redundant but explicit for validation
    # --- displayed <-> semantic mapping ---
    # canonical_options: {letter: text} in original (unpermuted) order.
    # displayed_options: {letter: text} as actually shown to the model this permutation.
    # displayed_to_semantic: {displayed_letter: canonical_letter} for this permutation.
    "canonical_options_json",
    "displayed_options_json",
    "displayed_to_semantic_json",
    "correct_option",      # canonical (semantic) gold letter
    # --- prompt / raw output ---
    "prompt",
    "prompt_sha256",
    "temperature",
    "max_tokens",
    "seed",
    "prompt_version",
    "raw_text",
    "finish_reason",
    # --- parse ---
    "displayed_parsed_choice",   # letter as parsed directly from raw_text (permuted space)
    "semantic_parsed_choice",    # displayed_parsed_choice mapped back to canonical letter
    "parse_status",
    "parse_reason",
    "normalized_text",
    # --- correctness (this permutation alone, not the vote) ---
    "is_correct",
    "score_status",
    # --- transport / failure ---
    "transport_status",
    "error_type",
    "error_message",
    "error_stage",
    "error_retryable",
    "latency_seconds",
    "timestamp_utc",
    "cache_hit",
    # --- majority-vote outcome (duplicated onto every row of the question,
    #     so a single trace row is self-sufficient for majority-vote QA
    #     without a join; still independently recomputable from the sibling
    #     rows' semantic_parsed_choice values) ---
    "majority_semantic_choice",
    "majority_is_tie",
    "majority_tie_break_used",
    "majority_is_correct",
    "is_diagnostic_canary",
]


class TraceValidationError(ValueError):
    """Raised when a trace row or a question's full permutation set is invalid."""


def build_trace_row(**kwargs) -> dict:
    """Assemble one trace row, enforcing the full column set is present.

    Every field in TRACE_COLUMNS must be supplied explicitly (use None for
    fields that don't apply, e.g. error_* on a successful row) — no silent
    defaults, so a caller can never accidentally omit a required field.
    """
    missing = [c for c in TRACE_COLUMNS if c not in kwargs]
    if missing:
        raise TraceValidationError(f"build_trace_row missing required fields: {missing}")
    extra = [k for k in kwargs if k not in TRACE_COLUMNS]
    if extra:
        raise TraceValidationError(f"build_trace_row got unknown fields: {extra}")
    return {c: kwargs[c] for c in TRACE_COLUMNS}


def validate_question_traces(question_id: str, rows: list[dict]) -> None:
    """Validate the full set of trace rows for one question.

    Raises TraceValidationError on any violation. Checked invariants:
      - at least one row, and exactly n_permutations rows (from the first
        row's own n_permutations field)
      - permutation_index values are exactly range(n_permutations), each
        appearing once
      - n_options == n_permutations for every row (no synthetic option was
        added or dropped mid-question)
      - no row's displayed_options_json contains a "nan" text value
      - every row's error fields are non-empty whenever transport_status is
        not "success" (no silent success-shaped failure row)
      - majority_semantic_choice, majority_is_tie, majority_is_correct are
        identical across all rows of the question (single vote per question)
    """
    if not rows:
        raise TraceValidationError(f"{question_id}: no trace rows found.")

    n_perm = rows[0]["n_permutations"]
    if any(r["n_permutations"] != n_perm for r in rows):
        raise TraceValidationError(f"{question_id}: n_permutations disagrees across rows.")
    if len(rows) != n_perm:
        raise TraceValidationError(
            f"{question_id}: expected {n_perm} trace rows, found {len(rows)}."
        )

    seen_indices = sorted(r["permutation_index"] for r in rows)
    if seen_indices != list(range(n_perm)):
        raise TraceValidationError(
            f"{question_id}: permutation_index set {seen_indices} != range({n_perm})."
        )

    for r in rows:
        if r["n_options"] != n_perm:
            raise TraceValidationError(
                f"{question_id} perm {r['permutation_index']}: n_options={r['n_options']} "
                f"!= n_permutations={n_perm}."
            )
        if "nan" in (r["displayed_options_json"] or "").lower():
            raise TraceValidationError(
                f"{question_id} perm {r['permutation_index']}: displayed_options_json "
                "contains a literal 'nan' — phantom option was rendered."
            )
        if r["transport_status"] != "success":
            if not r["error_type"] or not r["error_message"] or not r["error_stage"]:
                raise TraceValidationError(
                    f"{question_id} perm {r['permutation_index']}: transport_status="
                    f"{r['transport_status']!r} but error metadata is incomplete "
                    f"(error_type={r['error_type']!r}, error_message={r['error_message']!r}, "
                    f"error_stage={r['error_stage']!r})."
                )

    for field in ("majority_semantic_choice", "majority_is_tie", "majority_is_correct"):
        values = {r[field] for r in rows}
        if len(values) != 1:
            raise TraceValidationError(
                f"{question_id}: {field} disagrees across permutation rows: {values}."
            )
