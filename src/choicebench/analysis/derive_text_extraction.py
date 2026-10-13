# src/choicebench/analysis/derive_text_extraction.py

"""Zero-inference re-parse of saved text_extraction results with the intended matching cascade.

Production text_extraction runs executed through the generic async path were scored with the letter-based
free-form parser instead of the exact -> containment -> cosine cascade the method is defined by. The saved
raw_text is intact, so the correct answer is a pure function of it: this re-applies the cascade, with no
backend/client parameter at all (nothing can call a model), and keeps the original (buggy) parse next to the
corrected one for provenance.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from choicebench.clients.types import FAILURE_STATUS, SUCCESS_STATUS
from choicebench.parsing.types import PARSE_MISSING, PARSE_OK, ParseResult
from choicebench.pipeline.options import build_label_to_source_index, build_option_map
from choicebench.scoring.scorer import score_prediction
from choicebench.scoring.text_matcher import COSINE_THRESHOLD, EmbedFn, default_embed_fn, match_text_to_options

METHOD_NAME = "text_extraction"
PARSER_VERSION = f"text_matcher_cascade_v1 (exact->containment->cosine>={COSINE_THRESHOLD}, all-MiniLM-L6-v2)"
DERIVATION = "text_extraction_cascade_reparse_v1"
_ORIGINALS = ("parsed_choice", "parse_status", "parse_reason", "normalized_text", "is_correct", "score_status", "answer_status")


def _validate_source(source_df: pd.DataFrame) -> None:
    if "raw_text" not in source_df.columns:
        raise ValueError("reparse_text_extraction requires a raw_text column")
    methods = set(source_df["method_name"].astype(str).unique()) if "method_name" in source_df.columns else {METHOD_NAME}
    if methods != {METHOD_NAME}:
        raise ValueError(f"source must be text_extraction rows only, got method_name={sorted(methods)}")
    for column in ("model_name", "benchmark_name"):
        if column in source_df.columns and source_df[column].dropna().astype(str).nunique() > 1:
            raise ValueError(f"source mixes multiple {column} values -- refusing to derive from a heterogeneous artifact")


def _reparse_row(row: dict[str, Any], embed_fn: EmbedFn) -> dict[str, Any]:
    raw = row.get("raw_text")
    missing = raw is None or (isinstance(raw, float) and pd.isna(raw))
    matched = None
    if not missing:
        matched = match_text_to_options(
            raw, build_option_map(row), embed_fn=embed_fn,
            label_to_source_index=build_label_to_source_index(row),
            seed=row.get("seed"), benchmark_id=row.get("benchmark_name"),
            question_id=row.get("question_id"), method_name=METHOD_NAME,
        )
    parsed = ParseResult(
        final_choice=matched, status=PARSE_OK if matched is not None else PARSE_MISSING,
        raw_text=None if missing else raw, normalized_text=None if missing else raw, reason="text_matcher_cascade",
    )
    scored = score_prediction(parsed, row.get("correct_option")) if not missing else None

    out = dict(row)
    for col in _ORIGINALS:
        if col in row:
            out[f"original_{col}"] = row[col]
    out["parsed_choice"] = parsed.final_choice
    out["parse_status"] = parsed.status
    out["parse_reason"] = parsed.reason
    out["normalized_text"] = parsed.normalized_text
    out["answer_status"] = SUCCESS_STATUS if matched is not None else FAILURE_STATUS
    out["is_correct"] = scored.is_correct if scored else None
    out["score_status"] = scored.status if scored else "unscorable"
    out["derived_from_run_id"] = row.get("run_id")
    out["derived_from_condition_id"] = row.get("condition_id")
    out["derived_from_method_name"] = row.get("method_name")
    out["derivation"] = DERIVATION
    out["parser_version"] = PARSER_VERSION
    return out


def reparse_text_extraction(source_df: pd.DataFrame, *, embed_fn: EmbedFn | None = None) -> pd.DataFrame:
    """Re-derive parsed_choice/parse_*/is_correct/score_status/answer_status from each row's saved raw_text.

    The original values of those columns are kept as original_<column>; raw outputs, prompts and identity
    columns are untouched. No per_rotation_choices_json is written (visible options => not rotation-invariant).
    """
    _validate_source(source_df)
    fn = embed_fn or default_embed_fn
    return pd.DataFrame([_reparse_row(r, fn) for r in source_df.to_dict(orient="records")])
