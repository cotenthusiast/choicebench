# src/choicebench/analysis/derive_from_free_text.py

"""Zero-inference derivation: re-score an already-saved free-text response
against a different matching rule, without rerunning any model.

Any method whose accuracy is a pure function of an already-generated
free-text response (e.g. re-deriving a semantic-matching condition from a
two-stage run's saved Stage-1 output, or re-applying a corrected matcher/
tie-break rule to historical data) can reuse this instead of writing a new
ExperimentRunner that pretends to make backend calls it doesn't need.

This is deliberately NOT an ExperimentRunner subclass: it has no backend/
client dependency at all, by construction (see test_zero_new_model_calls),
because the whole point is that no new inference happens.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from choicebench.pipeline.options import build_label_to_source_index, build_option_map
from choicebench.parsing.types import PARSE_MISSING, PARSE_OK, ParseResult
from choicebench.scoring.scorer import score_prediction
from choicebench.scoring.text_matcher import EmbedFn, default_embed_fn, match_text_to_options

_REQUIRED_SOURCE_COLUMN = "free_text_response"


def _derive_one_row(row: dict[str, Any], *, method_name: str, embed_fn: EmbedFn) -> dict[str, Any]:
    free_text = row.get(_REQUIRED_SOURCE_COLUMN)
    options = build_option_map(row)
    label_to_source_index = build_label_to_source_index(row)

    matched_letter = None
    if free_text is not None and str(free_text).strip() != "":
        matched_letter = match_text_to_options(
            free_text, options,
            embed_fn=embed_fn,
            label_to_source_index=label_to_source_index,
            seed=row.get("seed"), benchmark_id=row.get("benchmark_name"),
            question_id=row.get("question_id"), method_name=method_name,
        )

    parsed = ParseResult(
        final_choice=matched_letter,
        status=PARSE_OK if matched_letter is not None else PARSE_MISSING,
        raw_text=free_text,
        normalized_text=str(free_text) if free_text is not None else None,
        reason="derived_text_matcher_cascade",
    )
    scored = score_prediction(parsed, row.get("correct_option"))

    derived = dict(row)
    derived["method_name"] = method_name
    derived["derived_from_run_id"] = row.get("run_id")
    derived["derived_from_method_name"] = row.get("method_name")
    derived["parsed_choice"] = parsed.final_choice
    derived["parse_status"] = parsed.status
    derived["parse_reason"] = parsed.reason
    derived["is_correct"] = scored.is_correct
    derived["score_status"] = scored.status
    # Text-matching cascade is a pure function of option TEXT content -- it
    # never consults which display letter/rotation a text occupies -- so
    # the matched answer is provably invariant to rotation. No synthetic
    # rotation loop or new calls needed: the trace is just this canonical
    # answer repeated once per option.
    derived["per_rotation_choices_json"] = json.dumps([matched_letter] * len(options))
    return derived


def derive_matched_results(
        source_df: pd.DataFrame,
        *,
        method_name: str,
        embed_fn: EmbedFn | None = None,
) -> pd.DataFrame:
    """Re-derive parsed_choice/is_correct/score_status from each row's
    already-saved free_text_response, via the shared text-matching cascade.
    Zero new model calls -- there is no backend/client parameter to take one.

    Args:
        source_df: Saved result rows from a free-text-producing method
            (e.g. two_stage_v1), each with a free_text_response column plus
            the usual question/option/provenance columns.
        method_name: The method_name to stamp on the derived rows (e.g.
            "semantic_matching_v1") -- distinct from the source rows'
            original method_name, which is preserved separately as
            derived_from_method_name for provenance.
        embed_fn: Embedder for the matcher's cosine stage; defaults lazily
            to the real sentence-transformers model (see
            choicebench.scoring.text_matcher.default_embed_fn).

    Returns:
        A new DataFrame, same columns as source_df plus derived_from_run_id/
        derived_from_method_name, with parsed_choice/parse_status/
        parse_reason/is_correct/score_status overwritten by the derivation.

    Raises:
        ValueError: if source_df lacks a free_text_response column.
    """
    if _REQUIRED_SOURCE_COLUMN not in source_df.columns:
        raise ValueError(
            f"derive_matched_results requires a {_REQUIRED_SOURCE_COLUMN!r} column "
            f"in source_df; got columns {list(source_df.columns)}."
        )
    effective_embed_fn = embed_fn or default_embed_fn
    rows = source_df.to_dict(orient="records")
    derived_rows = [
        _derive_one_row(row, method_name=method_name, embed_fn=effective_embed_fn)
        for row in rows
    ]
    return pd.DataFrame(derived_rows)
