# src/choicebench/analysis/derive_baseline_from_cyclic.py

"""Zero-inference derivation: the direct_mcq baseline IS cyclic_permutation's
own rotation-0 (canonical, unpermuted) observation -- re-scored from that
rotation's own raw_text, never a second independent target-model call.

build_rotations()'s rotation index 0 is the identity mapping (verified: for
i=0, `values[0:] + values[:0] == values`, `slot_to_canonical` is the
identity bijection) -- so rotation 0's prompt is byte-identical to what
DirectMCQRunner._build_prompt() would construct for the same question, and
PermutationRunner._build_result_row() already stores rotation 0's OWN
prompt/raw_text/transport_status/latency/timestamp/error fields as the
saved row's own top-level fields (model_response=responses[0] in both its
sync and async paths). Only parsed_choice/parse_status/normalized_text/
parse_reason/is_correct/score_status/answer_status reflect the MAJORITY
VOTE across every rotation, a materially different observation -- those
are the fields this derivation recomputes, by re-parsing rotation 0's own
raw_text against the canonical (unpermuted) options, exactly as
DirectMCQRunner itself would.

This is deliberately NOT an ExperimentRunner subclass and takes no backend/
client parameter, by construction (see test_zero_new_model_calls) -- the
whole point is that no new inference happens.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from choicebench.clients.types import FAILURE_STATUS, SUCCESS_STATUS
from choicebench.parsing.parser import parse_model_answer
from choicebench.parsing.types import PARSE_MISSING, ParseResult
from choicebench.pipeline.options import build_option_map
from choicebench.scoring.scorer import score_prediction

_REQUIRED_SOURCE_COLUMN = "per_rotation_choices_json"


def _derive_one_row(row: dict[str, Any], *, method_name: str) -> dict[str, Any]:
    raw_text = row.get("raw_text")
    transport_ok = row.get("transport_status") == SUCCESS_STATUS

    if transport_ok and raw_text is not None:
        options = build_option_map(row)
        parsed = parse_model_answer(raw_text, options)
    else:
        # Rotation 0's own call never returned successfully -- nothing to
        # parse, matching DirectMCQRunner's own model_response.is_success()
        # gate (parsed_result stays None, never crashes on missing text).
        parsed = ParseResult(
            final_choice=None, status=PARSE_MISSING, raw_text=raw_text,
            normalized_text=None, reason="rotation0_transport_failed",
        )
    scored = score_prediction(parsed, row.get("correct_option"))

    derived = dict(row)
    derived["method_name"] = method_name
    derived["derived_from_run_id"] = row.get("run_id")
    derived["derived_from_method_name"] = row.get("method_name")
    derived["parsed_choice"] = parsed.final_choice
    derived["parse_status"] = parsed.status
    derived["normalized_text"] = parsed.normalized_text
    derived["parse_reason"] = parsed.reason
    derived["is_correct"] = scored.is_correct
    derived["score_status"] = scored.status
    # Recomputed from THIS derivation's own parse, not inherited from the
    # source row -- the source row's answer_status reflects the majority
    # vote's own success, which can legitimately disagree with rotation 0
    # alone succeeding/failing.
    derived["answer_status"] = SUCCESS_STATUS if parsed.final_choice is not None else FAILURE_STATUS
    # direct_mcq has no rotations of its own -- its flip-rate is read from
    # the cyclic_permutation artifact directly (documented relationship),
    # never from direct_mcq's own file, so carrying this column through
    # would misleadingly suggest otherwise.
    derived.pop("per_rotation_choices_json", None)
    return derived


def derive_baseline_from_cyclic(
        source_df: pd.DataFrame,
        *,
        method_name: str = "direct_mcq",
) -> pd.DataFrame:
    """Re-derive the direct_mcq baseline from a saved cyclic_permutation
    run's rotation-0 observation. Zero new model calls -- there is no
    backend/client parameter to take one.

    Args:
        source_df: Saved result rows from cyclic_permutation (or
            reasoning_cyclic), each with rotation 0's own prompt/raw_text/
            transport_status/... as its top-level fields plus
            per_rotation_choices_json (used here only as an identity guard
            that source_df really is cyclic_permutation-shaped; the actual
            derivation re-parses raw_text directly rather than trusting the
            JSON field's precomputed value).
        method_name: The method_name to stamp on the derived rows (e.g.
            "direct_mcq", or "reasoning_mcq" when deriving from
            reasoning_cyclic) -- distinct from the source rows' own
            method_name, preserved separately as derived_from_method_name.

    Returns:
        A new DataFrame, same columns as source_df (minus
        per_rotation_choices_json) plus derived_from_run_id/
        derived_from_method_name, with parsed_choice/parse_status/
        normalized_text/parse_reason/is_correct/score_status/answer_status
        overwritten to reflect rotation 0 alone.

    Raises:
        ValueError: if source_df lacks a per_rotation_choices_json column
            (i.e. doesn't look like a cyclic_permutation/reasoning_cyclic
            result file).
    """
    if _REQUIRED_SOURCE_COLUMN not in source_df.columns:
        raise ValueError(
            f"derive_baseline_from_cyclic requires a {_REQUIRED_SOURCE_COLUMN!r} "
            f"column in source_df (i.e. a saved cyclic_permutation/reasoning_cyclic "
            f"result); got columns {list(source_df.columns)}."
        )
    rows = source_df.to_dict(orient="records")
    derived_rows = [_derive_one_row(row, method_name=method_name) for row in rows]
    return pd.DataFrame(derived_rows)
