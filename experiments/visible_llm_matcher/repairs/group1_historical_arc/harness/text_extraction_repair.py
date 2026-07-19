# experiments/visible_llm_matcher/repairs/group1_historical_arc/harness/text_extraction_repair.py
#
# Repair for `text_extraction`: one call (visible options, free text),
# then the historical offline embedding/exact/containment matcher
# (text_matcher_ported.py) — but called with the real 3-option dict, never
# a phantom D. This repair's free-text output (raw_text/free_text_response)
# is the SAME artifact the fourth cell's Stage-1 reuse consumes — see
# stage1_replacement_source in the fourth-cell configs. Do not generate a
# second Stage-1 sample for the fourth cell; the fourth cell's configs
# point at this repair's staged output.

from __future__ import annotations

from typing import Any

from choicebench.backends.base import BaseBackend

from experiments.visible_llm_matcher.stage1_sources import KNOWN_3OPTION_ARC_QUESTION_IDS

from .repair_infra import assert_fresh_call
from .text_matcher_ported import parse_result_from_text_match

_REPAIRED_TEMPLATE_CACHE: str | None = None


def _load_repaired_template() -> str:
    global _REPAIRED_TEMPLATE_CACHE
    if _REPAIRED_TEMPLATE_CACHE is None:
        from .repair_infra import REPO_ROOT

        path = REPO_ROOT / "experiments" / "visible_llm_matcher" / "prompts" / "v1" / "text_extraction_repaired_3option.txt"
        _REPAIRED_TEMPLATE_CACHE = path.read_text(encoding="utf-8")
    return _REPAIRED_TEMPLATE_CACHE


def _build_repaired_text_extraction_prompt(template: str, question: str, a: str, b: str, c: str) -> str:
    return template.format(question=question, option_a=a, option_b=b, option_c=c)


async def repair_one_text_extraction_question(
    backend: BaseBackend,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Repair one text_extraction question: 1 call, A/B/C-only Stage 1
    prompt, then the ported embedding matcher against a real 3-key options
    dict (never a phantom D)."""
    question_id = row["question_id"]
    if question_id not in KNOWN_3OPTION_ARC_QUESTION_IDS:
        raise ValueError(
            f"repair_one_text_extraction_question: question_id={question_id!r} "
            "is not in the audited 3-option repair set."
        )

    prompt = _build_repaired_text_extraction_prompt(
        _load_repaired_template(), row["question_text"], row["choice_a"], row["choice_b"], row["choice_c"]
    )

    response = await backend.generate_single_async(prompt)
    assert_fresh_call(response, question_id=question_id)

    parsed_result = None
    score_result = None
    match_score = None
    if response.is_success():
        from choicebench.scoring.scorer import score_prediction

        options = {"A": row["choice_a"], "B": row["choice_b"], "C": row["choice_c"]}
        parsed_result = parse_result_from_text_match(response.raw_text, options)
        score_result = score_prediction(parsed_result, row["correct_option"])
        from .text_matcher_ported import match_text_to_options

        match_score = match_text_to_options(response.raw_text, options).score

    return {
        "question_id": question_id,
        "subject": row["subject"],
        "method_name": "text_extraction",
        "model_name": row.get("model_name"),
        "question_text": row["question_text"],
        "choice_a": row["choice_a"],
        "choice_b": row["choice_b"],
        "choice_c": row["choice_c"],
        "choice_d": None,
        "correct_option": row["correct_option"],
        "prompt": prompt,
        "raw_text": response.raw_text,
        "free_text_response": response.raw_text,
        "model_status": response.status,
        "latency_seconds": response.latency_seconds,
        "timestamp_utc": response.timestamp_utc,
        "parsed_choice": parsed_result.final_choice if parsed_result else None,
        "parse_status": parsed_result.status if parsed_result else None,
        "text_match_score": match_score,
        "is_correct": score_result.is_correct if score_result else None,
        "score_status": score_result.status if score_result else None,
    }
