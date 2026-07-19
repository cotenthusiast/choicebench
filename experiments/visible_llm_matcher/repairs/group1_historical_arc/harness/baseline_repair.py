# experiments/visible_llm_matcher/repairs/group1_historical_arc/harness/baseline_repair.py
#
# Repair for the `baseline` (DirectMCQRunner) method: single call per
# question. Ordinary 4-option rows are byte-faithful to the historical
# direct_mcq.txt path (not touched by this harness at all — only the 3
# audited question_ids are ever routed through the repaired path). For the
# 3 repaired IDs: one call, direct_mcq_repaired_3option.txt, real A/B/C
# only.

from __future__ import annotations

from typing import Any

from choicebench.backends.base import BaseBackend

from experiments.visible_llm_matcher.historical_protocol import parse_model_answer
from experiments.visible_llm_matcher.stage1_sources import KNOWN_3OPTION_ARC_QUESTION_IDS

from .repair_infra import assert_fresh_call
from .variable_option_protocol import build_direct_mcq_prompt_repaired_3option


async def repair_one_baseline_question(
    backend: BaseBackend,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Repair one baseline question. row must be a genuinely 3-option
    question (choice_d absent) among KNOWN_3OPTION_ARC_QUESTION_IDS —
    caller (run_repair.py) enforces that; this function trusts it and
    fails loudly if given anything else, rather than silently rendering a
    4-option prompt or a synthetic one.

    Async: choicebench.backends.api_backend.APIBackend.generate() always
    raises RuntimeError ("requires a running event loop") — the only
    working single-request entry point is the async
    generate_single_async(), which this function awaits.
    """
    question_id = row["question_id"]
    if question_id not in KNOWN_3OPTION_ARC_QUESTION_IDS:
        raise ValueError(
            f"repair_one_baseline_question: question_id={question_id!r} is "
            "not in the audited 3-option repair set — refusing to run the "
            "repaired path for an unaudited question."
        )

    prompt = build_direct_mcq_prompt_repaired_3option(
        template=_load_repaired_template(),
        question=row["question_text"],
        option_a=row["choice_a"],
        option_b=row["choice_b"],
        option_c=row["choice_c"],
    )

    response = await backend.generate_single_async(prompt)
    assert_fresh_call(response, question_id=question_id)

    parsed_result = None
    score_result = None
    if response.is_success():
        from choicebench.scoring.scorer import score_prediction

        options = {"A": row["choice_a"], "B": row["choice_b"], "C": row["choice_c"]}
        parsed_result = parse_model_answer(response.raw_text, options)
        score_result = score_prediction(parsed_result, row["correct_option"])

    return {
        "question_id": question_id,
        "subject": row["subject"],
        "method_name": "baseline",
        "model_name": row.get("model_name"),
        "question_text": row["question_text"],
        "choice_a": row["choice_a"],
        "choice_b": row["choice_b"],
        "choice_c": row["choice_c"],
        "choice_d": None,
        "correct_option": row["correct_option"],
        "prompt": prompt,
        "raw_text": response.raw_text,
        "model_status": response.status,
        "latency_seconds": response.latency_seconds,
        "timestamp_utc": response.timestamp_utc,
        "parsed_choice": parsed_result.final_choice if parsed_result else None,
        "parse_status": parsed_result.status if parsed_result else None,
        "is_correct": score_result.is_correct if score_result else None,
        "score_status": score_result.status if score_result else None,
    }


_REPAIRED_TEMPLATE_CACHE: str | None = None


def _load_repaired_template() -> str:
    global _REPAIRED_TEMPLATE_CACHE
    if _REPAIRED_TEMPLATE_CACHE is None:
        from .repair_infra import REPO_ROOT

        path = REPO_ROOT / "experiments" / "visible_llm_matcher" / "prompts" / "v1" / "direct_mcq_repaired_3option.txt"
        _REPAIRED_TEMPLATE_CACHE = path.read_text(encoding="utf-8")
    return _REPAIRED_TEMPLATE_CACHE
