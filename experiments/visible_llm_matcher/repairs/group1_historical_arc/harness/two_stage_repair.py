# experiments/visible_llm_matcher/repairs/group1_historical_arc/harness/two_stage_repair.py
#
# Repair for two_stage_v1/v2/v3 (TwoStageRunner, prompt_version v1/v2/v3):
# Stage 1 (hidden options, free_text.txt — structurally never shows
# options at all, so it cannot render a synthetic D regardless of prompt
# version) is REUSED, not regenerated, for all 3 repaired questions across
# all 3 versions and all 6 models (4 API + 2 local) — see
# STAGE1_REUSE_DECISION below for the investigation this rests on. Only
# Stage 2 (option_matching, which DOES show options and DOES render
# "D. nan" historically) is corrected, via the existing
# repaired_stage2.build_repaired_3option_matching_prompt (built for the
# fourth cell, reused here unchanged — same protocol, same correction).

from __future__ import annotations

from typing import Any

from choicebench.backends.base import BaseBackend
from choicebench.scoring.scorer import score_prediction

from experiments.visible_llm_matcher.historical_protocol import parse_model_answer
from experiments.visible_llm_matcher.repaired_stage2 import build_repaired_3option_matching_prompt
from experiments.visible_llm_matcher.stage1_sources import KNOWN_3OPTION_ARC_QUESTION_IDS

from .repair_infra import assert_fresh_call

# Investigated directly against stored artifacts before deciding reuse
# (2026-07-20), not assumed and not inferred from a cache hit:
#   - Stage 1 (free_text.txt) never shows options at all for ANY prompt
#     version (v1/v2/v3 differ only in wording, not in whether options are
#     shown) -- structurally cannot be affected by the phantom-D defect,
#     which is specifically an "options are rendered with a NaN 4th slot"
#     bug.
#   - All 16 (version, model) historical sources -- v1/v2/v3 x 4 API
#     models, plus v2/v3 x 2 local models (v1 local is not in the repair
#     scope) -- have complete, non-empty free_text_response for all 3
#     repaired question_ids (verified by direct row count + content check,
#     not by trusting row presence alone).
#   - Content is domain-plausible (e.g. "Solid", "decrease", "The truck
#     will most likely roll slower." -- short, on-topic free-text answers
#     matching the question domain), not placeholder/error text.
#   - latency_seconds == 0.0 / timestamp_utc empty across the ENTIRE
#     historical dataset (all 1000 rows, every column, not just these 3)
#     is a pre-existing limitation of two-stage-prompting's OpenAIClient
#     (which hardcodes latency_seconds=0.0, timestamp_utc=None
#     unconditionally -- verified by reading the client source), not a
#     signal specific to these 3 rows or evidence of a cache-hit artifact.
#     It provides no positive OR negative evidence either way and is
#     explicitly not relied upon as provenance for the reuse decision.
STAGE1_REUSE_DECISION = "reused_verified_clean"
STAGE1_REUSE_REASON = (
    "Stage 1 (free_text.txt, all versions) never renders options at all, "
    "so it cannot show a synthetic D regardless of prompt version; "
    "confirmed complete (non-empty free_text_response) and content-plausible "
    "for all 3 repaired IDs across all 16 (version, model) historical "
    "sources. Not reused on the basis of a cache hit or row presence alone."
)

_REPAIRED_STAGE2_TEMPLATE_CACHE: str | None = None


def _load_repaired_stage2_template() -> str:
    global _REPAIRED_STAGE2_TEMPLATE_CACHE
    if _REPAIRED_STAGE2_TEMPLATE_CACHE is None:
        from .repair_infra import REPO_ROOT

        path = (
            REPO_ROOT / "experiments" / "visible_llm_matcher" / "prompts" / "v1"
            / "option_matching_repaired_3option.txt"
        )
        _REPAIRED_STAGE2_TEMPLATE_CACHE = path.read_text(encoding="utf-8")
    return _REPAIRED_STAGE2_TEMPLATE_CACHE


async def repair_one_two_stage_question(
    backend: BaseBackend,
    row: dict[str, Any],
    *,
    reused_free_text_response: str,
    method_name: str,
) -> dict[str, Any]:
    """Repair one two_stage_v1/v2/v3 question: Stage 1 reused
    (reused_free_text_response, from the historical source — see
    STAGE1_REUSE_DECISION), Stage 2 is the one fresh call, using the
    repaired A/B/C-only option_matching prompt.
    """
    question_id = row["question_id"]
    if question_id not in KNOWN_3OPTION_ARC_QUESTION_IDS:
        raise ValueError(
            f"repair_one_two_stage_question: question_id={question_id!r} is "
            "not in the audited 3-option repair set."
        )
    if not isinstance(reused_free_text_response, str) or not reused_free_text_response.strip():
        raise ValueError(
            f"repair_one_two_stage_question: question_id={question_id!r} has "
            "no usable reused Stage-1 free text — refusing to proceed "
            "(this should have been caught by the Stage-1 completeness "
            "check before this function was ever called)."
        )

    matching_prompt = build_repaired_3option_matching_prompt(
        template=_load_repaired_stage2_template(),
        question=row["question_text"],
        free_text=reused_free_text_response,
        option_a=row["choice_a"],
        option_b=row["choice_b"],
        option_c=row["choice_c"],
    )

    response = await backend.generate_single_async(matching_prompt)
    assert_fresh_call(response, question_id=question_id)

    parsed_result = None
    score_result = None
    if response.is_success():
        options = {"A": row["choice_a"], "B": row["choice_b"], "C": row["choice_c"]}
        parsed_result = parse_model_answer(response.raw_text, options)
        score_result = score_prediction(parsed_result, row["correct_option"])

    return {
        "question_id": question_id,
        "subject": row["subject"],
        "method_name": method_name,
        "model_name": row.get("model_name"),
        "question_text": row["question_text"],
        "choice_a": row["choice_a"],
        "choice_b": row["choice_b"],
        "choice_c": row["choice_c"],
        "choice_d": None,
        "correct_option": row["correct_option"],
        "prompt": matching_prompt,
        "raw_text": response.raw_text,
        "free_text_response": reused_free_text_response,
        "free_text_stage1_disposition": STAGE1_REUSE_DECISION,
        "model_status": response.status,
        "latency_seconds": response.latency_seconds,
        "timestamp_utc": response.timestamp_utc,
        "parsed_choice": parsed_result.final_choice if parsed_result else None,
        "parse_status": parsed_result.status if parsed_result else None,
        "is_correct": score_result.is_correct if score_result else None,
        "score_status": score_result.status if score_result else None,
    }
