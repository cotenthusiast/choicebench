# experiments/visible_llm_matcher/repairs/group1_historical_arc/harness/cyclic_repair.py
#
# Repair for `cyclic_generation_majority` (PermutationRunner): one cyclic
# rotation per REAL option. For the 3 repaired 3-option questions, that is
# exactly 3 rotations/calls, not the historical 4 (which always rotated a
# 4th "D: nan" slot into the mix). Rotation ordering, majority vote, and
# tie-break are otherwise byte-for-byte the historical logic (see
# variable_option_protocol.py).

from __future__ import annotations

import asyncio
from typing import Any

from choicebench.backends.base import BaseBackend
from choicebench.scoring.scorer import score_prediction

from experiments.visible_llm_matcher.stage1_sources import KNOWN_3OPTION_ARC_QUESTION_IDS

from .repair_infra import assert_fresh_call
from .variable_option_protocol import (
    build_permuted_direct_mcq_prompt,
    build_voted_parse_result,
    generate_cyclic_permutations,
    majority_vote,
    parse_permutation_response,
    unpermute_choice,
)

_REPAIRED_TEMPLATE_CACHE: str | None = None


def _load_repaired_template() -> str:
    global _REPAIRED_TEMPLATE_CACHE
    if _REPAIRED_TEMPLATE_CACHE is None:
        from .repair_infra import REPO_ROOT

        path = REPO_ROOT / "experiments" / "visible_llm_matcher" / "prompts" / "v1" / "direct_mcq_repaired_3option.txt"
        _REPAIRED_TEMPLATE_CACHE = path.read_text(encoding="utf-8")
    return _REPAIRED_TEMPLATE_CACHE


async def repair_one_cyclic_question(
    backend: BaseBackend,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Repair one cyclic_generation_majority question with exactly 3
    rotations (real option count), not 4. All 3 rotation calls fire in
    parallel via asyncio.gather, matching PermutationRunner.run_one's own
    concurrency (it gathers all N permutation calls together, not
    sequentially).
    """
    question_id = row["question_id"]
    if question_id not in KNOWN_3OPTION_ARC_QUESTION_IDS:
        raise ValueError(
            f"repair_one_cyclic_question: question_id={question_id!r} is "
            "not in the audited 3-option repair set."
        )

    canonical_options = {"A": row["choice_a"], "B": row["choice_b"], "C": row["choice_c"]}
    permutations = generate_cyclic_permutations(canonical_options)
    assert len(permutations) == 3, (
        f"expected exactly 3 rotations for a 3-option question, got {len(permutations)}"
    )

    template = _load_repaired_template()
    prompts = [
        build_permuted_direct_mcq_prompt(row["question_text"], perm, template)
        for perm in permutations
    ]

    responses = await asyncio.gather(*[backend.generate_single_async(p) for p in prompts])
    for response in responses:
        assert_fresh_call(response, question_id=question_id)

    canonical_choices: list[str | None] = []
    for response, permutation in zip(responses, permutations):
        if response.is_success():
            parsed = parse_permutation_response(response.raw_text, permutation)
            if parsed.final_choice is not None:
                canonical_choices.append(
                    unpermute_choice(parsed.final_choice, permutation, canonical_options)
                )
            else:
                canonical_choices.append(None)
        else:
            canonical_choices.append(None)

    voted_letter = majority_vote(canonical_choices)
    voted_parse = build_voted_parse_result(voted_letter)

    score_result = None
    if voted_letter:
        score_result = score_prediction(voted_parse, row["correct_option"])

    return {
        "question_id": question_id,
        "subject": row["subject"],
        "method_name": "cyclic_generation_majority",
        "model_name": row.get("model_name"),
        "question_text": row["question_text"],
        "choice_a": row["choice_a"],
        "choice_b": row["choice_b"],
        "choice_c": row["choice_c"],
        "choice_d": None,
        "correct_option": row["correct_option"],
        "prompt": prompts[0],
        "raw_text": responses[0].raw_text,
        "n_rotations": len(permutations),
        "per_rotation_canonical_choices": canonical_choices,
        "model_status": responses[0].status,
        "latency_seconds": sum(r.latency_seconds or 0 for r in responses),
        "timestamp_utc": responses[0].timestamp_utc,
        "parsed_choice": voted_parse.final_choice,
        "parse_status": voted_parse.status,
        "is_correct": score_result.is_correct if score_result else None,
        "score_status": score_result.status if score_result else None,
    }
