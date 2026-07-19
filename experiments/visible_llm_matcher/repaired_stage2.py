# experiments/visible_llm_matcher/repaired_stage2.py
#
# Narrow, audited repair for EXACTLY the 3 known-3-option ARC-Challenge
# questions (see stage1_sources.KNOWN_3OPTION_ARC_QUESTION_IDS). These
# questions genuinely have only 3 real options. TSP's historical Stage-2
# option_matching.txt protocol (see historical_protocol.py) always
# interpolates 4 hardcoded slots, so a missing 4th option renders as the
# literal text "D. nan" — that is precisely the phantom-option contamination
# the corrected replacement Stage-1 rows exist to remove. Reusing the same
# 4-slot serialization at Stage 2 for these repaired rows would silently
# reintroduce the same contamination one stage later.
#
# This module does NOT modify or generalize historical_protocol.py.
# historical_protocol.build_option_matching_prompt is unchanged, still used
# for every ordinary 4-option row (byte-faithful to history — see
# tests/experiments/test_historical_protocol.py), and still exercised by
# tests as provenance evidence of what the historical D.nan behavior
# actually was. It is simply no longer the function the production runner
# calls for these 3 specific, audited-repair question_ids. This is a
# targeted variable-option correction for exactly this one repair policy,
# not a rewrite of the historical protocol to handle arbitrary option
# counts in general.

from __future__ import annotations

REPAIRED_3OPTION_TEMPLATE_NAME = "option_matching_repaired_3option"


def build_repaired_3option_matching_prompt(
    template: str,
    question: str,
    free_text: str,
    option_a: str,
    option_b: str,
    option_c: str,
) -> str:
    """Render a Stage-2 matching prompt for a genuinely 3-option question.

    Renders exactly A/B/C. Never a D line, never the text "nan", never an
    empty 4th option — there is no option_d parameter at all, so there is
    no way to accidentally interpolate one.

    Args:
        template: prompts/v1/option_matching_repaired_3option.txt contents.
        question: Question stem.
        free_text: The reused Stage-1 free-text answer being matched.
        option_a, option_b, option_c: The 3 real option texts, in order.

    Returns:
        Fully formatted 3-option Stage-2 prompt string.
    """
    return template.format(
        question=question,
        free_text=free_text,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
    )
