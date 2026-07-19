# experiments/visible_llm_matcher/repairs/group1_historical_arc/harness/variable_option_protocol.py
#
# Ported historical prompt-building + cyclic-permutation logic for the
# `baseline` (DirectMCQRunner) and `cyclic_generation_majority`
# (PermutationRunner) methods, plus the ONE permitted correction for the 3
# audited three-option ARC questions: real option count only, never a
# synthetic "D. nan".
#
# Source of truth: two-stage-prompting @ b8e784f3eb5d2a727a97eb675140b383a34584fa
#   src/twoprompt/runners/direct_mcq.py    (DirectMCQRunner)
#   src/twoprompt/runners/permutation.py   (PermutationRunner)
#   src/twoprompt/pipeline/prompt_builder.py::build_direct_mcq_prompt
#
# Parsing is NOT duplicated here — both historical runners use the shared
# ExperimentRunner._parse (twoprompt.parsing.parser.parse_model_answer),
# already ported byte-for-byte in
# experiments/visible_llm_matcher/historical_protocol.py. This module
# reuses that, unmodified, for both the 4-option and 3-option paths — per
# instruction, parsing itself is not part of the one permitted correction.

from __future__ import annotations

import collections
from typing import Any

from choicebench.parsing.types import PARSE_MISSING, PARSE_OK, ParseResult

from experiments.visible_llm_matcher.historical_protocol import parse_model_answer

# ---------------------------------------------------------------------------
# baseline (direct_mcq)
# ---------------------------------------------------------------------------


def build_direct_mcq_prompt(
    template: str,
    question: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
) -> str:
    """Ported verbatim from twoprompt.pipeline.prompt_builder.build_direct_mcq_prompt.
    Historical, 4-slot, hardcoded — for ordinary 4-option rows only."""
    return template.format(
        question=question,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )


def build_direct_mcq_prompt_repaired_3option(
    template: str,
    question: str,
    option_a: str,
    option_b: str,
    option_c: str,
) -> str:
    """The one permitted correction for baseline: real option count only.
    template must be direct_mcq_repaired_3option.txt (no option_d slot at
    all, so there is no way to accidentally interpolate one)."""
    return template.format(question=question, option_a=option_a, option_b=option_b, option_c=option_c)


# ---------------------------------------------------------------------------
# cyclic_generation_majority (permutation)
# ---------------------------------------------------------------------------


def generate_cyclic_permutations(options: dict[str, str]) -> list[dict[str, str]]:
    """Ported verbatim (generalized to any option count — TSP's own
    version already loops over len(options), so for a 3-key `options` dict
    this naturally produces exactly 3 rotations, not 4; no separate
    "repaired" variant of this function is needed).

    See PermutationRunner._generate_permutations.
    """
    keys = list(options.keys())
    values = list(options.values())
    return [dict(zip(keys, values[i:] + values[:i])) for i in range(len(options))]


def build_permuted_direct_mcq_prompt(
    question_text: str,
    permuted_options: dict[str, str],
    template: str,
) -> str:
    """Ported from PermutationRunner._build_permuted_prompt, generalized to
    any real option count (historical version hardcodes exactly 4
    positional values — vals[0..3] — assuming len(permuted_options) == 4
    always; for the repaired 3-option path, permuted_options has 3 keys
    and template is direct_mcq_repaired_3option.txt, so only vals[0..2] are
    used).
    """
    vals = list(permuted_options.values())
    if len(vals) == 4:
        return build_direct_mcq_prompt(
            template=template,
            question=question_text,
            option_a=vals[0],
            option_b=vals[1],
            option_c=vals[2],
            option_d=vals[3],
        )
    if len(vals) == 3:
        return build_direct_mcq_prompt_repaired_3option(
            template=template,
            question=question_text,
            option_a=vals[0],
            option_b=vals[1],
            option_c=vals[2],
        )
    raise ValueError(
        f"build_permuted_direct_mcq_prompt: expected 3 or 4 real options, got {len(vals)}. "
        "This harness only handles the audited 3-option repair or the historical 4-option case."
    )


def unpermute_choice(
    parsed_letter: str,
    permuted_options: dict[str, str],
    canonical_options: dict[str, str],
) -> str | None:
    """Ported from PermutationRunner._unpermute_choice, with one addition:
    the historical parser (parse_model_answer) never restricts valid
    letters to the real option count (see historical_protocol.py), so for
    a repaired 3-option question the model's raw text could still yield a
    parsed_letter of "D" even though no D option exists. The historical
    code (permuted_options[parsed_letter]) would KeyError in that case,
    which never happened historically because real 4-option questions
    always had a D. Here, a parsed letter absent from permuted_options is
    treated as an unmapped/failed vote (returns None) rather than crashing
    — the model referenced an option that does not exist for this
    question, which is not evidence for any real option.
    """
    selected_text = permuted_options.get(parsed_letter)
    if selected_text is None:
        return None
    for key, value in canonical_options.items():
        if value == selected_text:
            return key
    return None


def majority_vote(choices: list[str | None]) -> str | None:
    """Ported verbatim from PermutationRunner._majority_vote — tie-break
    is "first valid vote wins", unchanged for both 3- and 4-rotation
    cases."""
    cleaned = [x for x in choices if x is not None]
    if not cleaned:
        return None
    top = collections.Counter(cleaned).most_common(2)
    if len(top) == 1 or top[0][1] != top[1][1]:
        return top[0][0]
    return cleaned[0]


def build_voted_parse_result(voted_letter: str | None) -> ParseResult:
    """Ported from PermutationRunner.run_one's synthetic ParseResult
    construction for the majority-vote outcome."""
    return ParseResult(
        final_choice=voted_letter,
        status=PARSE_OK if voted_letter else PARSE_MISSING,
        raw_text=None,
        normalized_text="",
        reason="majority_vote",
    )


def parse_permutation_response(raw_text: str, permutation_options: dict[str, str]) -> ParseResult:
    """The per-permutation parse step (PermutationRunner.run_one calls
    self._parse(response.raw_text, permutation) — the shared, ported
    parse_model_answer). Exposed as a named function here so callers don't
    need to import historical_protocol directly.
    """
    return parse_model_answer(raw_text, permutation_options)
