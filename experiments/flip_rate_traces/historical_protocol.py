# experiments/flip_rate_traces/historical_protocol.py
#
# Documents, and where necessary reproduces, the historical cyclic-generation-
# majority protocol this experiment must stay faithful to, verified against
# clean checkouts (not memory or paper prose) of the two source repos:
#
#   Cloud/API side (gpt-4.1-mini):
#     two-stage-prompting @ b8e784f (HEAD as of this experiment; unchanged
#     since 5adc049 "feat: implement experiment runners with full test
#     coverage")
#       src/twoprompt/runners/permutation.py  (PermutationRunner)
#
#   Local side (Qwen/Qwen2.5-7B-Instruct):
#     model-generalization @ bb9d3c6 ("Fix leading-letter and span-isolation
#     gaps in abcd/text_extraction matching" — the commit immediately before
#     f306e6e "Redesign additional_option (Eq.6) and cyclic (Eq.1) for paper
#     fidelity", which replaced generation+majority-vote with score_options()
#     probability averaging). Current model-generalization HEAD (24403d8) is
#     NOT the historical generation-majority protocol for `cyclic` — it is a
#     different, later-designed estimator. bb9d3c6 is.
#       src/modelgen/runners/permutation.py  (PermutationRunner)
#
# Canonical frozen source data for both cells (paper_data_freeze/manifests/
# canonical_results_manifest.csv):
#   cbp__gpt-4-1-mini__arc_challenge__cyclic_generation_majority
#     -> raw/local_two_stage_prompting/runs/20260601_183346/..., sha256
#        d7a1d3244294d8a4486581ba377d62187fe7626cf8e000ef0428b69525200ef9
#        status=malformed_requires_inference; damaged_question_ids=
#        ["79e8c959bbeb74a0","ad6b5d46ae54842c","c30e75b011696a95"]
#   cbp__gpt-4-1-mini__mmlu__cyclic_generation_majority
#     -> same run, sha256 dba83af2b8b25db4b689bac61a43e3068f776c028a3aa8e2c608b16180113708
#        status=canonical_complete
#   cbp__qwen-qwen2-5-7b-instruct__arc_challenge__cyclic_generation_majority
#     -> raw/local_model_generalization/runs_archive/
#        cyclic_majority_vote_pre_eq1_redesign_20260621/20260529_145812/...,
#        sha256 ad7de543bf8bda698c9442f07619e591c4eca7708623f182561ebf82ce08d883
#        status=canonical_complete
#   cbp__qwen-qwen2-5-7b-instruct__mmlu__cyclic_generation_majority
#     -> same run, sha256 570016a9c291240c960a4cd2023fa89b662e81e02e8d0fd3804dc1329b9b0843
#        status=canonical_complete
#
# Historical generation settings (read directly from the canonical CSVs'
# own columns, both sides agree): temperature=0.0, max_tokens=500, seed=42,
# prompt_version="v1".
#
# --- What is byte-identical across ChoiceBench, the cloud repo, and the
#     local (pre-redesign) repo, verified by direct diff ---
#
#   _generate_permutations(): all three implementations are the same
#   algorithm: `dict(zip(keys, values[i:] + values[:i])) for i in
#   range(len(options))`. i=0 is the canonical (unrotated) ordering. Reused
#   directly from choicebench.methods.library.permutation.PermutationRunner.
#
#   _unpermute_choice(): all three map the permuted letter -> its text ->
#   scan canonical_options for the matching text -> canonical letter.
#   Byte-identical logic. Reused directly.
#
#   Prompt rendering: v1 of this experiment reused choicebench's own packaged
#   v1 direct_mcq template for both sides. That template's CONTENT is
#   byte-identical to model-generalization's own v1 template (both use a
#   dynamic per-label `{options}` block) -- but choicebench's packaged copy
#   carries an extra trailing newline neither historical template has, and
#   this went undetected through a full broad launch (found only via
#   post-hoc comparison against preserved historical data; see
#   diagnostics_excluded_from_authoritative/flip_traces_trailing_newline_variant/
#   in the paper bundle for the excluded v1 traces and full writeup). This
#   version instead loads the two historical template files verbatim (see
#   build_api_prompt/build_local_prompt below and
#   historical_templates/{api,local}_v1_direct_mcq.txt) and never touches
#   choicebench's packaged copy.
#
#   Cloud/API template specifically: two-stage-prompting's v1 template
#   hardcodes four literal lines ("A. {option_a}\nB. {option_b}\nC.
#   {option_c}\nD. {option_d}"), and its _build_options() never dropped a
#   missing choice_d. Verified directly against the canonical CSV
#   (20260601_183346, question 79e8c959bbeb74a0): the historical stored
#   prompt for this run literally contains the line "D. nan" for gpt-4.1-mini
#   cyclic. This is exactly the contamination
#   paper_data_freeze/manifests/canonical_results_manifest.csv flags as
#   status=malformed_requires_inference / damaged_question_ids for this one
#   cell. Per this experiment's explicit instructions ("no synthetic D. nan
#   option may be rendered, parsed or permuted" / "the three genuine
#   three-option ARC questions use exactly three permutations"),
#   build_api_prompt does NOT reproduce that bug: for the 997 ordinary
#   questions it substitutes the historical template's 4 positional slots
#   exactly (byte-identical to history, verified against real historical
#   rows in tests/experiments/test_flip_rate_traces.py); for the 3
#   known-contaminated ARC questions it removes only the "D. {option_d}"
#   line from that same historical template string before substituting --
#   so the repaired prompt differs from the historical contaminated prompt
#   by exactly that one line's removal, nothing else.
#
#   Majority-vote tie-break: choicebench's own current
#   methods/library/permutation.py._majority_vote breaks ties by
#   canonical-letter order (an explicit, documented improvement to cancel
#   positional correlation — see that file's own docstring). BOTH historical
#   implementations (cloud b8e784f and local bb9d3c6) instead break ties by
#   "the first valid vote", i.e. the first permutation index (in
#   for-loop/generation order) that produced a parseable answer:
#   `return cleaned[0]` after `collections.Counter` tie detection — verified
#   identical in both repos' _majority_vote. This experiment must preserve
#   history's tie-break, not choicebench's improved one, so it is
#   reimplemented here rather than reused from choicebench.

from __future__ import annotations

import collections

KNOWN_3OPTION_ARC_QUESTION_IDS = frozenset({
    "79e8c959bbeb74a0",
    "ad6b5d46ae54842c",
    "c30e75b011696a95",
})

HISTORICAL_TEMPERATURE = 0.0
HISTORICAL_MAX_TOKENS = 500
HISTORICAL_SEED = 42
HISTORICAL_PROMPT_VERSION = "v1"


def generate_permutations(options: dict[str, str]) -> list[dict[str, str]]:
    """Cyclic rotations of the canonical option ordering.

    Byte-identical to two-stage-prompting @ b8e784f, model-generalization @
    bb9d3c6, and choicebench's own PermutationRunner. i=0 is the canonical
    (unrotated) ordering — included as one of the N permutations, per
    paper_data_freeze/raw/prior_reconciliation/11_CHOICEBENCH_INTEGRATION_MAP.md
    ("Generate all cyclic orders -- including original order").
    """
    keys = list(options.keys())
    values = list(options.values())
    return [
        dict(zip(keys, values[i:] + values[:i]))
        for i in range(len(options))
    ]


def unpermute_choice(
        parsed_letter: str,
        permuted_options: dict[str, str],
        canonical_options: dict[str, str],
) -> str | None:
    """Map a parsed letter from permuted ordering back to canonical.

    Byte-identical logic across both historical repos and choicebench.
    """
    if parsed_letter not in permuted_options:
        return None
    selected_text = permuted_options[parsed_letter]
    for key, value in canonical_options.items():
        if value == selected_text:
            return key
    return None


def historical_majority_vote(choices: list[str | None]) -> str | None:
    """Determine the final answer by majority vote, historical tie-break.

    Ties are broken by the first valid (non-None) vote in permutation-
    processing order — NOT choicebench's current canonical-letter-order
    tiebreak. Verified identical in two-stage-prompting @ b8e784f
    (PermutationRunner._majority_vote) and model-generalization @ bb9d3c6
    (same method, same body).
    """
    cleaned = [x for x in choices if x is not None]
    if not cleaned:
        return None

    top = collections.Counter(cleaned).most_common(2)
    if len(top) == 1 or top[0][1] != top[1][1]:
        return top[0][0]
    return cleaned[0]


# --- Prompt rendering (v2 correction) ---
#
# v1 of this experiment (see docs/flip_rate_traces/PROTOCOL_DIFF.md and the
# excluded-diagnostic README under
# diagnostics_excluded_from_authoritative/flip_traces_trailing_newline_variant/
# in the paper bundle) reused choicebench's own packaged v1 direct_mcq
# template. That template is byte-identical in CONTENT to the historical
# local template, but ChoiceBench's copy carries an extra trailing newline
# neither historical template has, and it renders options via a dynamic
# {options} block rather than the historical API template's hardcoded
# A/B/C/D slots. Confirmed by direct byte diff (od -c) against clean
# checkouts of both source repos -- not assumed.
#
# This version instead loads the two historical template files verbatim
# (historical_templates/api_v1_direct_mcq.txt, byte-identical to
# two-stage-prompting @ b8e784f's prompts/v1/direct_mcq.txt, 165 bytes, no
# trailing newline; historical_templates/local_v1_direct_mcq.txt,
# byte-identical to model-generalization @ bb9d3c6's prompts/v1/direct_mcq.txt,
# 119 bytes, no trailing newline) and renders each per its own historical
# structure.

from pathlib import Path as _Path

_TEMPLATE_DIR = _Path(__file__).resolve().parent / "historical_templates"
API_TEMPLATE = (_TEMPLATE_DIR / "api_v1_direct_mcq.txt").read_text()
LOCAL_TEMPLATE = (_TEMPLATE_DIR / "local_v1_direct_mcq.txt").read_text()

_LETTER_TO_SLOT = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}


def build_api_prompt(question_text: str, options: dict[str, str]) -> str:
    """Render the API-side prompt using the historical hardcoded template.

    For the 997 ordinary 4-option questions, this substitutes all four
    positional slots exactly as two-stage-prompting's
    build_direct_mcq_prompt did -- byte-identical output.

    For the 3 known 3-option ARC questions, historically this template's
    unconditional 4-slot substitution produced the literal contaminated line
    "D. nan" (option_d arriving as NaN from pandas, then str-formatted). Per
    this experiment's explicit instruction, that contamination is not
    reproduced. The repaired prompt is derived by removing ONLY the "D.
    {option_d}" line from the same historical template string and
    substituting the remaining slots -- so the repaired prompt differs from
    the historical contaminated prompt by exactly that one line's removal
    (the surrounding blank line before "Respond with only the letter."
    already exists in the historical template and needs no adjustment).
    """
    real_letters = list(options.keys())
    if real_letters == ["A", "B", "C", "D"]:
        return API_TEMPLATE.format(
            question=question_text,
            option_a=options["A"], option_b=options["B"],
            option_c=options["C"], option_d=options["D"],
        )
    # Reduced case: drop the template line for every missing letter.
    missing = [letter for letter in ("A", "B", "C", "D") if letter not in options]
    lines = API_TEMPLATE.split("\n")
    kept_lines = [
        line for line in lines
        if not any(line.startswith(f"{letter}. {{{_LETTER_TO_SLOT[letter]}}}") for letter in missing)
    ]
    reduced_template = "\n".join(kept_lines)
    format_kwargs = {"question": question_text}
    for letter in real_letters:
        format_kwargs[_LETTER_TO_SLOT[letter]] = options[letter]
    return reduced_template.format(**format_kwargs)


def build_local_prompt(question_text: str, options: dict[str, str]) -> str:
    """Render the local-side prompt using the historical dynamic-block template.

    Byte-identical to model-generalization @ bb9d3c6's own rendering for
    every question (that repo's local _build_options already dropped missing
    choices before this template ever saw them, so no separate reduced case
    is needed here).
    """
    options_block = "\n".join(f"{letter}. {text}" for letter, text in options.items())
    return LOCAL_TEMPLATE.format(question=question_text, options=options_block)
