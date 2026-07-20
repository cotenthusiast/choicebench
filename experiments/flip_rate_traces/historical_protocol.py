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
#   Prompt rendering for the *local* side: model-generalization's own v1
#   direct_mcq template (at bb9d3c6) is `"...Options:\n{options}\n\n..."`
#   with a dynamic per-label options block — textually identical to
#   choicebench's own src/choicebench/resources/prompts/v1/direct_mcq.txt
#   and to choicebench.pipeline.prompt_builder._build_options_block's
#   "LETTER. text" newline-joined rendering. Reusing choicebench's own
#   prompt_builder/template here reproduces the local historical prompt
#   byte-for-byte, including correctly omitting a missing option (no
#   phantom "D. nan"), because model-generalization's local _build_options
#   already dropped empty choices at bb9d3c6 (unlike the cloud side; see
#   below).
#
# --- What is NOT identical, and must be corrected here (per explicit user
#     instruction, not a silent framework fix) ---
#
#   Prompt rendering for the *cloud* side: two-stage-prompting's v1 template
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
#   three-option ARC questions use exactly three permutations"), this
#   runner does NOT reproduce that bug: it uses choicebench's own dynamic
#   options-block rendering (identical to the local side) for BOTH models,
#   which produces byte-identical output to history for all ordinary
#   4-option questions (verified: two-stage-prompting's hardcoded "A. x\nB.
#   y\nC. z\nD. w" and choicebench's "\n".join(f"{L}. {t}") produce the same
#   literal text whenever all four options are real) and correctly renders
#   exactly 3 lines / 3 permutations for the 3 known-contaminated ARC
#   questions instead.
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
