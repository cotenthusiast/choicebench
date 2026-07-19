# experiments/visible_llm_matcher/arc_question_id_mapping.py
#
# TSP's own question_id (used throughout stage1_sources.py and runner.py,
# since Stage 1 is reused as-is from two-stage-prompting/model-generalization
# CSVs) is NOT the identifier the ChoiceBench Stage-1-freeze/import workflow
# expects. ChoiceBench's own benchmark normalizer (src/choicebench/benchmarks/
# arc.py) hashes subject+question+choices independently, and — verified by
# direct content comparison against ChoiceBench's own prepared
# data/processed/arc_challenge_normalized.csv (the "trusted ARC dataset"),
# matching on question_text + ordered real options + gold answer — that hash
# disagrees with TSP's for exactly 6 of the 1000 ARC-Challenge
# robustness-split questions used by this experiment. Any output eventually
# handed to the import workflow must key these 6 rows by their ChoiceBench
# identifier, not TSP's.
#
# Of those 6, content comparison puts them in two very different categories:
#
# RESOLVED_HASH_ONLY_MISMATCHES (3 rows) — question text, all real option
# texts (in order), and the gold answer are IDENTICAL between TSP and
# ChoiceBench. The hash differs for a purely mechanical reason: these are
# the same 3 genuinely-3-option questions tracked in stage1_sources.
# KNOWN_3OPTION_ARC_QUESTION_IDS. TSP's arc.py always builds its hash content
# string as "arc_challenge|{q}|{a}|{b}|{c}|{d}" with choice_d="" for a
# missing 4th option (still 4 pipe-delimited slots, last one empty).
# ChoiceBench's own normalizer (src/choicebench/benchmarks/base.py) builds
# its hash content as "{subject}|{question}|c0|c1|...|cN" over only the REAL
# choices — 3 slots, no trailing empty one. Different string in, different
# sha256 out, same question. Safe to remap directly.
#
# UNRESOLVED_SCIENTIFIC_MISMATCHES (3 rows) — NOT safe to remap, and NOT
# remapped here. Content comparison shows ChoiceBench's trusted ARC dataset
# carries a genuine 5th option (label E) for these 3 questions that is
# ABSENT from every historical TSP row (text_extraction and two_prompt
# alike). This is not a hashing/normalization artifact: it traces to TSP's
# src/twoprompt/benchmarks/arc.py, whose normalizer only ever extracts
# labels A-D (`_VALID_LABELS = {"A","B","C","D"}`, `label_to_text.get("A",
# ""), .get("B",...), .get("C",...), .get("D",...)` — no E handling exists
# at all), so any ARC-Challenge question with 5 real options had its true
# 5th option silently discarded at TSP's normalization step, for every
# historical run, not just this experiment's reused Stage 1. The model was
# never shown the real full option set for these 3 questions in any
# historical TSP cell. Per explicit instruction, this is flagged rather than
# silently resolved — see the top-level report for the exact question-by-
# question comparison. Do not use these 3 question_ids in this experiment's
# ARC condition until a human decision is made on how to treat them (e.g.
# exclude from the 1000-question set entirely, re-run Stage 1 with the true
# 5-option prompt, or some other resolution) — see MismatchBlockedError.

from __future__ import annotations

# tsp_question_id -> choicebench_question_id.
# Content-verified identical (question_text, 3 real ordered options, gold
# answer) against data/processed/arc_challenge_normalized.csv on 2026-07-19.
RESOLVED_HASH_ONLY_MISMATCHES: dict[str, str] = {
    # "A toy truck rolls over a smooth surface..." — slower/faster/at the
    # same speed — gold A.
    "79e8c959bbeb74a0": "e3d2e85eb821d276",
    # "In which state of matter does water have a definite shape and
    # volume?" — gas/liquid/solid — gold C.
    "ad6b5d46ae54842c": "3d671e2f1721290d",
    # "An object is attracted to a magnet..." — decrease/increase/remain
    # the same — gold A.
    "c30e75b011696a95": "ed9db36b3d68e1f7",
}

# tsp_question_id -> {choicebench_question_id, missing_option_label,
# missing_option_text} for the 3 rows where TSP's historical data is
# missing a real option ChoiceBench's trusted dataset has. NOT safe to
# auto-remap — see module docstring. Kept here (rather than omitted
# entirely) so the blocked set is explicit and auditable, not just absent.
UNRESOLVED_SCIENTIFIC_MISMATCHES: dict[str, dict[str, str]] = {
    "f87cb129d9aa26c0": {
        "choicebench_question_id": "be30ca6f5bbf0daf",
        "question_text": "How are warm-blooded animals different from cold-blooded animals?",
        "missing_option_label": "E",
        "missing_option_text": "Warm-blooded animals are found only in warm climates.",
    },
    "e970a6b50d905595": {
        "choicebench_question_id": "751926ea49e0b65b",
        "question_text": (
            "Sally placed electrodes into a beaker containing a solution and "
            "connected the electrodes to a battery..."
        ),
        "missing_option_label": "E",
        "missing_option_text": "a hypothesis",
    },
    "8aec8773c1d6b508": {
        "choicebench_question_id": "97e41313a7c454ea",
        "question_text": (
            "Years ago farmers found that corn plants grew better if decaying "
            "fish were buried near by. What did the decaying fish provide "
            "for the corn plants?"
        ),
        "missing_option_label": "E",
        "missing_option_text": "water",
    },
}


class MismatchBlockedError(RuntimeError):
    """Raised by translate_tsp_arc_question_id() for a question_id in
    UNRESOLVED_SCIENTIFIC_MISMATCHES — refuses to silently proceed."""


def translate_tsp_arc_question_id(tsp_question_id: str) -> str:
    """Map a TSP-sourced ARC-Challenge question_id to the identifier the
    ChoiceBench Stage-1-freeze/import workflow expects.

    For the 994/1000 questions where TSP's and ChoiceBench's hashes already
    agree, this is the identity (both are the same string). For the 3
    resolved hash-only mismatches, returns the mapped ChoiceBench id. For
    the 3 unresolved scientific mismatches, raises MismatchBlockedError
    rather than guessing or silently passing the TSP id through — this
    experiment must not merely bypass the mismatch.

    Args:
        tsp_question_id: question_id as it appears in the reused TSP/MG
            Stage-1 CSVs (see stage1_sources.py).

    Returns:
        The ChoiceBench-canonical question_id.

    Raises:
        MismatchBlockedError: tsp_question_id is one of the 3 questions
            where TSP's historical data is missing a real option present in
            ChoiceBench's trusted dataset — a human decision is required
            before this question can be used in the import-facing output.
    """
    if tsp_question_id in UNRESOLVED_SCIENTIFIC_MISMATCHES:
        info = UNRESOLVED_SCIENTIFIC_MISMATCHES[tsp_question_id]
        raise MismatchBlockedError(
            f"tsp_question_id={tsp_question_id!r} ({info['question_text']!r}) "
            f"is missing real option {info['missing_option_label']} "
            f"({info['missing_option_text']!r}) in every historical TSP row, "
            f"but ChoiceBench's trusted ARC dataset "
            f"(question_id={info['choicebench_question_id']!r}) has it. "
            "This is a scientific content difference, not a hashing "
            "mismatch — do not use this question_id in import-facing "
            "output until a human decision is made."
        )
    return RESOLVED_HASH_ONLY_MISMATCHES.get(tsp_question_id, tsp_question_id)
