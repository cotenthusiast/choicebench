# experiments/visible_llm_matcher/historical_protocol.py
#
# Verbatim port of the two-stage-prompting (TSP) Stage-2 "LLM matcher"
# protocol, kept deliberately separate from choicebench.parsing.parser and
# choicebench.pipeline.prompt_builder because those two modules are NOT
# behavior-identical to TSP's originals (verified by direct diff):
#
#   - choicebench.pipeline.prompt_builder renders a dynamic {options} block
#     that silently drops missing trailing options. TSP's build_option_matching_prompt
#     always interpolates option_a..option_d verbatim, so a missing option_d
#     (NaN) renders as the literal string "D. nan". This is not cosmetic: the
#     already-published two_prompt cell's own stored Stage-2 prompts contain
#     this exact "D. nan" text for the same 3 ARC question IDs this module's
#     stage1_sources.py flags as historically contaminated (verified against
#     paper_results/eval_ready/paper_api_main/20260601_183346_two_prompt_*.csv
#     in two-stage-prompting). Reproducing it here — rather than "fixing" it —
#     keeps the Stage-2 protocol byte-identical between the two LLM-matcher
#     cells (hidden+LLM vs visible+LLM), so the only experimental delta
#     between them stays the Stage-1 visibility axis, not a second,
#     uncontrolled difference in how Stage 2 renders options.
#   - choicebench.parsing.parser.extract_choice_letter is called with
#     valid_choices=tuple(options.keys()) (real letters only). TSP's
#     parse_model_answer calls extract_choice_letter with no valid_choices
#     override, i.e. always DEFAULT_VALID_CHOICES = ("A","B","C","D"),
#     regardless of how many real options a question has. Also,
#     choicebench's option-text fallback match uses token-boundary regex
#     matching; TSP's uses plain substring containment. Both are faithfully
#     reproduced below rather than "corrected".
#
# Source of truth (verified against a clean checkout, not memory):
#   two-stage-prompting @ b8e784f3eb5d2a727a97eb675140b383a34584fa
#     src/twoprompt/parsing/parser.py       (parse_model_answer, extract_choice_letter,
#                                             extract_choice_text_match, normalize_output_text)
#     src/twoprompt/pipeline/prompt_builder.py  (build_option_matching_prompt)
#     src/twoprompt/scoring/scorer.py       (score_prediction — verified functionally
#                                             identical to choicebench.scoring.scorer;
#                                             that one IS reused unmodified, see runner.py)
#   prompts/v1/option_matching.txt copied byte-for-byte into
#   experiments/visible_llm_matcher/prompts/option_matching.txt (diffed clean).

from __future__ import annotations

from collections.abc import Collection, Mapping

from choicebench.parsing.types import (
    PARSE_AMBIGUOUS,
    PARSE_MISSING,
    PARSE_OK,
    ParseResult,
)

DEFAULT_VALID_CHOICES = ("A", "B", "C", "D")


def normalize_output_text(raw_text: str | None) -> str:
    """Ported verbatim from twoprompt.parsing.parser.normalize_output_text."""
    if raw_text is None:
        return ""
    raw_text = raw_text.strip()
    if raw_text == "":
        return ""
    words = raw_text.split()
    return " ".join(words)


def extract_choice_letter(
    normalized_text: str,
    valid_choices: Collection[str] = DEFAULT_VALID_CHOICES,
) -> ParseResult:
    """Ported verbatim from twoprompt.parsing.parser.extract_choice_letter.

    Extraction priority (later occurrences always win within a tier):
    1. Standalone single-token response.
    2. Last strong cue pattern: "final answer is X", "answer is X",
       "choice is X", "option is X", "final answer X", "answer X",
       "choice X", "option X", "therefore X", "thus X"  (case-insensitive).
    3. Last standalone A/B/C/D in the response when no strong cue matched.
    """
    if normalized_text == "":
        return ParseResult(
            final_choice=None,
            status=PARSE_MISSING,
            raw_text=None,
            normalized_text=normalized_text,
            reason="No output to parse",
        )

    words = normalized_text.split()
    stripped = [w.strip('()[]{}<>".,:;!?' + "'") for w in words]

    # Priority 1: single-token response
    if len(stripped) == 1 and stripped[0].upper() in valid_choices:
        return ParseResult(
            final_choice=stripped[0].upper(),
            status=PARSE_OK,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Answer successfully parsed",
        )

    cue_words = {"answer", "choice", "option"}
    concluding_words = {"therefore", "thus"}

    last_strong: tuple[int, str] | None = None
    last_weak: tuple[int, str] | None = None

    for i, w in enumerate(stripped):
        wl = w.lower()

        if (
            i + 3 < len(stripped)
            and wl == "final"
            and stripped[i + 1].lower() == "answer"
            and stripped[i + 2].lower() == "is"
            and stripped[i + 3].upper() in valid_choices
        ):
            last_strong = (i, stripped[i + 3].upper())

        if (
            i + 2 < len(stripped)
            and wl in cue_words
            and stripped[i + 1].lower() == "is"
            and stripped[i + 2].upper() in valid_choices
        ):
            last_strong = (i, stripped[i + 2].upper())

        if (
            i + 2 < len(stripped)
            and wl == "final"
            and stripped[i + 1].lower() == "answer"
            and stripped[i + 2].upper() in valid_choices
        ):
            last_strong = (i, stripped[i + 2].upper())

        if (
            i + 1 < len(stripped)
            and wl in cue_words
            and stripped[i + 1].upper() in valid_choices
        ):
            last_strong = (i, stripped[i + 1].upper())

        if (
            i + 1 < len(stripped)
            and wl in concluding_words
            and stripped[i + 1].upper() in valid_choices
        ):
            last_strong = (i, stripped[i + 1].upper())

        if w.upper() in valid_choices:
            last_weak = (i, w.upper())

    if last_strong is not None:
        return ParseResult(
            final_choice=last_strong[1],
            status=PARSE_OK,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Answer successfully parsed from strong cue pattern",
        )
    if last_weak is not None:
        return ParseResult(
            final_choice=last_weak[1],
            status=PARSE_OK,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Answer successfully parsed from last standalone letter",
        )
    return ParseResult(
        final_choice=None,
        status=PARSE_MISSING,
        raw_text=None,
        normalized_text=normalized_text,
        reason="No direct answer letter found",
    )


def extract_choice_text_match(
    normalized_text: str,
    options: Mapping[str, str],
) -> ParseResult:
    """Ported verbatim from twoprompt.parsing.parser.extract_choice_text_match.

    Plain substring containment (normalized_option in normalized_text), NOT
    choicebench.parsing.parser's token-boundary regex match. Intentionally
    reproduces TSP's looser (and occasionally spurious-match-prone) behavior.
    """
    candidates = []
    for letter, option_text in options.items():
        normalized_option = normalize_output_text(option_text)
        if normalized_option.lower() == normalized_text.lower():
            candidates.append(letter)
        elif normalized_option.lower() in normalized_text.lower():
            candidates.append(letter)

    candidates = set(candidates)

    if len(candidates) == 1:
        return ParseResult(
            final_choice=next(iter(candidates)),
            status=PARSE_OK,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Answer successfully parsed from option text match",
        )
    elif len(candidates) > 1:
        return ParseResult(
            final_choice=None,
            status=PARSE_AMBIGUOUS,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Multiple conflicting option text matches found",
        )
    else:
        return ParseResult(
            final_choice=None,
            status=PARSE_MISSING,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Text present but no option text match found",
        )


def parse_model_answer(
    raw_text: str | None,
    options: Mapping[str, str],
) -> ParseResult:
    """Ported verbatim from twoprompt.parsing.parser.parse_model_answer.

    Note options is accepted (for the text-match fallback and to mirror the
    original signature) but is NOT used to restrict valid_choices for letter
    extraction — matching TSP's behavior of always allowing A-D regardless of
    how many real options the question has.
    """
    normalized_text = normalize_output_text(raw_text)
    temp_result = extract_choice_letter(normalized_text)
    if temp_result.status == PARSE_MISSING:
        temp_result = extract_choice_text_match(normalized_text, options)
    return ParseResult(
        final_choice=temp_result.final_choice,
        status=temp_result.status,
        raw_text=raw_text,
        normalized_text=temp_result.normalized_text,
        reason=temp_result.reason,
    )


def build_option_matching_prompt(
    template: str,
    question: str,
    free_text: str,
    option_a: object,
    option_b: object,
    option_c: object,
    option_d: object,
) -> str:
    """Ported verbatim from twoprompt.pipeline.prompt_builder.build_option_matching_prompt.

    Deliberately does NOT drop a missing option: str.format() renders a NaN
    float option_d as the literal text "nan" (e.g. "D. nan"), exactly
    matching the stored two_prompt Stage-2 prompts for the 3 historically
    3-option ARC-Challenge questions (see stage1_sources.py:
    KNOWN_3OPTION_ARC_QUESTION_IDS). Do not "fix" this to drop D — see the
    module docstring above for why that would break Stage-2 fidelity with
    the sibling two_prompt cell.
    """
    return template.format(
        question=question,
        free_text=free_text,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )
