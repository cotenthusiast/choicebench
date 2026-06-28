# src/choicebench/parsing/parser.py

from __future__ import annotations

from collections.abc import Collection, Mapping

from choicebench.constants import MCQ_OPTIONS
from choicebench.parsing.types import (
    PARSE_AMBIGUOUS,
    PARSE_MISSING,
    PARSE_OK,
    ParseResult,
)


DEFAULT_VALID_CHOICES: tuple[str, ...] = tuple(MCQ_OPTIONS)


def normalize_output_text(raw_text: str | None) -> str:
    """
    Normalize raw model output before parsing.

    Returns an empty string for None input. Otherwise strips leading and
    trailing whitespace and collapses repeated internal whitespace into
    single spaces, preserving content for both letter extraction and text
    matching.

    Args:
        raw_text: Raw model output text from a provider response.

    Returns:
        Normalized text string suitable for downstream parsing.
    """
    if raw_text is None or not isinstance(raw_text, str):
        return ""
    raw_text = raw_text.strip()
    if raw_text == "":
        return ""
    words = raw_text.split()
    normalized_string = " ".join(words)
    return normalized_string


def extract_choice_letter(
    normalized_text: str,
    valid_choices: Collection[str] = DEFAULT_VALID_CHOICES,
) -> ParseResult:
    """
    Attempt to extract a direct answer letter from normalized output.

    Extraction priority (later occurrences always win within a tier):
    1. Standalone single-token response.
    2. Last strong cue pattern: "final answer is X", "answer is X",
       "choice is X", "option is X", "final answer X", "answer X",
       "choice X", "option X", "therefore X", "thus X"  (case-insensitive).
    3. Last standalone A/B/C/D in the response when no strong cue matched.

    Using last-occurrence rather than collecting all candidates avoids
    false AMBIGUOUS results from reasoning models that enumerate options
    before stating a final answer.

    Args:
        normalized_text: Pre-normalized model output text.
        valid_choices: Allowed answer letters.

    Returns:
        ParseResult describing the letter-extraction attempt.
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

    # last_strong / last_weak: (word_index, letter) — updated as we scan left→right
    last_strong: tuple[int, str] | None = None
    last_weak: tuple[int, str] | None = None

    for i, w in enumerate(stripped):
        wl = w.lower()

        # "final answer is X"
        if (
            i + 3 < len(stripped)
            and wl == "final"
            and stripped[i + 1].lower() == "answer"
            and stripped[i + 2].lower() == "is"
            and stripped[i + 3].upper() in valid_choices
        ):
            last_strong = (i, stripped[i + 3].upper())

        # "answer is X" / "choice is X" / "option is X"
        if (
            i + 2 < len(stripped)
            and wl in cue_words
            and stripped[i + 1].lower() == "is"
            and stripped[i + 2].upper() in valid_choices
        ):
            last_strong = (i, stripped[i + 2].upper())

        # "final answer X"
        if (
            i + 2 < len(stripped)
            and wl == "final"
            and stripped[i + 1].lower() == "answer"
            and stripped[i + 2].upper() in valid_choices
        ):
            last_strong = (i, stripped[i + 2].upper())

        # "answer X" / "choice X" / "option X"
        if (
            i + 1 < len(stripped)
            and wl in cue_words
            and stripped[i + 1].upper() in valid_choices
        ):
            last_strong = (i, stripped[i + 1].upper())

        # "therefore X" / "thus X"
        if (
            i + 1 < len(stripped)
            and wl in concluding_words
            and stripped[i + 1].upper() in valid_choices
        ):
            last_strong = (i, stripped[i + 1].upper())

        # weak: any standalone valid letter
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
    """
    Fallback matching against the option texts themselves, for outputs where
    the model writes out the answer text instead of a letter.

    Each option's normalized text is checked against the normalized model
    output for an exact or substring match. Returns PARSE_OK if exactly one
    option matches, PARSE_AMBIGUOUS if more than one option matches, and
    PARSE_MISSING if none match.

    Args:
        normalized_text: Pre-normalized model output text.
        options: Mapping from answer letter to answer text.

    Returns:
        ParseResult describing the text-matching attempt.
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
            reason="Answer successfully parsed from option text match"
        )
    elif len(candidates) > 1:
        return ParseResult(
            final_choice=None,
            status=PARSE_AMBIGUOUS,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Multiple conflicting option text matches found"
        )
    else:
        return ParseResult(
            final_choice=None,
            status=PARSE_MISSING,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Text present but no option text match found"
        )


def parse_model_answer(
    raw_text: str | None,
    options: Mapping[str, str],
) -> ParseResult:
    """
    Parse a model's MCQ answer into a structured result.

    Normalizes the raw text, then tries direct letter extraction first; if
    that returns PARSE_MISSING, falls back to option-text matching. Never
    raises on junk, empty, or unexpected output.

    Args:
        raw_text: Raw text returned by the model.
        options: Mapping from answer letter to answer text.

    Returns:
        Final ParseResult for downstream scoring.
    """
    normalized_text = normalize_output_text(raw_text)
    temp_result = extract_choice_letter(normalized_text, valid_choices=tuple(options.keys()))
    if temp_result.status == PARSE_MISSING:
        temp_result = extract_choice_text_match(normalized_text, options)
    return ParseResult(
        final_choice=temp_result.final_choice,
        status=temp_result.status,
        raw_text=raw_text,
        normalized_text=temp_result.normalized_text,
        reason=temp_result.reason
    )
