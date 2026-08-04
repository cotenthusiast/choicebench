# src/choicebench/parsing/parser.py

from __future__ import annotations

import re
from collections.abc import Collection, Mapping

from choicebench.constants import LEGACY_OPTION_LETTERS
from choicebench.parsing.types import (
    PARSE_AMBIGUOUS,
    PARSE_MISSING,
    PARSE_OK,
    ParseResult,
)


DEFAULT_VALID_CHOICES: tuple[str, ...] = tuple(LEGACY_OPTION_LETTERS)

_CUE_WORDS = {"answer", "choice", "option"}
_CONCLUDING_WORDS = {"therefore", "thus"}


def _match_final_answer_is(
    stripped: list[str], i: int, wl: str, valid_choices: Collection[str]
) -> str | None:
    """"final answer is X"."""
    if (
        i + 3 < len(stripped)
        and wl == "final"
        and stripped[i + 1].lower() == "answer"
        and stripped[i + 2].lower() == "is"
        and stripped[i + 3].upper() in valid_choices
    ):
        return stripped[i + 3].upper()
    return None


def _match_cue_is(
    stripped: list[str], i: int, wl: str, valid_choices: Collection[str]
) -> str | None:
    """"answer is X" / "choice is X" / "option is X"."""
    if (
        i + 2 < len(stripped)
        and wl in _CUE_WORDS
        and stripped[i + 1].lower() == "is"
        and stripped[i + 2].upper() in valid_choices
    ):
        return stripped[i + 2].upper()
    return None


def _match_final_answer(
    stripped: list[str], i: int, wl: str, valid_choices: Collection[str]
) -> str | None:
    """"final answer X"."""
    if (
        i + 2 < len(stripped)
        and wl == "final"
        and stripped[i + 1].lower() == "answer"
        and stripped[i + 2].upper() in valid_choices
    ):
        return stripped[i + 2].upper()
    return None


def _match_cue(
    stripped: list[str], i: int, wl: str, valid_choices: Collection[str]
) -> str | None:
    """"answer X" / "choice X" / "option X"."""
    if (
        i + 1 < len(stripped)
        and wl in _CUE_WORDS
        and stripped[i + 1].upper() in valid_choices
    ):
        return stripped[i + 1].upper()
    return None


def _match_concluding(
    stripped: list[str], i: int, wl: str, valid_choices: Collection[str]
) -> str | None:
    """"therefore X" / "thus X"."""
    if (
        i + 1 < len(stripped)
        and wl in _CONCLUDING_WORDS
        and stripped[i + 1].upper() in valid_choices
    ):
        return stripped[i + 1].upper()
    return None


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

    # last_strong / last_weak: (word_index, letter) — updated as we scan left→right
    last_strong: tuple[int, str] | None = None
    last_weak: tuple[int, str] | None = None

    for i, w in enumerate(stripped):
        wl = w.lower()

        letter = _match_final_answer_is(stripped, i, wl, valid_choices)
        if letter is not None:
            last_strong = (i, letter)

        letter = _match_cue_is(stripped, i, wl, valid_choices)
        if letter is not None:
            last_strong = (i, letter)

        letter = _match_final_answer(stripped, i, wl, valid_choices)
        if letter is not None:
            last_strong = (i, letter)

        letter = _match_cue(stripped, i, wl, valid_choices)
        if letter is not None:
            last_strong = (i, letter)

        letter = _match_concluding(stripped, i, wl, valid_choices)
        if letter is not None:
            last_strong = (i, letter)

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


def _option_matches_as_token(normalized_option: str, normalized_text: str) -> bool:
    """
    Return True if the option text appears in the model output as a distinct
    token sequence rather than as an incidental substring.

    Matching is bounded by non-alphanumeric characters (or the string edges),
    so option ``"4"`` matches the standalone ``4`` in ``"the answer is 4"`` but
    not the ``4`` inside ``"14"``. Comparison is case-insensitive.
    """
    if normalized_option == "":
        return False
    pattern = (
        r"(?<![A-Za-z0-9])"
        + re.escape(normalized_option)
        + r"(?![A-Za-z0-9])"
    )
    return re.search(pattern, normalized_text, flags=re.IGNORECASE) is not None


def extract_choice_text_match(
    normalized_text: str,
    options: Mapping[str, str],
) -> ParseResult:
    """
    Fallback matching against the option texts themselves, for outputs where
    the model writes out the answer text instead of a letter.

    Each option's normalized text is matched against the normalized model
    output on token boundaries (see _option_matches_as_token): the option must
    appear as a distinct token sequence, not merely as a raw substring, so a
    short/numeric option like ``"4"`` does not spuriously match inside ``"14"``.
    Returns PARSE_OK if exactly one option matches, PARSE_AMBIGUOUS if more than
    one option matches, and PARSE_MISSING if none match.

    Args:
        normalized_text: Pre-normalized model output text.
        options: Mapping from answer letter to answer text.

    Returns:
        ParseResult describing the text-matching attempt.
    """
    candidates = []
    for letter, option_text in options.items():
        normalized_option = normalize_output_text(option_text)
        if _option_matches_as_token(normalized_option, normalized_text):
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
