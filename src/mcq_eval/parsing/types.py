# src/mcq_eval/parsing/types.py

from __future__ import annotations

from dataclasses import dataclass


PARSE_OK = "parse_ok"
PARSE_INVALID = "parse_invalid"
PARSE_MISSING = "parse_missing"
PARSE_AMBIGUOUS = "parse_ambiguous"

PARSE_STATUSES = {
    PARSE_OK,
    PARSE_INVALID,
    PARSE_MISSING,
    PARSE_AMBIGUOUS,
}


@dataclass(frozen=True, slots=True)
class ParsedChoice:
    """
    Represents a single parsed answer candidate.

    Attributes:
        letter: Canonical answer letter such as "A", "B", "C", or "D".
        matched_text: Optional option text that matched the model output.
    """

    letter: str
    matched_text: str | None = None


@dataclass(frozen=True, slots=True)
class ParseResult:
    """
    Structured result returned by parsing functions.

    Attributes:
        final_choice: Final canonical answer letter if parsing succeeded.
        status: Parse outcome status constant.
        raw_text: Original raw model output before normalization.
        normalized_text: Cleaned text used by downstream parsing logic.
        reason: Optional human-readable explanation for the parse outcome.
    """

    final_choice: str | None
    status: str
    raw_text: str | None
    normalized_text: str
    reason: str | None = None
