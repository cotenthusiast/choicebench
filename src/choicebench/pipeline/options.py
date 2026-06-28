from __future__ import annotations

from typing import Any, Mapping

from choicebench.constants import MCQ_OPTIONS


def normalize_option_text(value: object) -> str:
    """Return normalized option text, treating None/NaN/non-strings as missing."""
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            if value != value:
                return ""
        except Exception:
            return ""
        value = str(value)
    return " ".join(value.strip().split())


def build_option_map(question_row: Mapping[str, Any]) -> dict[str, str]:
    """Build and validate the legacy A-D option map for one normalized row.

    Missing, NaN, and empty-string choices are dropped. The existing legacy
    schema is preserved: callers still read choice_a through choice_d, but
    downstream prompt rendering and parsing only see real options.
    """
    options: dict[str, str] = {}
    for label in MCQ_OPTIONS:
        field = f"choice_{label.lower()}"
        text = normalize_option_text(question_row.get(field))
        if text:
            options[label] = text

    if len(options) < 2:
        qid = question_row.get("question_id", "<unknown>")
        raise ValueError(
            f"Question {qid!r} has fewer than 2 valid answer options after "
            "dropping missing/empty choices."
        )

    correct_option = str(question_row.get("correct_option", "")).strip().upper()
    if correct_option not in options:
        qid = question_row.get("question_id", "<unknown>")
        raise ValueError(
            f"Question {qid!r} has correct_option={correct_option!r}, but valid "
            f"options are {list(options)} after dropping missing/empty choices."
        )

    return options
