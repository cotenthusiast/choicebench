from __future__ import annotations

import json
from typing import Any, Mapping

from choicebench.constants import MCQ_OPTIONS, letters_for

# Column that holds the variable-choice representation in the new schema. When
# present, it takes precedence over the legacy choice_a..choice_d columns.
CHOICES_JSON_COL = "choices_json"


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


def _has_choices_json(question_row: Mapping[str, Any]) -> bool:
    """True if the row carries a non-empty variable-choice `choices_json` value."""
    if CHOICES_JSON_COL not in question_row:
        return False
    raw = question_row[CHOICES_JSON_COL]
    if raw is None:
        return False
    # pandas encodes a missing cell as NaN (a float), which != itself.
    if isinstance(raw, float):
        try:
            if raw != raw:
                return False
        except Exception:
            return False
    if isinstance(raw, str) and raw.strip() == "":
        return False
    return True


def parse_choices_json(raw: object) -> list[dict[str, Any]]:
    """Parse a `choices_json` value into a list of {"text", "source_index"} dicts.

    Accepts either a JSON string (as read from a CSV) or an already-decoded
    list (as passed in an in-memory dict row).
    """
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    if not isinstance(parsed, list):
        raise ValueError(f"{CHOICES_JSON_COL} must decode to a list; got {type(parsed).__name__}.")
    return parsed


def build_choices(question_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical, ordered choice records for one question row.

    Each record is ``{"label", "text", "source_index"}``:
      - ``label``: the render-time letter (A, B, C, ...), derived from choice
        *order*, never persisted from source.
      - ``text``: normalized option text.
      - ``source_index``: the option's original index in the raw source dataset,
        preserved for audit.

    Supports both schemas:
      - New: a ``choices_json`` column (list of {text, source_index}). Labels are
        re-derived from position via letters_for().
      - Legacy: ``choice_a..choice_d`` columns. Each column keeps its fixed
        letter as the label (A=choice_a, ...) and its column index as
        source_index; missing/empty columns are dropped.
    """
    if _has_choices_json(question_row):
        raw = parse_choices_json(question_row[CHOICES_JSON_COL])
        labels = letters_for(len(raw)) if raw else []
        choices: list[dict[str, Any]] = []
        for pos, (item, label) in enumerate(zip(raw, labels)):
            text = normalize_option_text(item.get("text"))
            source_index = item.get("source_index", pos)
            choices.append(
                {"label": label, "text": text, "source_index": int(source_index)}
            )
        return choices

    # Legacy path: fixed choice_a..choice_d columns.
    choices = []
    for i, label in enumerate(MCQ_OPTIONS):
        text = normalize_option_text(question_row.get(f"choice_{label.lower()}"))
        if text:
            choices.append({"label": label, "text": text, "source_index": i})
    return choices


def build_option_map(question_row: Mapping[str, Any]) -> dict[str, str]:
    """Build and validate the label→text option map for one normalized row.

    Missing, NaN, and empty-string choices are dropped. Works for both the new
    variable-choice schema and the legacy choice_a..choice_d schema (see
    build_choices). Downstream prompt rendering and parsing only see real
    options.

    Raises:
        ValueError: if fewer than 2 valid options remain, or if the row's
            correct answer does not point at one of them.
    """
    options = {c["label"]: c["text"] for c in build_choices(question_row) if c["text"]}

    if len(options) < 2:
        qid = question_row.get("question_id", "<unknown>")
        raise ValueError(
            f"Question {qid!r} has fewer than 2 valid answer options after "
            "dropping missing/empty choices."
        )

    correct_option = correct_option_for_row(question_row, options)
    if correct_option not in options:
        qid = question_row.get("question_id", "<unknown>")
        raise ValueError(
            f"Question {qid!r} has correct_option={correct_option!r}, but valid "
            f"options are {list(options)} after dropping missing/empty choices."
        )

    return options


def correct_option_for_row(
    question_row: Mapping[str, Any],
    options: dict[str, str] | None = None,
) -> str:
    """Derive the correct answer *letter* for a row.

    Prefers a persisted ``correct_option`` letter (present in both schemas). If
    absent, falls back to deriving it from ``correct_index`` against the built
    label order — so a new-schema CSV authored with only correct_index still
    resolves correctly.
    """
    raw = question_row.get("correct_option")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().upper()

    correct_index = question_row.get("correct_index")
    if correct_index is None:
        return ""
    labels = [c["label"] for c in build_choices(question_row)]
    idx = int(correct_index)
    if 0 <= idx < len(labels):
        return labels[idx]
    return ""


def serialize_choices(choices: list[dict[str, Any]]) -> str:
    """Serialize choice records to the persisted `choices_json` string form.

    Only ``text`` and ``source_index`` are stored; labels are always re-derived
    from order on read.
    """
    return json.dumps(
        [{"text": c["text"], "source_index": int(c["source_index"])} for c in choices]
    )
