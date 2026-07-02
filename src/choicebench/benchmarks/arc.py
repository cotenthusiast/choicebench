# src/choicebench/benchmarks/arc.py

import pandas as pd

from choicebench.benchmarks.base import build_normalized_dataframe as _build_normalized_dataframe
from choicebench.benchmarks.base import make_normalized_row
from choicebench.benchmarks.registry import benchmark

_ARC_SUBJECT = "arc_challenge"
_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def normalize_row(row: dict[str, object]) -> dict[str, object]:
    """Convert one raw ARC-Challenge row into the project's normalized schema.

    ARC-Challenge items from HuggingFace (allenai/ai2_arc) arrive with:
      - id:         string identifier
      - question:   question text
      - choices:    dict with "text" (list[str]) and "label" (list[str])
      - answerKey:  "A", "B", "C", or "D"  (occasionally "1"-"4" in older splits)

    Choices are kept in their source order; render labels (A, B, C, ...) are
    re-derived from that order. Most ARC items have 4 options but some have 3
    or 5 — all are preserved.

    Args:
        row: Dictionary with the raw ARC fields listed above.

    Returns:
        Dictionary with the normalized (variable-choice) schema fields.
    """
    question = str(row["question"])
    choices_raw = row["choices"]

    labels: list[str] = [str(x).strip().upper() for x in choices_raw["label"]]
    texts: list[str] = [str(x) for x in choices_raw["text"]]

    # Normalize numeric labels ("1"-"5") to letters for matching answerKey.
    norm_labels = [_NUM_TO_LETTER.get(lbl, lbl) for lbl in labels]

    answer_key = str(row["answerKey"]).strip().upper()
    answer_key = _NUM_TO_LETTER.get(answer_key, answer_key)

    correct_index = norm_labels.index(answer_key)

    return make_normalized_row(
        subject=_ARC_SUBJECT,
        question_text=question,
        choices=texts,
        correct_index=correct_index,
    )


@benchmark(
    name="arc_challenge",
    hf_path="allenai/ai2_arc",
    hf_subset="ARC-Challenge",
    default_split="test",
)
def build_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build a normalized DataFrame from a raw ARC-Challenge DataFrame.

    Args:
        df: Raw ARC DataFrame with columns: id, question, choices, answerKey.

    Returns:
        DataFrame where each row follows the normalized schema.
    """
    return _build_normalized_dataframe(df, normalize_row)
