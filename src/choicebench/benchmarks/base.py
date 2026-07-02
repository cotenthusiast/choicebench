# src/choicebench/benchmarks/base.py

import hashlib
from typing import Any, Callable, Sequence

import pandas as pd

from choicebench.constants import letters_for
from choicebench.pipeline.options import serialize_choices


def make_normalized_row(
    subject: str,
    question_text: str,
    choices: Sequence[str],
    correct_index: int,
    *,
    source_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Assemble one normalized row in the variable-choice schema.

    Shared by every benchmark normalizer so the schema (and the question_id
    hash) is produced in exactly one place. Labels (A, B, C, ...) are derived
    from choice order via letters_for(); ``source_index`` preserves each
    option's original position in the source dataset for audit.

    The question_id hash content is ``"{subject}|{question}|c0|c1|...|cN"``,
    which is byte-identical to the legacy 4-column hash for a 4-choice question
    — so existing normalized CSVs keep their question_ids.

    Args:
        subject: Subject/category label for the question.
        question_text: The question stem.
        choices: Ordered option texts (canonical render order).
        correct_index: 0-based index into ``choices`` of the correct option.
        source_indices: Original source positions per choice; defaults to
            0..len(choices)-1 when the source order is already canonical.

    Returns:
        A dict with the normalized schema columns.
    """
    n = len(choices)
    if n < 1:
        raise ValueError("A normalized question must have at least one choice.")
    if not (0 <= correct_index < n):
        raise ValueError(
            f"correct_index={correct_index} is out of range for {n} choices."
        )
    labels = letters_for(n)
    if source_indices is None:
        source_indices = list(range(n))

    choice_records = [
        {"label": labels[i], "text": str(choices[i]), "source_index": int(source_indices[i])}
        for i in range(n)
    ]

    content = f"{subject}|{question_text}|" + "|".join(str(c) for c in choices)
    question_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    return {
        "question_id": question_id,
        "subject": subject,
        "question_text": question_text,
        "choices_json": serialize_choices(choice_records),
        "correct_index": correct_index,
        "correct_option": labels[correct_index],
        "correct_answer_text": str(choices[correct_index]),
        "n_choices": n,
    }


def build_normalized_dataframe(
    df: pd.DataFrame,
    row_normalizer_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> pd.DataFrame:
    """Build a normalized DataFrame by applying a row normalizer to each row.

    Shared by every benchmark module (e.g. choicebench.benchmarks.mmlu,
    choicebench.benchmarks.arc) so each only needs to provide its own
    normalize_row function; the row-iteration and DataFrame assembly are
    identical across benchmarks.

    Args:
        df: Raw benchmark DataFrame.
        row_normalizer_fn: Function that converts one raw row dict into the
            project's normalized schema.

    Returns:
        DataFrame where each row follows the normalized schema produced by
        row_normalizer_fn.
    """
    rows = [row_normalizer_fn(row.to_dict()) for _, row in df.iterrows()]
    return pd.DataFrame(rows)
