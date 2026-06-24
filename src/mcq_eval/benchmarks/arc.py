# src/mcq_eval/benchmarks/arc.py

import hashlib

import pandas as pd

from mcq_eval.benchmarks.base import build_normalized_dataframe as _build_normalized_dataframe

_ARC_SUBJECT = "arc_challenge"
_VALID_LABELS = {"A", "B", "C", "D"}


def normalize_row(row: dict[str, object]) -> dict[str, object]:
    """Convert one raw ARC-Challenge row into the project's normalized schema.

    ARC-Challenge items from HuggingFace (allenai/ai2_arc) arrive with:
      - id:         string identifier
      - question:   question text
      - choices:    dict with "text" (list[str]) and "label" (list[str])
      - answerKey:  "A", "B", "C", or "D"  (occasionally "1"-"4" in older splits)

    The output schema matches the MMLU normalized schema so that all downstream
    runners, parsers, and evaluation code can handle both benchmarks identically.

    Args:
        row: Dictionary with the raw ARC fields listed above.

    Returns:
        Dictionary with the normalized question fields:
            question_id, subject, question_text,
            choice_a, choice_b, choice_c, choice_d,
            correct_option, correct_answer_text
    """
    question_id_raw = str(row["id"])
    question = str(row["question"])
    choices_raw = row["choices"]

    labels: list[str] = list(choices_raw["label"])
    texts: list[str] = list(choices_raw["text"])

    label_to_text: dict[str, str] = dict(zip(labels, texts))

    # Normalize numeric labels ("1"-"4") to letter labels if present.
    _num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}
    if not _VALID_LABELS.intersection(label_to_text):
        label_to_text = {
            _num_to_letter.get(k, k): v for k, v in label_to_text.items()
        }

    choice_a = label_to_text.get("A", "")
    choice_b = label_to_text.get("B", "")
    choice_c = label_to_text.get("C", "")
    choice_d = label_to_text.get("D", "")

    answer_key = str(row["answerKey"]).strip().upper()
    answer_key = _num_to_letter.get(answer_key, answer_key)

    correct_answer_text = label_to_text.get(answer_key, "")

    content = f"{_ARC_SUBJECT}|{question}|{choice_a}|{choice_b}|{choice_c}|{choice_d}"
    question_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    return {
        "question_id": question_id,
        "subject": _ARC_SUBJECT,
        "question_text": question,
        "choice_a": choice_a,
        "choice_b": choice_b,
        "choice_c": choice_c,
        "choice_d": choice_d,
        "correct_option": answer_key,
        "correct_answer_text": correct_answer_text,
    }


def build_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build a normalized DataFrame from a raw ARC-Challenge DataFrame.

    Args:
        df: Raw ARC DataFrame with columns: id, question, choices, answerKey.

    Returns:
        DataFrame where each row follows the normalized schema.
    """
    return _build_normalized_dataframe(df, normalize_row)
