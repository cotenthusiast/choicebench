# src/mcq_eval/benchmarks/mmlu.py

import ast
import hashlib

import pandas as pd

from mcq_eval.benchmarks.base import build_normalized_dataframe as _build_normalized_dataframe
from mcq_eval.config.experiment import MCQ_ANSWER_MAP


def normalize_row(row: dict[str, object]) -> dict[str, object]:
    """
    Converts one raw MMLU row into the project's normalized schema.

    Args:
        row: Dictionary containing the raw MMLU fields:
            - subject
            - question
            - choices
            - answer

    Returns:
        Dictionary containing the normalized question fields:
            - question_id
            - subject
            - question_text
            - choice_a
            - choice_b
            - choice_c
            - choice_d
            - correct_option
            - correct_answer_text
    """
    subject, question, choices, answer = row["subject"], row["question"], row["choices"], row["answer"]
    parsed_choices = ast.literal_eval(choices)
    choice_a, choice_b, choice_c, choice_d = parsed_choices[0], parsed_choices[1], parsed_choices[2], parsed_choices[3]
    answer = MCQ_ANSWER_MAP[answer]
    content = f"{subject}|{question}|{choice_a}|{choice_b}|{choice_c}|{choice_d}"
    question_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    if answer == "A":
        correct_answer_text = choice_a
    elif answer == "B":
        correct_answer_text = choice_b
    elif answer == "C":
        correct_answer_text = choice_c
    else:
        correct_answer_text = choice_d
    return ({
        "question_id": question_id,
        "subject": subject,
        "question_text": question,
        "choice_a": choice_a,
        "choice_b": choice_b,
        "choice_c": choice_c,
        "choice_d": choice_d,
        "correct_option": answer,
        "correct_answer_text": correct_answer_text,
    })


def build_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a normalized dataframe from a raw MMLU dataframe.

    Each raw row is converted into the project's normalized schema
    by calling normalize_row.

    Args:
        df: Raw MMLU dataframe containing the columns:
            - subject
            - question
            - choices
            - answer

    Returns:
        Pandas dataframe where each row follows the normalized schema.
    """
    return _build_normalized_dataframe(df, normalize_row)
