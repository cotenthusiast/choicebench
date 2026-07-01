# src/choicebench/benchmarks/truthful_qa.py

import hashlib
import logging

import pandas as pd

from choicebench.benchmarks.registry import benchmark
from choicebench.constants import MCQ_ANSWER_MAP

logger = logging.getLogger(__name__)

_SUBJECT = "truthful_qa"


def normalize_row(row: dict[str, object]) -> dict[str, object] | None:
    """Convert one raw TruthfulQA row into the project's normalized schema.

    TruthfulQA (truthful_qa, multiple_choice subset) items arrive with:
      - question:      question text
      - mc1_targets:   dict with "choices" (list[str]) and "labels" (list[int]),
                       where exactly one label is 1 (the correct answer)
      - mc2_targets:   dict (not used — mc2 has multiple correct answers)

    Only mc1_targets is used (single-correct-answer format). Only the first 4
    choices are kept. If the correct answer (label == 1) does not appear among
    the first 4 choices, the row is skipped and None is returned.

    Subject is set to the constant string "truthful_qa" for all rows.

    Args:
        row: Dictionary with the raw TruthfulQA fields listed above.

    Returns:
        Dictionary with the normalized question fields, or None if the correct
        answer falls outside the first 4 choices.
    """
    question = str(row["question"])
    mc1 = row["mc1_targets"]
    choices = list(mc1["choices"])
    labels = list(mc1["labels"])

    correct_idx = next(i for i, lbl in enumerate(labels) if int(lbl) == 1)

    if correct_idx > 3:
        logger.warning(
            "Skipping TruthfulQA question (correct_idx=%d > 3): %.60s",
            correct_idx,
            question,
        )
        return None

    choice_a = str(choices[0]) if len(choices) > 0 else ""
    choice_b = str(choices[1]) if len(choices) > 1 else ""
    choice_c = str(choices[2]) if len(choices) > 2 else ""
    choice_d = str(choices[3]) if len(choices) > 3 else ""

    correct_option = MCQ_ANSWER_MAP[correct_idx]
    correct_texts = {"A": choice_a, "B": choice_b, "C": choice_c, "D": choice_d}
    correct_answer_text = correct_texts[correct_option]

    content = f"{_SUBJECT}|{question}|{choice_a}|{choice_b}|{choice_c}|{choice_d}"
    question_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    return {
        "question_id": question_id,
        "subject": _SUBJECT,
        "question_text": question,
        "choice_a": choice_a,
        "choice_b": choice_b,
        "choice_c": choice_c,
        "choice_d": choice_d,
        "correct_option": correct_option,
        "correct_answer_text": correct_answer_text,
    }


@benchmark(
    name="truthful_qa",
    hf_path="truthfulqa/truthful_qa",
    hf_subset="multiple_choice",
    default_split="validation",
)
def build_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build a normalized DataFrame from a raw TruthfulQA DataFrame.

    Rows where the correct answer falls outside the first 4 choices are
    silently dropped (a warning is logged per dropped row via normalize_row).

    Args:
        df: Raw TruthfulQA DataFrame with columns: question, mc1_targets,
            mc2_targets.

    Returns:
        DataFrame where each row follows the normalized schema.
    """
    rows = []
    for _, row in df.iterrows():
        result = normalize_row(row.to_dict())
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)
