# src/choicebench/benchmarks/mmlu_pro.py

import hashlib
import logging

import pandas as pd

from choicebench.benchmarks.registry import benchmark
from choicebench.constants import MCQ_ANSWER_MAP

logger = logging.getLogger(__name__)


def normalize_row(row: dict[str, object]) -> dict[str, object] | None:
    """Convert one raw MMLU-Pro row into the project's normalized schema.

    MMLU-Pro (TIGER-Lab/MMLU-Pro) items arrive with:
      - question:      question text
      - options:       list of up to 10 answer strings
      - answer:        letter string e.g. "A" (may refer to options beyond D)
      - answer_index:  0-based int index of the correct option
      - category:      subject/category string
      - src:           source tag (not used)

    ChoiceBench normalizes to exactly 4 options (A–D). Only the first 4 options
    are kept. If answer_index > 3 the correct answer does not appear among the
    retained options; those rows are skipped and None is returned. Callers
    (build_normalized_dataframe) must filter out None returns.

    Limitation: approximately 50 % of MMLU-Pro questions have correct answers
    beyond index 3 and will be dropped during normalization. This is a known
    consequence of forcing a 10-option benchmark into the 4-option schema.

    Args:
        row: Dictionary with the raw MMLU-Pro fields listed above.

    Returns:
        Dictionary with the normalized question fields, or None if the correct
        answer falls outside the first 4 options.
    """
    category = str(row["category"])
    question = str(row["question"])
    options = list(row["options"])
    answer_index = int(row["answer_index"])

    if answer_index > 3:
        logger.warning(
            "Skipping MMLU-Pro question (answer_index=%d > 3): %.60s",
            answer_index,
            question,
        )
        return None

    choice_a = str(options[0]) if len(options) > 0 else ""
    choice_b = str(options[1]) if len(options) > 1 else ""
    choice_c = str(options[2]) if len(options) > 2 else ""
    choice_d = str(options[3]) if len(options) > 3 else ""

    correct_option = MCQ_ANSWER_MAP[answer_index]
    correct_texts = {"A": choice_a, "B": choice_b, "C": choice_c, "D": choice_d}
    correct_answer_text = correct_texts[correct_option]

    content = f"{category}|{question}|{choice_a}|{choice_b}|{choice_c}|{choice_d}"
    question_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    return {
        "question_id": question_id,
        "subject": category,
        "question_text": question,
        "choice_a": choice_a,
        "choice_b": choice_b,
        "choice_c": choice_c,
        "choice_d": choice_d,
        "correct_option": correct_option,
        "correct_answer_text": correct_answer_text,
    }


@benchmark(
    name="mmlu_pro",
    hf_path="TIGER-Lab/MMLU-Pro",
    hf_subset=None,
    default_split="test",
)
def build_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build a normalized DataFrame from a raw MMLU-Pro DataFrame.

    Rows where the correct answer falls outside the first 4 options are
    silently dropped (a warning is logged per dropped row via normalize_row).

    Args:
        df: Raw MMLU-Pro DataFrame with columns: question, options,
            answer, answer_index, category, src.

    Returns:
        DataFrame where each row follows the normalized schema.
    """
    rows = []
    for _, row in df.iterrows():
        result = normalize_row(row.to_dict())
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)
