# src/choicebench/benchmarks/mmlu_pro.py

import pandas as pd

from choicebench.benchmarks.base import build_normalized_dataframe as _build_normalized_dataframe
from choicebench.benchmarks.base import make_normalized_row
from choicebench.benchmarks.registry import benchmark


def normalize_row(row: dict[str, object]) -> dict[str, object]:
    """Convert one raw MMLU-Pro row into the project's normalized schema.

    MMLU-Pro (TIGER-Lab/MMLU-Pro) items arrive with:
      - question:      question text
      - options:       list of up to 10 answer strings
      - answer:        letter string e.g. "A" (may refer to options beyond D)
      - answer_index:  0-based int index of the correct option
      - category:      subject/category string
      - src:           source tag (not used)

    Every option is preserved faithfully — the full choice set (up to 10) is
    kept and no row is dropped based on answer_index. Render labels (A..J) are
    re-derived from choice order by make_normalized_row.

    Args:
        row: Dictionary with the raw MMLU-Pro fields listed above.

    Returns:
        Dictionary with the normalized (variable-choice) schema fields.
    """
    category = str(row["category"])
    question = str(row["question"])
    options = [str(o) for o in row["options"]]
    answer_index = int(row["answer_index"])

    return make_normalized_row(
        subject=category,
        question_text=question,
        choices=options,
        correct_index=answer_index,
    )


@benchmark(
    name="mmlu_pro",
    hf_path="TIGER-Lab/MMLU-Pro",
    hf_subset=None,
    default_split="test",
)
def build_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build a normalized DataFrame from a raw MMLU-Pro DataFrame.

    Every question is kept with its full option set intact — MMLU-Pro's up-to-10
    options survive normalization unchanged.

    Args:
        df: Raw MMLU-Pro DataFrame with columns: question, options,
            answer, answer_index, category, src.

    Returns:
        DataFrame where each row follows the normalized schema.
    """
    return _build_normalized_dataframe(df, normalize_row)
