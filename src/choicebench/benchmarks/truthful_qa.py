# src/choicebench/benchmarks/truthful_qa.py

import pandas as pd

from choicebench.benchmarks.base import build_normalized_dataframe as _build_normalized_dataframe
from choicebench.benchmarks.base import make_normalized_row
from choicebench.benchmarks.registry import benchmark

_SUBJECT = "truthful_qa"


def normalize_row(row: dict[str, object]) -> dict[str, object]:
    """Convert one raw TruthfulQA row into the project's normalized schema.

    TruthfulQA (truthful_qa, multiple_choice subset) items arrive with:
      - question:      question text
      - mc1_targets:   dict with "choices" (list[str]) and "labels" (list[int]),
                       where exactly one label is 1 (the correct answer). Across
                       the full validation split the correct answer is always the
                       first-listed choice (index 0).
      - mc2_targets:   dict (not used — mc2 has multiple correct answers)

    Only mc1_targets is used (single-correct-answer format). Every choice is
    preserved — raw mc1 choice counts range 2..13, and nothing is truncated or
    dropped. correct_index is derived from the labels array (exactly one label
    is 1). This resolves to 0 for every row in the current dataset (the mc1
    correct answer is always listed first, verified across all 817 validation
    rows), but self-corrects if a future row ever lists the gold answer
    elsewhere rather than silently mislabeling it.

    Subject is set to the constant string "truthful_qa" for all rows.

    Args:
        row: Dictionary with the raw TruthfulQA fields listed above.

    Returns:
        Dictionary with the normalized question fields.
    """
    question = str(row["question"])
    mc1 = row["mc1_targets"]
    choices = [str(c) for c in mc1["choices"]]
    labels = list(mc1["labels"])

    correct_index = next(i for i, lbl in enumerate(labels) if int(lbl) == 1)

    return make_normalized_row(
        subject=_SUBJECT,
        question_text=question,
        choices=choices,
        correct_index=correct_index,
    )


@benchmark(
    name="truthful_qa",
    hf_path="truthfulqa/truthful_qa",
    hf_subset="multiple_choice",
    default_split="validation",
)
def build_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build a normalized DataFrame from a raw TruthfulQA DataFrame.

    Every row is kept with its full mc1 choice set intact (no truncation, no
    skipping).

    Args:
        df: Raw TruthfulQA DataFrame with columns: question, mc1_targets,
            mc2_targets.

    Returns:
        DataFrame where each row follows the normalized schema.
    """
    return _build_normalized_dataframe(df, normalize_row)
