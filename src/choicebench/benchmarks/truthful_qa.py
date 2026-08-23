# src/choicebench/benchmarks/truthful_qa.py

import logging

from choicebench.benchmarks.base import make_normalized_row
from choicebench.benchmarks.registry import benchmark

logger = logging.getLogger(__name__)

_SUBJECT = "truthful_qa"


@benchmark(
    name="truthful_qa",
    hf_path="truthfulqa/truthful_qa",
    hf_subset="multiple_choice",
    default_split="validation",
)
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

    # P3-F5: the spec allows exactly one gold label; a pathological source
    # row with several would resolve first-wins. Keep that resolution but
    # make it loud.
    gold_count = sum(1 for lbl in labels if int(lbl) == 1)
    if gold_count != 1:
        logger.warning(
            "TruthfulQA mc1 row has %d gold (label==1) choices (spec: exactly "
            "one); correct_index resolves to the first occurrence.",
            gold_count,
        )

    correct_index = next(i for i, lbl in enumerate(labels) if int(lbl) == 1)

    return make_normalized_row(
        subject=_SUBJECT,
        question_text=question,
        choices=choices,
        correct_index=correct_index,
    )
