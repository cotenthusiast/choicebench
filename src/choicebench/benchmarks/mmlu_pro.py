# src/choicebench/benchmarks/mmlu_pro.py

import math

from choicebench.benchmarks.base import make_normalized_row
from choicebench.benchmarks.registry import benchmark


@benchmark(
    name="mmlu_pro",
    hf_path="TIGER-Lab/MMLU-Pro",
    hf_subset=None,
    default_split="test",
)
def normalize_row(row: dict[str, object]) -> dict[str, object]:
    """Convert one raw MMLU-Pro row into the project's normalized schema.

    MMLU-Pro (TIGER-Lab/MMLU-Pro) items arrive with:
      - question:      question text
      - options:       list of up to 10 answer strings
      - answer:        letter string e.g. "A" (may refer to options beyond D)
      - answer_index:  0-based int index of the correct option
      - category:      subject/category string
      - src:           source tag (not used)

    Every non-null option is preserved faithfully — the full choice set
    (up to 10) is kept and no row is dropped based on answer_index. Render
    labels (A..J) are re-derived from choice order by make_normalized_row.

    Null-option policy (F5): a null/NaN option text must never silently
    become the literal string "nan". A null DISTRACTOR is dropped under this
    explicit tested policy (later options shift left, and answer_index is
    remapped so the correct answer is preserved); a null GOLD option rejects
    the item outright — there is no valid question without an answer.

    Args:
        row: Dictionary with the raw MMLU-Pro fields listed above.

    Returns:
        Dictionary with the normalized (variable-choice) schema fields.

    Raises:
        ValueError: If the gold option itself is null, or no valid options
            remain.
    """
    category = str(row["category"])
    question = str(row["question"])
    answer_index = int(row["answer_index"])

    def _is_null(value: object) -> bool:
        if value is None:
            return True
        return isinstance(value, float) and math.isnan(value)

    raw_options = list(row["options"])
    options: list[str] = []
    kept_before_gold = 0
    for i, opt in enumerate(raw_options):
        if _is_null(opt):
            if i == answer_index:
                raise ValueError(
                    "MMLU-Pro row has a null gold option (answer_index="
                    f"{answer_index}); rejecting item instead of emitting "
                    f"'nan' as the correct answer (question={question!r})."
                )
            continue
        if i < answer_index:
            kept_before_gold += 1
        options.append(str(opt))

    return make_normalized_row(
        subject=category,
        question_text=question,
        choices=options,
        correct_index=kept_before_gold,
    )
