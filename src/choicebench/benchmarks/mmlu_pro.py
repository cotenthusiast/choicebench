# src/choicebench/benchmarks/mmlu_pro.py

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
