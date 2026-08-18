# src/choicebench/benchmarks/hellaswag.py

from choicebench.benchmarks.base import make_normalized_row
from choicebench.benchmarks.registry import benchmark


@benchmark(
    name="hellaswag",
    hf_path="Rowan/hellaswag",
    hf_subset=None,
    default_split="validation",
)
def normalize_row(row: dict[str, object]) -> dict[str, object]:
    """Convert one raw HellaSwag row into the project's normalized schema.

    HellaSwag (Rowan/hellaswag) items arrive with:
      - ind:            string identifier
      - activity_label: activity/subject string
      - ctx:            context string used as the question stem
      - endings:        list of exactly 4 completion strings
      - label:          int 0–3 indicating the correct completion

    The ctx field becomes question_text; the four endings become the
    ordered choices list (choices_json). subject is set from activity_label.

    Note: HellaSwag labels are in the "validation" split only — the "test"
    split has no labels. Always prepare with --split validation.

    Args:
        row: Dictionary with the raw HellaSwag fields listed above.

    Returns:
        Dictionary with the normalized question fields.
    """
    activity_label = str(row["activity_label"])
    ctx = str(row["ctx"])
    endings = [str(e) for e in row["endings"]]
    label = int(row["label"])

    return make_normalized_row(
        subject=activity_label,
        question_text=ctx,
        choices=endings,
        correct_index=label,
    )
