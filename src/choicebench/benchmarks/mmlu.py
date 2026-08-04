# src/choicebench/benchmarks/mmlu.py

import ast
from collections.abc import Iterable

from choicebench.benchmarks.base import make_normalized_row
from choicebench.benchmarks.registry import benchmark


def _parse_choices(choices: object) -> list[str]:
    """Parse MMLU choices from HF/Pandas-native and legacy string formats."""
    parsed_choices = ast.literal_eval(choices) if isinstance(choices, str) else choices

    if hasattr(parsed_choices, "tolist"):
        parsed_choices = parsed_choices.tolist()

    if isinstance(parsed_choices, Iterable) and not isinstance(parsed_choices, str | bytes | dict):
        choices_list = list(parsed_choices)
    else:
        choices_list = [parsed_choices]

    if len(choices_list) != 4:
        raise ValueError(
            "MMLU choices must contain exactly 4 choices; "
            f"found {len(choices_list)} choices: {choices_list!r}"
        )

    return [str(choice) for choice in choices_list]


@benchmark(
    name="mmlu",
    hf_path="cais/mmlu",
    hf_subset="all",
    default_split="test",
)
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
            - choices_json
            - correct_index
            - correct_option
            - correct_answer_text
            - n_choices
    """
    subject, question, choices, answer = row["subject"], row["question"], row["choices"], row["answer"]
    parsed_choices = _parse_choices(choices)
    return make_normalized_row(
        subject=str(subject),
        question_text=str(question),
        choices=parsed_choices,
        correct_index=int(answer),
    )
