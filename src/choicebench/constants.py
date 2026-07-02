# src/choicebench/constants.py

import string

# Legacy fixed A-D option set. Kept as the default for backwards compatibility
# (parser defaults, 4-option PriDe path). New code that needs to support a
# variable number of choices should derive its label set with letters_for(n).
MCQ_OPTIONS: list[str] = ["A", "B", "C", "D"]

MCQ_ANSWER_MAP: dict[int, str] = {i: letter for i, letter in enumerate(MCQ_OPTIONS)}

# Full A-Z label alphabet. MCQ answer labels are single letters, so this caps a
# question at 26 options — far above the 10-option maximum of MMLU-Pro.
ALL_OPTION_LETTERS: list[str] = list(string.ascii_uppercase)


def letters_for(n: int) -> list[str]:
    """Return the first ``n`` canonical option labels (A, B, C, ...).

    Labels are always derived from choice *order* this way, never trusted from
    a source dataset (which is inconsistent past J). Raises ValueError for
    n < 1 or n > 26 (labels are single letters).
    """
    if n < 1 or n > len(ALL_OPTION_LETTERS):
        raise ValueError(
            f"letters_for(n) supports 1..{len(ALL_OPTION_LETTERS)} options; got n={n}."
        )
    return ALL_OPTION_LETTERS[:n]
