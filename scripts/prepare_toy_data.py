"""Generate the toy dataset used by config/toy_experiment.yaml.

Deterministically (re)produces 10 synthetic MCQ rows in the same normalized
schema as the real benchmarks (question_id, subject, question_text,
choice_a..d, correct_option), so the onboarding toy config has reproducible
data instead of a hand-committed CSV.

3 of the 10 rows are set up so the correct answer lands in option A — this
matches DummyBackend's default fixed_text ("The answer is A."), which the
parser resolves to letter A every time. Running the toy config with the
default DummyBackend (no override) should therefore score 3/10 correct.
"""

import random
from pathlib import Path

import pandas as pd

_SEED = 42

_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "toy_normalized.csv"

# Each entry: (question_text, correct_answer_text, [distractor_1, distractor_2, distractor_3]).
# correct_letter is assigned below, not baked in here — only 3 of these 10 are
# pinned to land on "A" (matching DummyBackend's default output); the rest are
# shuffled for variety via the seeded RNG.
_QUESTIONS: list[tuple[str, str, list[str]]] = [
    ("What is 2 + 2?", "4", ["3", "5", "6"]),
    ("What color is the sky on a clear day?", "Blue", ["Green", "Red", "Purple"]),
    ("How many days are in a week?", "7", ["5", "6", "8"]),
    ("What is the capital of France?", "Paris", ["Berlin", "Madrid", "Rome"]),
    ("Which animal says 'moo'?", "Cow", ["Dog", "Cat", "Horse"]),
    ("What is the freezing point of water in Celsius?", "0", ["10", "-10", "100"]),
    ("How many legs does a spider have?", "8", ["4", "6", "10"]),
    ("What is the opposite of 'hot'?", "Cold", ["Warm", "Mild", "Humid"]),
    ("Which planet is known as the Red Planet?", "Mars", ["Venus", "Jupiter", "Saturn"]),
    ("What is the chemical symbol for water?", "H2O", ["CO2", "O2", "NaCl"]),
]

# Indices (0-based, into _QUESTIONS) whose correct answer is pinned to letter A —
# exactly 3, to match DummyBackend's default 3/10 verification.
_PINNED_TO_A = {0, 7, 8}

_LETTERS = ("A", "B", "C", "D")


def _build_row(index: int, question_text: str, correct_text: str, distractors: list[str], rng: random.Random) -> dict:
    question_id = f"toy_{index + 1:03d}"

    if index in _PINNED_TO_A:
        correct_letter = "A"
    else:
        # Never A — A is reserved for exactly the 3 pinned rows above.
        correct_letter = rng.choice(["B", "C", "D"])

    remaining_letters = [letter for letter in _LETTERS if letter != correct_letter]
    shuffled_distractors = list(distractors)
    rng.shuffle(shuffled_distractors)

    slot_for = {correct_letter: correct_text}
    for letter, text in zip(remaining_letters, shuffled_distractors):
        slot_for[letter] = text

    return {
        "question_id": question_id,
        "subject": "toy",
        "question_text": question_text,
        "choice_a": slot_for["A"],
        "choice_b": slot_for["B"],
        "choice_c": slot_for["C"],
        "choice_d": slot_for["D"],
        "correct_option": correct_letter,
    }


def main() -> None:
    rng = random.Random(_SEED)
    rows = [
        _build_row(i, question_text, correct_text, distractors, rng)
        for i, (question_text, correct_text, distractors) in enumerate(_QUESTIONS)
    ]

    df = pd.DataFrame(rows)
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to data/processed/toy_normalized.csv")


if __name__ == "__main__":
    main()
