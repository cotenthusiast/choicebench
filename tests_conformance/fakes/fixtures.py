"""The tiny diagnostic benchmark (spec section C), plus an independently
hand-computed rotation oracle for it.

Every question row here is a plain dict in the same normalized-record shape
choicebench.pipeline.options.build_choices() expects: question_id, subject,
question_text, choices_json (a list of {"text", "source_index"} dicts, in
canonical order), correct_option.

Question IDs are strings (never bare ints): choicebench.scoring.tiebreak
.resolve_tie() folds question_id through str(question_id) into its BLAKE2b
key, so a fixture using ints could mask a str/int inconsistency bug rather
than exercising the real string-keyed path every real benchmark row uses.

The rotation oracle tables below are computed by hand from
build_rotations()'s own documented contract (rotation r displays canonical
option (i + r) % n at display slot i) -- NOT by calling build_rotations()
itself -- so tests that compare against these tables are checking
build_rotations() against an independent computation, not against its own
output fed back into itself.
"""

from __future__ import annotations

from choicebench.constants import letters_for


def _row(question_id: str, subject: str, question_text: str,
         texts: list[str], correct_pos: int) -> dict:
    choices = [{"text": t, "source_index": i} for i, t in enumerate(texts)]
    labels = letters_for(len(texts))
    return {
        "question_id": question_id,
        "subject": subject,
        "question_text": question_text,
        "choices_json": choices,
        "correct_option": labels[correct_pos],
        "correct_index": correct_pos,
    }


def _rotation_oracle(texts: list[str]) -> list[dict[str, str]]:
    """Hand-derive, for each rotation r, the displayed label -> canonical
    text map: slot i shows canonical text at position (i + r) % n."""
    n = len(texts)
    labels = letters_for(n)
    oracle = []
    for r in range(n):
        mapping = {labels[i]: texts[(i + r) % n] for i in range(n)}
        oracle.append(mapping)
    return oracle


def _canonical_letter_oracle(texts: list[str]) -> list[list[int]]:
    """For each rotation r, slot_to_canonical[i] = (i + r) % n -- the
    canonical position displayed at slot i."""
    n = len(texts)
    return [[(i + r) % n for i in range(n)] for r in range(n)]


# --- q_n4_paris: plain 4-option question, gold = A (Paris). ---------------

N4_TEXTS = ["Paris", "London", "Berlin", "Madrid"]
q_n4_paris = _row(
    "diag-n4-paris-001", "geography",
    "What is the capital of France?", N4_TEXTS, correct_pos=0,
)
q_n4_paris_rotation_oracle = _rotation_oracle(N4_TEXTS)
q_n4_paris_slot_to_canonical_oracle = _canonical_letter_oracle(N4_TEXTS)

# --- q_n4_none_literal: one option's TEXT is literally "None". ------------
# Gold is NOT the "None" option, so a bug that treats the literal string
# "None" as a missing/NaN option (dropping it, or matching it against a
# real missing value) would surface as a wrong option count or wrong
# canonical mapping.

N4_NONE_TEXTS = ["Acetaminophen", "None", "Ibuprofen", "Aspirin"]
q_n4_none_literal = _row(
    "diag-n4-none-001", "pharmacology",
    "Which of these is a common pain reliever?", N4_NONE_TEXTS, correct_pos=0,
)

# --- q_n3_primary: 3-option question. --------------------------------------

N3_TEXTS = ["Mitochondria", "Nucleus", "Ribosome"]
q_n3_primary = _row(
    "diag-n3-primary-001", "biology",
    "Which organelle is the 'powerhouse of the cell'?", N3_TEXTS, correct_pos=0,
)
q_n3_rotation_oracle = _rotation_oracle(N3_TEXTS)
q_n3_slot_to_canonical_oracle = _canonical_letter_oracle(N3_TEXTS)

# --- q_n5_wide: 5-option question. -----------------------------------------

N5_TEXTS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter"]
q_n5_wide = _row(
    "diag-n5-wide-001", "astronomy",
    "Which planet is closest to the Sun?", N5_TEXTS, correct_pos=0,
)
q_n5_rotation_oracle = _rotation_oracle(N5_TEXTS)
q_n5_slot_to_canonical_oracle = _canonical_letter_oracle(N5_TEXTS)

# --- q_n4_similar_wording: two near-duplicate option texts, to stress the
# exact/containment/cosine text-matching cascade. ---------------------------

N4_SIMILAR_TEXTS = [
    "A rapid decline in population",
    "A rapid decrease in population",
    "A slow increase in population",
    "No change in population",
]
q_n4_similar_wording = _row(
    "diag-n4-similar-001", "ecology",
    "What is a population crash?", N4_SIMILAR_TEXTS, correct_pos=0,
)

ALL_DIAGNOSTIC_QUESTIONS = [
    q_n4_paris,
    q_n4_none_literal,
    q_n3_primary,
    q_n5_wide,
    q_n4_similar_wording,
]
