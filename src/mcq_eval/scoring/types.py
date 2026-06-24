# src/mcq_eval/scoring/types.py

from __future__ import annotations

from dataclasses import dataclass


SCORE_CORRECT = "score_correct"
SCORE_INCORRECT = "score_incorrect"
SCORE_UNSCORABLE = "score_unscorable"

SCORE_STATUSES = {
    SCORE_CORRECT,
    SCORE_INCORRECT,
    SCORE_UNSCORABLE,
}


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """
    Structured result returned by scoring functions.

    Attributes:
        is_correct: True/False when scorable, otherwise None.
        predicted_choice: Parsed predicted choice letter if available.
        gold_choice: Ground-truth choice letter.
        status: Score outcome status constant.
        parse_status: Parse status carried through from parsing.
    """

    is_correct: bool | None
    predicted_choice: str | None
    gold_choice: str
    status: str
    parse_status: str