# src/mcq_eval/scoring/scorer.py

from mcq_eval.parsing.types import PARSE_OK, ParseResult
from mcq_eval.scoring.types import (
    SCORE_CORRECT,
    SCORE_INCORRECT,
    SCORE_UNSCORABLE,
    ScoreResult,
)


def is_choice_correct(predicted_choice: str, gold_choice: str) -> bool:
    """
    Compare a parsed prediction against the gold label.

    Intended later behavior:
    - Canonicalize both inputs if needed.
    - Return True only for exact choice-letter equality.
    - Keep this helper intentionally simple and deterministic.

    Args:
        predicted_choice: Parsed answer letter from the model output.
        gold_choice: Ground-truth answer letter.

    Returns:
        Boolean correctness result.
    """
    return predicted_choice == gold_choice


def score_prediction(parse_result: ParseResult, gold_choice: str) -> ScoreResult:
    """
    Convert a ParseResult into a scoring outcome.

    Intended later rules:
    - If parse_result.status is not PARSE_OK, mark the result unscorable.
    - If parsed choice equals gold_choice, mark correct.
    - Otherwise mark incorrect.
    - Carry parse status through to the score result.

    Args:
        parse_result: Structured parser output.
        gold_choice: Ground-truth answer letter.

    Returns:
        ScoreResult for downstream aggregation.
    """
    if parse_result.status != PARSE_OK:
        return ScoreResult(
            is_correct = None,
            predicted_choice = None,
            gold_choice = gold_choice,
            status = SCORE_UNSCORABLE,
            parse_status=parse_result.status,
        )
    result = is_choice_correct(parse_result.final_choice, gold_choice)
    return ScoreResult(
        is_correct = result,
        predicted_choice = parse_result.final_choice,
        gold_choice = gold_choice,
        status = SCORE_CORRECT if result else SCORE_INCORRECT,
        parse_status = parse_result.status,
    )