# src/choicebench/scoring/scorer.py

from choicebench.parsing.types import PARSE_OK, ParseResult
from choicebench.scoring.types import (
    SCORE_CORRECT,
    SCORE_INCORRECT,
    SCORE_UNSCORABLE,
    ScoreResult,
)


def score_prediction(parse_result: ParseResult, gold_choice: str) -> ScoreResult:
    """
    Convert a ParseResult into a scoring outcome.

    If parse_result.status is not PARSE_OK, the result is marked
    unscorable. Otherwise it is marked correct or incorrect based on
    whether the parsed choice equals gold_choice. The parse status is
    always carried through to the returned ScoreResult.

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
    result = parse_result.final_choice == gold_choice
    return ScoreResult(
        is_correct = result,
        predicted_choice = parse_result.final_choice,
        gold_choice = gold_choice,
        status = SCORE_CORRECT if result else SCORE_INCORRECT,
        parse_status = parse_result.status,
    )