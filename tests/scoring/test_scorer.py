# tests/scoring/test_scorer.py

from __future__ import annotations

from choicebench.parsing.types import (
    PARSE_AMBIGUOUS,
    PARSE_MISSING,
    PARSE_OK,
    ParseResult,
)
from choicebench.scoring.scorer import score_prediction
from choicebench.scoring.types import (
    SCORE_CORRECT,
    SCORE_INCORRECT,
    SCORE_UNSCORABLE,
)


class TestScorePrediction:
    """Tests for score_prediction."""

    def test_correct_parsed_answer_scores_correct(self, sample_gold_choice: str) -> None:
        """Marks a successfully parsed matching answer as correct."""
        parse_result = ParseResult(
            final_choice=sample_gold_choice, status=PARSE_OK,
            raw_text=sample_gold_choice, normalized_text=sample_gold_choice,
            reason="Successfully parsed direct answer",
        )
        assert score_prediction(parse_result, gold_choice=sample_gold_choice).status == SCORE_CORRECT

    def test_wrong_parsed_answer_scores_incorrect(self, sample_gold_choice: str) -> None:
        """Marks a successfully parsed non-matching answer as incorrect."""
        wrong_choice = "A" if sample_gold_choice != "A" else "C"
        parse_result = ParseResult(
            final_choice=wrong_choice, status=PARSE_OK,
            raw_text=wrong_choice, normalized_text=wrong_choice,
            reason="Successfully parsed direct answer",
        )
        assert score_prediction(parse_result, gold_choice=sample_gold_choice).status == SCORE_INCORRECT

    def test_ambiguous_parse_scores_unscorable(self, sample_gold_choice: str) -> None:
        """Marks a non-OK parse as unscorable."""
        parse_result = ParseResult(
            final_choice=None, status=PARSE_AMBIGUOUS,
            raw_text="A or B", normalized_text="A or B",
            reason="Multiple conflicting candidates found",
        )
        assert score_prediction(parse_result, gold_choice=sample_gold_choice).status == SCORE_UNSCORABLE

    def test_missing_parse_scores_unscorable(self, sample_gold_choice: str) -> None:
        """Marks a missing parse as unscorable."""
        parse_result = ParseResult(
            final_choice=None, status=PARSE_MISSING,
            raw_text="", normalized_text="",
            reason="No output to parse",
        )
        assert score_prediction(parse_result, gold_choice=sample_gold_choice).status == SCORE_UNSCORABLE
