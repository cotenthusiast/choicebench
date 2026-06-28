# tests/parsing/test_parser.py

import pytest

from choicebench.parsing.parser import (
    extract_choice_letter,
    extract_choice_text_match,
    normalize_output_text,
    parse_model_answer,
)
from choicebench.parsing.types import (
    PARSE_AMBIGUOUS,
    PARSE_MISSING,
    PARSE_OK,
    ParseResult,
)


class TestNormalizeOutputText:
    """Tests for normalize_output_text."""

    def test_returns_empty_string_for_none(self) -> None:
        assert normalize_output_text(None) == ""

    def test_returns_empty_string_for_blank_input(self) -> None:
        assert normalize_output_text(" ") == ""

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert normalize_output_text(" a b c ") == "a b c"

    def test_collapses_repeated_internal_spaces(self) -> None:
        assert normalize_output_text("a  b  c") == "a b c"

    def test_collapses_tabs_and_newlines_into_single_spaces(self) -> None:
        assert normalize_output_text("a\n\t b\t c") == "a b c"

    def test_preserves_meaningful_punctuation(self) -> None:
        assert normalize_output_text("(a)") == "(a)"


class TestExtractChoiceLetter:
    """Tests for extract_choice_letter."""

    def test_returns_missing_for_empty_normalized_text(self) -> None:
        assert extract_choice_letter("").status == PARSE_MISSING

    def test_parses_single_uppercase_letter(self) -> None:
        result = extract_choice_letter("B")
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

    def test_parses_single_lowercase_letter(self) -> None:
        result = extract_choice_letter("b")
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

    def test_parses_letter_wrapped_in_punctuation(self) -> None:
        result = extract_choice_letter("B:")
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

    def test_parses_answer_is_pattern(self) -> None:
        result = extract_choice_letter("answer is B")
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

    def test_parses_answer_direct_pattern(self) -> None:
        result = extract_choice_letter("answer B")
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

    def test_parses_final_answer_pattern(self) -> None:
        result = extract_choice_letter("final answer B")
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

    def test_parses_final_answer_is_pattern(self) -> None:
        result = extract_choice_letter("final answer is B")
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

    def test_last_strong_cue_wins_when_multiple_strong_candidates(self) -> None:
        result = extract_choice_letter(
            "The final answer is B or maybe the answer is C"
        )
        assert result.status == PARSE_OK
        assert result.final_choice == "C"

    def test_last_weak_candidate_wins_when_multiple_weak_candidates(self) -> None:
        result = extract_choice_letter("A or C or B")
        assert result.status == PARSE_OK
        assert result.final_choice == "B"

    def test_prefers_strong_candidate_over_weak_mentions(self) -> None:
        result = extract_choice_letter("The final answer is B, not A or C")
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

    def test_returns_missing_when_no_direct_letter_exists(self) -> None:
        assert extract_choice_letter("Im not sure").status == PARSE_MISSING


class TestExtractChoiceTextMatch:
    """Tests for extract_choice_text_match."""

    def test_returns_ok_for_exact_option_text_match(self, sample_options) -> None:
        result = extract_choice_text_match(sample_options["B"], sample_options)
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

    def test_returns_ok_when_option_text_appears_in_longer_response(self, sample_options) -> None:
        normalized_text = f"The answer is {sample_options['C']}"
        result = extract_choice_text_match(normalized_text, sample_options)
        assert result.final_choice == "C"
        assert result.status == PARSE_OK

    def test_returns_ambiguous_when_multiple_option_texts_match(self, sample_options) -> None:
        normalized_text = f"It could be {sample_options['A']} or {sample_options['D']}"
        result = extract_choice_text_match(normalized_text, sample_options)
        assert result.final_choice is None
        assert result.status == PARSE_AMBIGUOUS

    def test_returns_missing_when_no_option_text_matches(self, sample_options) -> None:
        result = extract_choice_text_match("I am not sure what the answer is.", sample_options)
        assert result.final_choice is None
        assert result.status == PARSE_MISSING


class TestParseModelAnswer:
    """Tests for parse_model_answer."""

    def test_parses_single_letter_answer(self, sample_case_map, sample_options) -> None:
        case = sample_case_map["clean_b"]
        result = parse_model_answer(str(case["raw_text"]), sample_options)
        assert result.final_choice == case["expected_choice"]
        assert result.status == case["expected_status"]

    def test_parses_lowercase_letter(self, sample_case_map, sample_options) -> None:
        case = sample_case_map["lowercase_with_spaces"]
        result = parse_model_answer(str(case["raw_text"]), sample_options)
        assert result.final_choice == case["expected_choice"]
        assert result.status == case["expected_status"]

    def test_ignores_punctuation_around_letter(self, sample_case_map, sample_options) -> None:
        case = sample_case_map["punctuation_around_letter"]
        result = parse_model_answer(str(case["raw_text"]), sample_options)
        assert result.final_choice == case["expected_choice"]
        assert result.status == case["expected_status"]

    def test_rejects_multiple_conflicting_letters(self, sample_case_map, sample_options) -> None:
        case = sample_case_map["conflicting_letters"]
        result = parse_model_answer(str(case["raw_text"]), sample_options)
        assert result.final_choice == case["expected_choice"]
        assert result.status == case["expected_status"]

    def test_rejects_empty_output(self, sample_case_map, sample_options) -> None:
        case = sample_case_map["empty_string"]
        result = parse_model_answer(str(case["raw_text"]), sample_options)
        assert result.final_choice == case["expected_choice"]
        assert result.status == case["expected_status"]

    def test_fallback_matches_option_text(self, sample_case_map, sample_options) -> None:
        case = sample_case_map["option_text_instead_of_letter"]
        result = parse_model_answer(str(case["raw_text"]), sample_options)
        assert result.final_choice == case["expected_choice"]
        assert result.status == case["expected_status"]

    def test_returns_ambiguous_on_multiple_valid_text_matches(self, sample_case_map, sample_options) -> None:
        case = sample_case_map["ambiguous_text_match"]
        result = parse_model_answer(str(case["raw_text"]), sample_options)
        assert result.final_choice == case["expected_choice"]
        assert result.status == case["expected_status"]

    def test_handles_reasoning_plus_final_answer_format(self, sample_case_map, sample_options) -> None:
        case = sample_case_map["reasoning_final_a"]
        result = parse_model_answer(str(case["raw_text"]), sample_options)
        assert result.final_choice == case["expected_choice"]
        assert result.status == case["expected_status"]

    def test_preserves_original_raw_text_in_final_result(self, sample_case_map, sample_options) -> None:
        case = sample_case_map["lowercase_with_spaces"]
        result = parse_model_answer(str(case["raw_text"]), sample_options)
        assert result.raw_text == case["raw_text"]

    def test_uses_text_match_only_after_missing_direct_letter(self, sample_options, monkeypatch) -> None:
        text_match_called = False

        def fake_extract_choice_letter(normalized_text):
            return ParseResult(
                final_choice=None, status=PARSE_MISSING,
                raw_text=None, normalized_text=normalized_text,
                reason="No direct answer letter found",
            )

        def fake_extract_choice_text_match(normalized_text, options):
            nonlocal text_match_called
            text_match_called = True
            return ParseResult(
                final_choice="B", status=PARSE_OK,
                raw_text=None, normalized_text=normalized_text,
                reason="Answer successfully parsed from option text match",
            )

        monkeypatch.setattr("choicebench.parsing.parser.extract_choice_letter", fake_extract_choice_letter)
        monkeypatch.setattr("choicebench.parsing.parser.extract_choice_text_match", fake_extract_choice_text_match)

        result = parse_model_answer("Type 2 diabetes mellitus", sample_options)

        assert text_match_called is True
        assert result.final_choice == "B"
        assert result.status == PARSE_OK
