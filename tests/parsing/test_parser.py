# tests/parsing/test_parser.py

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

    def test_coordinated_candidates_are_ambiguous(self) -> None:
        """P6-5 (ratified): a bare coordinated enumeration commits to no one;
        it must not silently resolve to the last mentioned letter."""
        result = extract_choice_letter("A or C or B")
        assert result.final_choice is None
        assert result.status == PARSE_AMBIGUOUS

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

    def test_numeric_option_does_not_match_inside_longer_number(self) -> None:
        # Correct answer "4" must NOT match the "4" inside "14".
        options = {"A": "4", "B": "5", "C": "3", "D": "6"}
        result = extract_choice_text_match("the total comes to 14", options)
        assert result.final_choice is None
        assert result.status == PARSE_MISSING

    def test_numeric_option_matches_as_standalone_token(self) -> None:
        # Correct answer "4" matches the standalone "4" in surrounding text.
        options = {"A": "4", "B": "5", "C": "3", "D": "6"}
        result = extract_choice_text_match("the answer is 4", options)
        assert result.final_choice == "A"
        assert result.status == PARSE_OK

    def test_numeric_option_matches_when_output_is_only_the_number(self) -> None:
        # Correct answer "4" matches when the output is exactly "4".
        options = {"A": "4", "B": "5", "C": "3", "D": "6"}
        result = extract_choice_text_match("4", options)
        assert result.final_choice == "A"
        assert result.status == PARSE_OK

    def test_word_option_matches_at_word_boundary_with_trailing_text(self) -> None:
        # Word-level (not digit-specific): "Paris" matches in "Paris, France".
        options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Rome"}
        result = extract_choice_text_match("Paris, France", options)
        assert result.final_choice == "A"
        assert result.status == PARSE_OK


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

    def test_rejects_label_not_present_in_current_options(self) -> None:
        options = {"A": "one", "B": "two", "C": "three"}

        result = parse_model_answer("D", options)

        assert result.final_choice is None
        assert result.status == PARSE_MISSING

    def test_fallback_path_preserves_ambiguity_recall_and_never_accepts_bare_mentions(self, sample_options, monkeypatch):
        """Ratified P6-1 contract for the fallback path:
        (a) an exact bare-option payload resolves via the structural gate (G1);
        (b) a payload with MULTIPLE option texts inside hedged prose stays
            PARSE_AMBIGUOUS even though no anchored grammar fired;
        (c) a SINGLE incidental mention can never become an answer through
            the fallback — blanket single hits are ignored by design."""
        # (a) G1: exact bare option text through the full path.
        result = parse_model_answer("Type 2 diabetes mellitus", sample_options)
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

        # (b) ambiguity recall without any anchored grammar hit.
        ambiguous = parse_model_answer(
            "It could be Paris or London honestly.", BATCH2_OPTIONS)
        assert ambiguous.status == PARSE_AMBIGUOUS

        # (c) single incidental mention stays missing; the legacy matcher is
        # consulted only to preserve AMBIGUOUS, and its OK result is refused.
        def fake_text_match(normalized_text, options):
            return ParseResult(
                final_choice="B", status=PARSE_OK, raw_text=None,
                normalized_text=normalized_text,
                reason="Answer successfully parsed from option text match")

        monkeypatch.setattr(
            "choicebench.parsing.parser.extract_choice_text_match", fake_text_match)
        refused = parse_model_answer(
            "I cannot help with questions about Madrid.", BATCH2_OPTIONS)
        assert refused.status == PARSE_MISSING
        assert refused.final_choice is None

    def test_numeric_substring_does_not_mis_score_through_full_path(self) -> None:
        # End-to-end: letter extraction misses, and the text-match fallback must
        # not score option "4" just because "4" is a substring of "14".
        options = {"A": "4", "B": "5", "C": "3", "D": "6"}
        result = parse_model_answer("After calculation, the total comes to 14.", options)
        assert result.final_choice is None
        assert result.status == PARSE_MISSING

    def test_realworld_text_answer_still_matches_through_full_path(self, sample_options) -> None:
        # No regression: a genuine written-out answer is still matched on the
        # fallback path when no letter is present.
        result = parse_model_answer(
            "Given the presentation, the diagnosis is Type 2 diabetes mellitus.",
            sample_options,
        )
        assert result.final_choice == "B"
        assert result.status == PARSE_OK


BATCH2_OPTIONS = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}


class TestBatch2Policies:
    """Ratified Batch-2 semantics: evidence classes, structural tier-4 gate,
    wrapper allowlist, compound ambiguity, last-explicit-selection rule."""

    def test_f2_incidental_reference_then_selection(self) -> None:
        result = parse_model_answer(
            "Option C refers to the capital of Germany. My pick is B.", BATCH2_OPTIONS)
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

    def test_p61_refusal_single_mention_is_missing(self) -> None:
        result = parse_model_answer(
            "I cannot help with questions about Madrid.", BATCH2_OPTIONS)
        assert result.status == PARSE_MISSING

    def test_p61_meta_single_mention_is_missing(self) -> None:
        result = parse_model_answer(
            "This reminds me of an old question about London.", BATCH2_OPTIONS)
        assert result.status == PARSE_MISSING

    def test_p61_prose_mention_is_missing(self) -> None:
        result = parse_model_answer(
            "Madrid appears in one of the options.", BATCH2_OPTIONS)
        assert result.status == PARSE_MISSING

    def test_p65_cued_compound_is_ambiguous(self) -> None:
        result = parse_model_answer("Answer: A or B", BATCH2_OPTIONS)
        assert result.status == PARSE_AMBIGUOUS
        assert result.final_choice is None

    def test_p65_hedged_compound_is_ambiguous(self) -> None:
        result = parse_model_answer("My answer is probably B or C", BATCH2_OPTIONS)
        assert result.status == PARSE_AMBIGUOUS

    def test_compound_last_selection_beats_clean_earlier(self) -> None:
        result = parse_model_answer(
            "Answer: B. Actually, my answer is A or C.", BATCH2_OPTIONS)
        assert result.status == PARSE_AMBIGUOUS
        assert result.final_choice is None

    def test_exclusion_prose_after_selection_preserves_selection(self) -> None:
        result = parse_model_answer("Final answer is B, not A or C.", BATCH2_OPTIONS)
        assert result.final_choice == "B"
        assert result.status == PARSE_OK

    def test_g5_present_tense_option_text_selections(self) -> None:
        for text in ("I choose Madrid.", "I pick Madrid.", "I select Madrid.",
                     "I go with Madrid.", "My pick is Madrid.",
                     "My choice is Madrid."):
            result = parse_model_answer(text, BATCH2_OPTIONS)
            assert result.final_choice == "D", text
            assert result.status == PARSE_OK

    def test_g5_past_tense_narration_is_not_commitment(self) -> None:
        result = parse_model_answer("I picked Madrid because it seemed right.", BATCH2_OPTIONS)
        assert result.status == PARSE_MISSING

    def test_tag_allowlist_positives(self) -> None:
        for text in ("<answer>A</answer>", "<final>B</final>", "<choice>C</choice>"):
            result = parse_model_answer(text, BATCH2_OPTIONS)
            assert result.status == PARSE_OK, text

    def test_tag_allowlist_negative_random_tag(self) -> None:
        result = parse_model_answer("<random>Madrid</random>", BATCH2_OPTIONS)
        assert result.status == PARSE_MISSING
        assert result.final_choice is None

    def test_letter_go_with_construction(self) -> None:
        result = parse_model_answer("I go with B.", BATCH2_OPTIONS)
        assert result.final_choice == "B"

    def test_wrapped_letters_still_parse(self) -> None:
        for text, expected in (("**B**", "B"), ("\u201cC\u201d is my choice", "C"),
                               ("<answer>A</answer>", "A")):
            result = parse_model_answer(text, BATCH2_OPTIONS)
            assert result.final_choice == expected, text
