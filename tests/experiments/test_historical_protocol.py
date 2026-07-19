# tests/experiments/test_historical_protocol.py
#
# Fidelity tests for experiments/visible_llm_matcher/historical_protocol.py.
# The first test class asserts byte-for-byte equality against a REAL,
# already-published two_prompt Stage-2 prompt pulled from
# two-stage-prompting/paper_results/eval_ready/paper_api_main/
# 20260601_183346_two_prompt_gpt-4.1-mini_arc_challenge.csv (question_id
# 79e8c959bbeb74a0) — not a hand-written expectation — so this is a
# regression test against real historical data, not just internal
# consistency.

import math

from experiments.visible_llm_matcher.historical_protocol import (
    DEFAULT_VALID_CHOICES,
    build_option_matching_prompt,
    extract_choice_letter,
    extract_choice_text_match,
    parse_model_answer,
)

_OPTION_MATCHING_TEMPLATE = (
    "You are given a question, a reference answer, and four options.\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Reference answer: {free_text}\n"
    "\n"
    "Options:\n"
    "A. {option_a}\n"
    "B. {option_b}\n"
    "C. {option_c}\n"
    "D. {option_d}\n"
    "\n"
    "Select the option that best matches the reference answer in the context of the question.\n"
    "If the reference answer is imperfect or incomplete, choose the closest option.\n"
    "Respond with only the letter."
)

# Captured verbatim from the real, already-published stored prompt in
# two-stage-prompting/paper_results/eval_ready/paper_api_main/
# 20260601_183346_two_prompt_gpt-4.1-mini_arc_challenge.csv, row
# question_id=79e8c959bbeb74a0 (`prompt` column, read 2026-07-19).
_REAL_HISTORICAL_NAN_PROMPT = (
    "You are given a question, a reference answer, and four options.\n"
    "\n"
    "Question: A toy truck rolls over a smooth surface. If the surface is covered with sand, the truck will most likely roll\n"
    "\n"
    "Reference answer: The truck will most likely roll slower.\n"
    "\n"
    "Options:\n"
    "A. slower\n"
    "B. faster\n"
    "C. at the same speed\n"
    "D. nan\n"
    "\n"
    "Select the option that best matches the reference answer in the context of the question.\n"
    "If the reference answer is imperfect or incomplete, choose the closest option.\n"
    "Respond with only the letter."
)


class TestBuildOptionMatchingPromptFidelity:
    def test_matches_real_historical_nan_row_byte_for_byte(self):
        """question_id 79e8c959bbeb74a0 is one of the 3 known 3-option ARC
        questions. two_prompt's own (already published) Stage 2 rendered its
        missing 4th option as the literal text "D. nan". This runner's
        ported prompt builder must reproduce that exactly when given the
        same inputs, proving Stage-2 serialization fidelity against real
        historical data rather than a hand-written expectation.
        """
        prompt = build_option_matching_prompt(
            template=_OPTION_MATCHING_TEMPLATE,
            question=(
                "A toy truck rolls over a smooth surface. If the surface is "
                "covered with sand, the truck will most likely roll"
            ),
            free_text="The truck will most likely roll slower.",
            option_a="slower",
            option_b="faster",
            option_c="at the same speed",
            option_d=math.nan,
        )
        assert prompt == _REAL_HISTORICAL_NAN_PROMPT

    def test_normal_four_option_question_has_no_nan(self):
        prompt = build_option_matching_prompt(
            template=_OPTION_MATCHING_TEMPLATE,
            question="Which protocol is primarily used to securely browse websites?",
            free_text="HTTPS",
            option_a="FTP",
            option_b="HTTP",
            option_c="HTTPS",
            option_d="SMTP",
        )
        assert "nan" not in prompt
        assert "D. SMTP" in prompt
        assert "Reference answer: HTTPS" in prompt

    def test_does_not_drop_missing_option_unlike_choicebench_builder(self):
        """Documents, via a live import comparison, that
        choicebench.pipeline.prompt_builder deliberately drops a missing
        option while this ported version deliberately does not — the two
        are NOT interchangeable, which is why historical_protocol.py exists
        as a separate module instead of reusing the shipped one.
        """
        from choicebench.pipeline.prompt_builder import build_option_matching_prompt as cb_build

        ours = build_option_matching_prompt(
            template=_OPTION_MATCHING_TEMPLATE,
            question="Q",
            free_text="FT",
            option_a="a",
            option_b="b",
            option_c="c",
            option_d=math.nan,
        )
        theirs = cb_build(
            template="Question: {question}\nReference answer: {free_text}\nOptions:\n{options}\n",
            question="Q",
            free_text="FT",
            options={"A": "a", "B": "b", "C": "c"},
        )
        assert "nan" in ours
        assert "nan" not in theirs


class TestExtractChoiceLetterFidelity:
    def test_default_valid_choices_is_always_abcd(self):
        assert DEFAULT_VALID_CHOICES == ("A", "B", "C", "D")

    def test_does_not_restrict_to_real_options_unlike_choicebench_parser(self):
        """TSP's parse_model_answer never narrows valid_choices to the
        question's real option letters — even a genuinely 3-option question
        would accept a stray "D" in the response. This is a known, faithfully
        reproduced quirk (not something to silently fix), since two_prompt's
        own already-published Stage-2 rows exhibit the same behavior for the
        same 3 ARC questions.
        """
        result = extract_choice_letter("D")
        assert result.final_choice == "D"

    def test_single_token_response(self):
        result = extract_choice_letter("C")
        assert result.final_choice == "C"

    def test_strong_cue_pattern_wins_over_earlier_letters(self):
        result = extract_choice_letter(
            "I was thinking A or B, but the final answer is C"
        )
        assert result.final_choice == "C"

    def test_last_standalone_letter_when_no_cue(self):
        result = extract_choice_letter("Maybe A, or perhaps B")
        assert result.final_choice == "B"

    def test_empty_text_is_missing(self):
        result = extract_choice_letter("")
        assert result.final_choice is None
        assert result.status == "parse_missing"


class TestExtractChoiceTextMatchFidelity:
    def test_substring_containment_not_token_boundary(self):
        """TSP's fallback uses plain substring containment, unlike
        choicebench's token-boundary regex match. A short numeric-like
        option embedded in a longer answer should still match here (that's
        the faithfully-reproduced, looser historical behavior).
        """
        options = {"A": "4", "B": "5", "C": "6", "D": "7"}
        result = extract_choice_text_match("the answer is 14 divided by something", options)
        # "4" is contained in "14" under plain substring matching.
        assert result.final_choice == "A"

    def test_exact_match(self):
        options = {"A": "slower", "B": "faster", "C": "at the same speed"}
        result = extract_choice_text_match("faster", options)
        assert result.final_choice == "B"

    def test_ambiguous_when_multiple_match(self):
        options = {"A": "cat", "B": "category"}
        result = extract_choice_text_match("category", options)
        assert result.status == "parse_ambiguous"

    def test_missing_when_none_match(self):
        options = {"A": "cat", "B": "dog"}
        result = extract_choice_text_match("elephant", options)
        assert result.status == "parse_missing"


class TestParseModelAnswer:
    def test_letter_extraction_takes_priority(self):
        options = {"A": "slower", "B": "faster"}
        result = parse_model_answer("A", options)
        assert result.final_choice == "A"

    def test_falls_back_to_text_match(self):
        options = {"A": "slower", "B": "faster"}
        result = parse_model_answer("faster", options)
        assert result.final_choice == "B"

    def test_never_raises_on_none(self):
        options = {"A": "slower", "B": "faster"}
        result = parse_model_answer(None, options)
        assert result.final_choice is None
        assert result.raw_text is None
