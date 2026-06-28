# tests/pipeline/test_prompt_builder.py

from pathlib import Path

from choicebench.pipeline.prompt_builder import (
    build_direct_mcq_prompt,
    build_free_text_prompt,
    build_option_matching_prompt,
    load_prompt_templates,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "prompts"
_TEMPLATES = load_prompt_templates("v1", _PROMPTS_DIR)


class TestBuildDirectMcqPrompt:
    """Tests for build_direct_mcq_prompt."""

    def test_includes_question_options_and_letter_instruction(self):
        question = "Which number has one factor?"
        option_a = "one"
        option_b = "two"
        option_c = "three"
        option_d = "four"

        prompt = build_direct_mcq_prompt(
            _TEMPLATES["direct_mcq"], question, option_a, option_b, option_c, option_d
        )

        assert question in prompt
        assert "Respond with only the letter." in prompt
        assert prompt.index("A. one") < prompt.index("B. two")
        assert prompt.index("B. two") < prompt.index("C. three")
        assert prompt.index("C. three") < prompt.index("D. four")


class TestBuildFreeTextPrompt:
    """Tests for build_free_text_prompt."""

    def test_includes_question_and_excludes_options(self):
        question = "Which number has one factor?"

        actual = build_free_text_prompt(_TEMPLATES["free_text"], question)

        assert question in actual
        assert "Options:" not in actual
        assert "A." not in actual
        assert "B." not in actual
        assert "C." not in actual
        assert "D." not in actual


class TestBuildOptionMatchingPrompt:
    """Tests for build_option_matching_prompt."""

    def test_includes_question_free_text_options_and_letter_instruction(self):
        question = "Which number has one factor?"
        option_a = "one"
        option_b = "two"
        option_c = "three"
        option_d = "four"
        free_response = "one"

        prompt = build_option_matching_prompt(
            _TEMPLATES["option_matching"],
            question,
            free_response,
            option_a,
            option_b,
            option_c,
            option_d,
        )

        assert "Select the option that best matches the reference answer in the context of the question.".lower() in prompt.lower()
        assert question in prompt
        assert "Respond with only the letter." in prompt
        assert prompt.index("A. one") < prompt.index("B. two")
        assert prompt.index("B. two") < prompt.index("C. three")
        assert prompt.index("C. three") < prompt.index("D. four")
        assert free_response in prompt
