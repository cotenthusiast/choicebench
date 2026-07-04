# tests/pipeline/test_prompt_builder.py

from pathlib import Path

import pytest

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
        options = {"A": "one", "B": "two", "C": "three", "D": "four"}

        prompt = build_direct_mcq_prompt(
            _TEMPLATES["direct_mcq"], question, options
        )

        assert question in prompt
        assert "Respond with only the letter." in prompt
        assert prompt.index("A. one") < prompt.index("B. two")
        assert prompt.index("B. two") < prompt.index("C. three")
        assert prompt.index("C. three") < prompt.index("D. four")

    def test_omits_missing_option_labels_from_options_block(self):
        question = "Which number has one factor?"
        options = {"A": "one", "B": "two", "C": "three"}

        prompt = build_direct_mcq_prompt(_TEMPLATES["direct_mcq"], question, options)

        assert "A. one" in prompt
        assert "B. two" in prompt
        assert "C. three" in prompt
        assert "D." not in prompt


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
        options = {"A": "one", "B": "two", "C": "three", "D": "four"}
        free_response = "one"

        prompt = build_option_matching_prompt(
            _TEMPLATES["option_matching"],
            question,
            free_response,
            options,
        )

        assert "Select the option that best matches the reference answer in the context of the question.".lower() in prompt.lower()
        assert question in prompt
        assert "Respond with only the letter." in prompt
        assert prompt.index("A. one") < prompt.index("B. two")
        assert prompt.index("B. two") < prompt.index("C. three")
        assert prompt.index("C. three") < prompt.index("D. four")
        assert free_response in prompt


class TestBuildDirectMcqPromptSubject:
    """Tests for the optional subject parameter (pride_repro template)."""

    _PRIDE_REPRO_TEMPLATES = load_prompt_templates("pride_repro", _PROMPTS_DIR)

    def test_subject_underscores_become_spaces(self):
        prompt = build_direct_mcq_prompt(
            self._PRIDE_REPRO_TEMPLATES["direct_mcq"],
            question="Q?",
            options={"A": "one", "B": "two"},
            subject="abstract_algebra",
        )
        assert "about abstract algebra." in prompt
        assert "abstract_algebra" not in prompt

    def test_prompt_ends_exactly_at_answer_colon(self):
        prompt = build_direct_mcq_prompt(
            self._PRIDE_REPRO_TEMPLATES["direct_mcq"],
            question="Q?",
            options={"A": "one", "B": "two"},
            subject="anatomy",
        )
        assert prompt.endswith("Answer:")

    def test_v1_template_ignores_absent_subject(self):
        # v1's direct_mcq.txt has no {subject} placeholder; omitting subject
        # (the default) must not raise and must not alter existing behavior.
        prompt = build_direct_mcq_prompt(
            _TEMPLATES["direct_mcq"], "Q?", {"A": "one", "B": "two"}
        )
        assert "Q?" in prompt

    def test_pride_repro_template_without_subject_raises_keyerror(self):
        with pytest.raises(KeyError):
            build_direct_mcq_prompt(
                self._PRIDE_REPRO_TEMPLATES["direct_mcq"], "Q?", {"A": "one", "B": "two"}
            )
