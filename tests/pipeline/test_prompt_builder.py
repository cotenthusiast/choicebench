# tests/pipeline/test_prompt_builder.py

from pathlib import Path

import pytest

from choicebench.pipeline.prompt_builder import (
    Rotation,
    build_direct_mcq_prompt,
    build_free_text_prompt,
    build_option_matching_prompt,
    build_rotations,
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


class TestBuildRotations:
    """F1 invariant: displayed option position -> exactly one canonical option
    position, produced by the same operation that renders the mapping."""

    def test_rotation_count_and_first_is_identity(self):
        options = {"A": "one", "B": "two", "C": "three", "D": "four"}
        rots = build_rotations(options)
        assert len(rots) == 4
        assert all(isinstance(r, Rotation) for r in rots)
        assert rots[0].mapping == options
        assert rots[0].slot_to_canonical == (0, 1, 2, 3)

    def test_slot_to_canonical_follows_shift_formula(self):
        n = 5
        options = {chr(65 + i): f"t{i}" for i in range(n)}
        for i, rot in enumerate(build_rotations(options)):
            assert rot.slot_to_canonical == tuple((i + j) % n for j in range(n))

    def test_slot_map_is_a_bijection_for_every_rotation(self):
        import itertools

        for n in (2, 3, 4, 7, 10):
            options = {chr(65 + i): f"t{i}" for i in range(n)}
            for rot in build_rotations(options):
                assert sorted(rot.slot_to_canonical) == list(range(n))

    def test_render_then_inverse_round_trips_every_slot(self):
        """Rendering slot j's text and inverting through the slot map must
        recover canonical slot slot_to_canonical[j] — with NO reference to
        option text values (the inverse is positional)."""
        options = {"A": "dup", "B": "dup", "C": "solo", "D": "other"}
        letters = list(options)
        for rot in build_rotations(options):
            for j, letter in enumerate(letters):
                # Positional inverse of displayed slot j:
                recovered = letters[rot.slot_to_canonical[j]]
                # The rendered text at slot j is exactly the canonical text
                # of the recovered slot (holds even for duplicate texts).
                assert rot.mapping[letter] == options[recovered]

    def test_duplicate_texts_do_not_alias_slots(self):
        """The motivating F1 case: two slots share identical text; their slot
        maps must still point at distinct canonical positions."""
        options = {"A": "x", "B": "dup", "C": "dup", "D": "y"}
        rots = build_rotations(options)
        identity = rots[0]
        assert identity.slot_to_canonical[1] == 1
        assert identity.slot_to_canonical[2] == 2
