# tests/runners/test_permutation.py

import math
from pathlib import Path

import pytest

from choicebench.benchmarks.base import make_normalized_row
from choicebench.constants import letters_for
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.pipeline.prompt_builder import load_prompt_templates

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"
_TEMPLATES = load_prompt_templates("v1", _PROMPTS_DIR)
from choicebench.clients.types import ProviderTimeoutError
from choicebench.scoring.types import SCORE_CORRECT

from tests.runners.conftest import MockBackend


class TestGenerateRotations:
    """Tests for PermutationRunner._generate_rotations (Rotation objects)."""

    def test_returns_four_rotations(self, canonical_options):
        """Should produce exactly 4 cyclic rotations for 4 options."""
        rots = PermutationRunner._generate_rotations(canonical_options)
        assert len(rots) == 4

    def test_first_rotation_is_original(self, canonical_options):
        """First rotation's mapping should be identical to the canonical ordering."""
        rots = PermutationRunner._generate_rotations(canonical_options)
        assert rots[0].mapping == canonical_options
        assert rots[0].slot_to_canonical == (0, 1, 2, 3)

    def test_keys_preserved(self, canonical_options):
        """All rotations should have the same keys A, B, C, D."""
        rots = PermutationRunner._generate_rotations(canonical_options)
        for rot in rots:
            assert list(rot.mapping.keys()) == ["A", "B", "C", "D"]

    def test_values_rotated(self, canonical_options):
        """Each rotation should contain the same set of values."""
        rots = PermutationRunner._generate_rotations(canonical_options)
        original_values = set(canonical_options.values())
        for rot in rots:
            assert set(rot.mapping.values()) == original_values

    def test_all_rotations_distinct(self, canonical_options):
        """All 4 rotations should be different from each other."""
        rots = PermutationRunner._generate_rotations(canonical_options)
        perm_tuples = [tuple(r.mapping.values()) for r in rots]
        assert len(set(perm_tuples)) == 4

    def test_second_rotation_shifted_by_one(self, canonical_options):
        """Second rotation should shift values by one position."""
        rots = PermutationRunner._generate_rotations(canonical_options)
        assert rots[1].mapping["A"] == canonical_options["B"]
        assert rots[1].mapping["B"] == canonical_options["C"]
        assert rots[1].mapping["C"] == canonical_options["D"]
        assert rots[1].mapping["D"] == canonical_options["A"]

    def test_slot_to_canonical_matches_shift_formula(self, canonical_options):
        """F1 invariant: slot j of rotation i displays canonical index (i+j) % n."""
        n = len(canonical_options)
        rots = PermutationRunner._generate_rotations(canonical_options)
        for i, rot in enumerate(rots):
            assert rot.slot_to_canonical == tuple((i + j) % n for j in range(n))

    def test_returns_three_rotations_for_three_options(self):
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS"}

        rots = PermutationRunner._generate_rotations(canonical)

        assert len(rots) == 3
        assert list(rots[0].mapping) == ["A", "B", "C"]
        assert rots[1].mapping == {"A": "HTTP", "B": "HTTPS", "C": "FTP"}

    @pytest.mark.parametrize("n", [2, 5, 6, 10])
    def test_rotation_count_equals_choice_count_not_fixed_or_factorial(self, n):
        options = {label: f"text_{i}" for i, label in enumerate(letters_for(n))}
        rots = PermutationRunner._generate_rotations(options)

        # Exactly n rotations — not a hardcoded 4 ...
        assert len(rots) == n
        # ... and not the full factorial n! set of orderings.
        if n > 2:
            assert len(rots) < math.factorial(n)
        # Every rotation is a distinct cyclic shift of the option texts.
        values = list(options.values())
        assert [list(r.mapping.values()) for r in rots] == [
            values[i:] + values[:i] for i in range(n)
        ]
        assert len({tuple(r.mapping.values()) for r in rots}) == n


class TestBuildPermutedPrompt:
    """Tests for PermutationRunner._build_permuted_prompt."""

    def test_prompt_contains_permuted_options(self, runner_question_row, canonical_options):
        """Prompt should include the permuted option texts."""
        perms = [r.mapping for r in PermutationRunner._generate_rotations(canonical_options)]
        prompt = PermutationRunner._build_permuted_prompt(
            runner_question_row, perms[1], _TEMPLATES["direct_mcq"]
        )

        assert runner_question_row["question_text"] in prompt
        # In permutation 1, A maps to HTTP (originally B)
        assert "HTTP" in prompt

    def test_prompt_contains_question(self, runner_question_row, canonical_options):
        """Prompt should include the question text."""
        prompt = PermutationRunner._build_permuted_prompt(
            runner_question_row, canonical_options, _TEMPLATES["direct_mcq"]
        )
        assert "securely browse websites" in prompt

    def test_prompt_omits_missing_option_label(self, runner_question_row):
        prompt = PermutationRunner._build_permuted_prompt(
            runner_question_row,
            {"A": "FTP", "B": "HTTP", "C": "HTTPS"},
            _TEMPLATES["direct_mcq"],
        )

        assert "A. FTP" in prompt
        assert "B. HTTP" in prompt
        assert "C. HTTPS" in prompt
        assert "D." not in prompt


class TestCanonicalLetter:
    """Tests for PermutationRunner._canonical_letter (positional, text-blind inverse)."""

    def test_identity_rotation(self, canonical_options):
        """Under the identity rotation a parsed letter maps to itself."""
        rot = PermutationRunner._generate_rotations(canonical_options)[0]
        letters = list(canonical_options)
        assert PermutationRunner._canonical_letter("C", rot, letters) == "C"

    def test_shifted_rotation(self, canonical_options):
        """A letter parsed under rotation 1 maps back through the slot map."""
        rots = PermutationRunner._generate_rotations(canonical_options)
        letters = list(canonical_options)
        # In rotation 1: A->HTTP, B->HTTPS, C->SMTP, D->FTP
        # If model picks B (HTTPS), canonical HTTPS is C
        result = PermutationRunner._canonical_letter("B", rots[1], letters)
        assert result == "C"

    def test_all_letters_map_correctly_in_every_rotation(self, canonical_options):
        """Every letter in every rotation maps to the canonical letter whose
        text it displayed — via position, never via text lookup."""
        rots = PermutationRunner._generate_rotations(canonical_options)
        letters = list(canonical_options)
        for i, rot in enumerate(rots):
            for j, letter in enumerate(letters):
                expected = letters[rot.slot_to_canonical[j]]
                result = PermutationRunner._canonical_letter(letter, rot, letters)
                assert result == expected
                assert canonical_options[result] == rot.mapping[letter]

    def test_duplicate_option_text_resolves_positionally(self):
        """F1 regression: with duplicated option texts the parsed display slot
        maps to ITS OWN canonical position — an earlier textual twin can no
        longer capture the vote."""
        canonical = {"A": "x", "B": "dup", "C": "dup", "D": "y"}
        letters = list(canonical)
        rotations = PermutationRunner._generate_rotations(canonical)
        # Identity rotation: picking C (gold) must return C, not its twin B.
        assert PermutationRunner._canonical_letter("C", rotations[0], letters) == "C"
        assert PermutationRunner._canonical_letter("B", rotations[0], letters) == "B"
        # Rotation 1 (values shifted by one): slot displaying "dup" still
        # resolves to that slot's own canonical source.
        assert PermutationRunner._canonical_letter(
            "A", rotations[1], letters) == letters[rotations[1].slot_to_canonical[0]]


class TestMajorityVote:
    """Tests for PermutationRunner._majority_vote."""

    def test_unanimous(self, canonical_options):
        """All votes agree — should return that letter."""
        assert PermutationRunner._majority_vote(["C", "C", "C", "C"], canonical_options) == "C"

    def test_clear_majority(self, canonical_options):
        """Three-to-one — should return the majority."""
        assert PermutationRunner._majority_vote(["A", "C", "C", "C"], canonical_options) == "C"

    def test_tie_uses_canonically_earliest_letter(self, canonical_options):
        """Two-to-two tie — should return the canonically-earliest tied letter."""
        result = PermutationRunner._majority_vote(["A", "B", "A", "B"], canonical_options)
        assert result == "A"

    def test_all_none(self, canonical_options):
        """All votes failed — should return None."""
        assert PermutationRunner._majority_vote([None, None, None, None], canonical_options) is None

    def test_empty_list(self, canonical_options):
        """Empty list — should return None."""
        assert PermutationRunner._majority_vote([], canonical_options) is None

    def test_some_none(self, canonical_options):
        """Mix of valid and None — should vote among valid only."""
        result = PermutationRunner._majority_vote([None, "B", "B", None], canonical_options)
        assert result == "B"

    def test_single_valid_vote(self, canonical_options):
        """Only one non-None — should return that vote."""
        result = PermutationRunner._majority_vote([None, None, "D", None], canonical_options)
        assert result == "D"

    def test_tie_with_none_first(self, canonical_options):
        """First vote is None, tie among valid — should return canonically-earliest tied letter."""
        result = PermutationRunner._majority_vote([None, "A", "B", "A"], canonical_options)
        assert result == "A"

    def test_tie_across_non_adjacent_rotation_positions(self, canonical_options):
        """Regression test: tie-break must use canonical letter order, not vote/rotation
        order. Votes arrive as [C, A] (C from rotation 0, A from rotation 1) — a tie
        between A and C where C was cast first. The canonically-earlier letter (A)
        must win regardless of which rotation produced its vote first; the old
        implementation returned cleaned[0] ("C") here, reintroducing positional
        correlation into a method whose purpose is to cancel it.
        """
        result = PermutationRunner._majority_vote(["C", "A"], canonical_options)
        assert result == "A"

    def test_tie_break_ignores_rotation_order_for_three_way_tie(self, canonical_options):
        """Three-way tie arriving in reverse canonical order — canonically-earliest
        letter (A) must still win, not the first-encountered vote (D)."""
        result = PermutationRunner._majority_vote(["D", "C", "A"], canonical_options)
        assert result == "A"


class TestPermutationRunnerRunOne:
    """Tests for PermutationRunner.run_one execution flow."""

    def test_all_agree_correct(self, runner_question_row):
        """All permutations return the correct answer — should score correct."""
        # Correct answer is C (HTTPS). For each permutation, figure out
        # which letter maps to HTTPS and return that letter.
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = [r.mapping for r in PermutationRunner._generate_rotations(canonical)]

        responses = []
        for perm in perms:
            # Find which letter points to HTTPS in this permutation
            for letter, text in perm.items():
                if text == "HTTPS":
                    responses.append(letter)
                    break

        backend = MockBackend(responses=responses)
        runner = PermutationRunner(
            backend=backend,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="test_run_001",
        )

        result = runner.run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_majority_correct(self, runner_question_row):
        """Three correct, one wrong — majority vote should give correct answer."""
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = [r.mapping for r in PermutationRunner._generate_rotations(canonical)]

        responses = []
        for i, perm in enumerate(perms):
            if i == 0:
                # First permutation returns wrong answer
                responses.append("A")
            else:
                for letter, text in perm.items():
                    if text == "HTTPS":
                        responses.append(letter)
                        break

        backend = MockBackend(responses=responses)
        runner = PermutationRunner(
            backend=backend,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="test_run_001",
        )

        result = runner.run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_all_fail(self, runner_question_row):
        """All four calls fail — should have no parsed choice."""
        responses = [ProviderTimeoutError("Request timed out.") for _ in range(4)]
        backend = MockBackend(responses=responses)
        runner = PermutationRunner(
            backend=backend,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="test_run_001",
        )

        result = runner.run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None

    def test_makes_four_api_calls(self, runner_question_row):
        """Should fire exactly 4 requests — one per permutation."""
        backend = MockBackend(responses=["C"] * 4)
        runner = PermutationRunner(
            backend=backend,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="test_run_001",
        )

        runner.run_one(runner_question_row, sample_index=0)

        assert len(backend.requests_received) == 4

    def test_missing_trailing_option_makes_three_api_calls(
        self, runner_question_row_missing_trailing_option
    ):
        row = runner_question_row_missing_trailing_option
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS"}
        perms = [r.mapping for r in PermutationRunner._generate_rotations(canonical)]
        # Respond with whichever letter holds "HTTPS" in each rotation, so all
        # three permutations unanimously vote for canonical C — this exercises
        # the "exactly 3 calls for a 3-option question" behavior without
        # depending on tie-break resolution.
        responses = []
        for perm in perms:
            for letter, text in perm.items():
                if text == "HTTPS":
                    responses.append(letter)
                    break
        backend = MockBackend(responses=responses)
        runner = PermutationRunner(
            backend=backend,
            method_name="cyclic_permutation",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="test_run_001",
        )

        result = runner.run_one(row, sample_index=0)

        assert len(backend.requests_received) == 3
        assert "D." not in result["prompt"]
        assert result["parsed_choice"] == "C"
        assert result["score_status"] == SCORE_CORRECT

    def test_six_option_question_makes_six_api_calls(self):
        # A variable-choice (new schema) 6-option question rotates exactly 6
        # times end-to-end — the runner derives the count from len(choices).
        row = make_normalized_row(
            subject="net",
            question_text="Which protocol is used to securely browse websites?",
            choices=["FTP", "HTTP", "HTTPS", "SMTP", "SNMP", "POP3"],
            correct_index=2,
        )
        backend = MockBackend(responses=["A"] * 6)
        runner = PermutationRunner(
            backend=backend,
            method_name="cyclic_permutation",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="test_run_6opt",
        )

        result = runner.run_one(row, sample_index=0)

        assert len(backend.requests_received) == 6
        assert "E. SNMP" in result["prompt"]
        assert "F. POP3" in result["prompt"]

    def test_result_row_has_metadata(self, runner_question_row):
        """Result row should carry trace metadata."""
        backend = MockBackend(responses=["C"] * 4)
        runner = PermutationRunner(
            backend=backend,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="test_run_001",
        )

        result = runner.run_one(runner_question_row, sample_index=0)

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "pride"
        assert result["split_name"] == "robustness"

    def test_first_call_fails_but_vote_succeeds_is_not_failure(self, runner_question_row):
        """PF-3 / FSF-5: if only the first permutation call fails but the vote
        still produces a valid answer, the row must not be answer_status=failure
        while carrying a valid scored parsed_choice."""
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = [r.mapping for r in PermutationRunner._generate_rotations(canonical)]
        responses = []
        for i, perm in enumerate(perms):
            if i == 0:
                responses.append(ProviderTimeoutError("first call timed out"))
            else:
                for letter, text in perm.items():
                    if text == "HTTPS":
                        responses.append(letter)
                        break
        backend = MockBackend(responses=responses)
        runner = PermutationRunner(
            backend=backend,
            method_name="cyclic_permutation",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="test_run_001",
        )

        result = runner.run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["answer_status"] != "failure"
        assert result["answer_status"] == "success"

    def test_all_fail_marks_answer_status_failure(self, runner_question_row):
        """When the vote yields nothing, the row is answer_status=failure."""
        responses = [ProviderTimeoutError("timeout") for _ in range(4)]
        backend = MockBackend(responses=responses)
        runner = PermutationRunner(
            backend=backend,
            method_name="cyclic_permutation",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="test_run_001",
        )

        result = runner.run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["answer_status"] == "failure"
