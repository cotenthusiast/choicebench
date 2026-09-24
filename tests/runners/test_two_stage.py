# tests/runners/test_two_stage.py

import json
from pathlib import Path

from choicebench.methods.library.permutation import PermutationRunner
from choicebench.methods.library.two_stage import TwoStageRunner
from choicebench.clients.types import ProviderTimeoutError
from choicebench.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend):
    return TwoStageRunner(
        backend=backend,
        method_name="two_stage",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
    )


class TestTwoStageRunnerRunOne:
    """Tests for TwoStageRunner.run_one execution flow."""

    def test_correct_two_stage(self, runner_question_row):
        """Free text returns 'HTTPS', matching returns 'C' — should score correct."""
        backend = MockBackend(responses=["HTTPS", "C"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_incorrect_two_stage(self, runner_question_row):
        """Free text returns 'FTP', matching returns 'A' — should score incorrect."""
        backend = MockBackend(responses=["FTP", "A"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_stage_one_failure_returns_early(self, runner_question_row):
        """If stage 1 fails, should return immediately with no parse or score."""
        backend = MockBackend(responses=[ProviderTimeoutError("Request timed out.")])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["score_status"] is None
        assert len(backend.requests_received) == 1

    def test_stage_two_failure(self, runner_question_row):
        """Stage 1 succeeds but stage 2 fails — parse and score should be None."""
        backend = MockBackend(responses=["HTTPS", ProviderTimeoutError("Request timed out.")])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["free_text_response"] == "HTTPS"

    def test_free_text_response_preserved(self, runner_question_row):
        """The intermediate free-text response should be saved in the result row."""
        backend = MockBackend(responses=["HTTPS", "C"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["free_text_response"] == "HTTPS"
        assert result["free_text_prompt"] is not None
        assert result["free_text_latency"] is not None

    def test_makes_two_api_calls(self, runner_question_row):
        """Should fire exactly 2 requests — free text then matching."""
        backend = MockBackend(responses=["HTTPS", "C"])
        _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert len(backend.requests_received) == 2

    def test_matching_prompt_contains_free_text(self, runner_question_row):
        """The option-matching prompt should include the free-text answer."""
        backend = MockBackend(responses=["HTTPS", "C"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert "HTTPS" in result["prompt"]

    def test_result_row_metadata(self, runner_question_row):
        """Result row should carry trace metadata."""
        backend = MockBackend(responses=["HTTPS", "C"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "two_stage"
        assert result["split_name"] == "robustness"

    def test_unparseable_matching_response(self, runner_question_row):
        """Stage 2 returns gibberish — should be unscorable."""
        backend = MockBackend(responses=["HTTPS", "I think it might be one of those"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE
        assert result["free_text_response"] == "HTTPS"

    def test_missing_trailing_option_is_not_rendered_or_parsed(
        self, runner_question_row_missing_trailing_option
    ):
        backend = MockBackend(responses=["HTTPS", "D"])

        result = _make_runner(backend).run_one(
            runner_question_row_missing_trailing_option, sample_index=0
        )

        assert "D." not in result["prompt"]
        assert result["parsed_choice"] is None


class TestRunStage2Rotations:
    """run_stage2_rotations reruns ONLY stage 2 (option matching) under every
    cyclic rotation of the options, reusing a fixed, already-elicited stage-1
    free-text answer -- no new stage-1 call. This is what makes two_stage_v1's
    flip-rate computable: does the stage-2 match change under a rotation,
    holding the free-text answer constant?"""

    def test_makes_exactly_n_stage2_calls_no_stage1_call(self, runner_question_row):
        # 4 options -> 4 rotations -> 4 stage-2 calls, zero stage-1 calls.
        backend = MockBackend(responses=["C", "A", "B", "D"])
        runner = _make_runner(backend)

        runner.run_stage2_rotations(runner_question_row, "HTTPS", sample_index=0)

        assert len(backend.requests_received) == 4
        for prompt in backend.requests_received:
            assert "HTTPS" in prompt  # reused free-text answer, not re-elicited

    def test_persists_one_canonical_choice_per_rotation(self, runner_question_row):
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = [r.mapping for r in PermutationRunner._generate_rotations(canonical)]
        responses = []
        for perm in perms:
            for letter, text in perm.items():
                if text == "HTTPS":
                    responses.append(letter)
                    break

        backend = MockBackend(responses=responses)
        runner = _make_runner(backend)

        result = runner.run_stage2_rotations(runner_question_row, "HTTPS", sample_index=0)

        per_rotation = json.loads(result["per_rotation_choices_json"])
        assert per_rotation == ["C", "C", "C", "C"]
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_free_text_response_field_carries_the_reused_answer(self, runner_question_row):
        backend = MockBackend(responses=["C", "C", "C", "C"])
        runner = _make_runner(backend)

        result = runner.run_stage2_rotations(runner_question_row, "HTTPS", sample_index=0)

        assert result["free_text_response"] == "HTTPS"

    def test_all_rotations_fail_yields_no_parsed_choice(self, runner_question_row):
        backend = MockBackend(
            responses=[ProviderTimeoutError("timed out") for _ in range(4)]
        )
        runner = _make_runner(backend)

        result = runner.run_stage2_rotations(runner_question_row, "HTTPS", sample_index=0)

        assert result["parsed_choice"] is None
        per_rotation = json.loads(result["per_rotation_choices_json"])
        assert per_rotation == [None, None, None, None]

    def test_dissenting_rotation_is_visible_in_the_trace_despite_majority_vote(
        self, runner_question_row
    ):
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = [r.mapping for r in PermutationRunner._generate_rotations(canonical)]
        responses = []
        for i, perm in enumerate(perms):
            if i == 0:
                responses.append("A")  # dissents: canonical A (FTP)
            else:
                for letter, text in perm.items():
                    if text == "HTTPS":
                        responses.append(letter)
                        break

        backend = MockBackend(responses=responses)
        runner = _make_runner(backend)

        result = runner.run_stage2_rotations(runner_question_row, "HTTPS", sample_index=0)

        per_rotation = json.loads(result["per_rotation_choices_json"])
        assert per_rotation == ["A", "C", "C", "C"]
        assert result["parsed_choice"] == "C"  # majority vote still wins overall

    def test_result_row_has_metadata(self, runner_question_row):
        backend = MockBackend(responses=["C", "C", "C", "C"])
        runner = _make_runner(backend)

        result = runner.run_stage2_rotations(runner_question_row, "HTTPS", sample_index=0)

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "two_stage"
        assert result["split_name"] == "robustness"
