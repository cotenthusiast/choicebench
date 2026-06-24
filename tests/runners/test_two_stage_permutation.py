# tests/runners/test_two_stage_permutation.py

from pathlib import Path

from mcq_eval.methods.two_stage_permutation import TwoStagePermutationRunner
from mcq_eval.methods.permutation import PermutationRunner
from mcq_eval.clients.types import ProviderTimeoutError
from mcq_eval.scoring.types import SCORE_CORRECT, SCORE_INCORRECT

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend):
    return TwoStagePermutationRunner(
        backend=backend,
        method_name="two_prompt_pride",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
    )


class TestTwoStagePermutationRunnerRunOne:
    """Tests for TwoStagePermutationRunner.run_one execution flow."""

    def test_correct_combined(self, runner_question_row):
        """Free text correct, all permutations match — should score correct."""
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = PermutationRunner._generate_permutations(canonical)

        responses = ["HTTPS"]
        for perm in perms:
            for letter, text in perm.items():
                if text == "HTTPS":
                    responses.append(letter)
                    break

        result = _make_runner(MockBackend(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_stage_one_failure_returns_early(self, runner_question_row):
        """If free-text call fails, should return immediately."""
        backend = MockBackend(responses=[ProviderTimeoutError("Request timed out.")])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert len(backend.requests_received) == 1

    def test_all_permutations_fail(self, runner_question_row):
        """Free text succeeds but all 4 matching calls fail — no valid vote."""
        responses = ["HTTPS"]
        responses.extend([ProviderTimeoutError("Request timed out.") for _ in range(4)])
        result = _make_runner(MockBackend(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["free_text_response"] == "HTTPS"

    def test_majority_wins(self, runner_question_row):
        """Three permutations correct, one wrong — majority should win."""
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = PermutationRunner._generate_permutations(canonical)

        responses = ["HTTPS"]
        for i, perm in enumerate(perms):
            if i == 0:
                responses.append("A")
            else:
                for letter, text in perm.items():
                    if text == "HTTPS":
                        responses.append(letter)
                        break

        result = _make_runner(MockBackend(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_makes_five_api_calls(self, runner_question_row):
        """Should fire 5 requests — 1 free text + 4 permutations."""
        responses = ["HTTPS"] + ["C"] * 4
        backend = MockBackend(responses=responses)
        _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert len(backend.requests_received) == 5

    def test_free_text_response_preserved(self, runner_question_row):
        """The intermediate free-text response should be in the result row."""
        responses = ["HTTPS"] + ["C"] * 4
        result = _make_runner(MockBackend(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["free_text_response"] == "HTTPS"
        assert result["free_text_prompt"] is not None
        assert result["free_text_latency"] is not None

    def test_result_row_metadata(self, runner_question_row):
        """Result row should carry trace metadata."""
        responses = ["HTTPS"] + ["C"] * 4
        result = _make_runner(MockBackend(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "two_prompt_pride"
        assert result["split_name"] == "robustness"

    def test_incorrect_combined(self, runner_question_row):
        """All permutations agree on wrong answer — should score incorrect."""
        responses = ["FTP"] + ["A"] * 4
        result = _make_runner(MockBackend(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT
