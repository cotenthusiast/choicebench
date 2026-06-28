# tests/runners/test_two_stage.py

from pathlib import Path

from choicebench.methods.library.two_stage import TwoStageRunner
from choicebench.clients.types import ProviderTimeoutError
from choicebench.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE
from choicebench.parsing.types import PARSE_OK, PARSE_MISSING

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend):
    return TwoStageRunner(
        backend=backend,
        method_name="two_prompt",
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
        assert result["method_name"] == "two_prompt"
        assert result["split_name"] == "robustness"

    def test_unparseable_matching_response(self, runner_question_row):
        """Stage 2 returns gibberish — should be unscorable."""
        backend = MockBackend(responses=["HTTPS", "I think it might be one of those"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE
        assert result["free_text_response"] == "HTTPS"
