# tests/runners/test_direct_mcq.py

from pathlib import Path

from choicebench.clients.types import ProviderTimeoutError
from choicebench.methods.library.direct_mcq import DirectMCQRunner
from choicebench.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE
from choicebench.parsing.types import PARSE_OK, PARSE_MISSING

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend, method_name="baseline"):
    return DirectMCQRunner(
        backend=backend,
        method_name=method_name,
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
    )


class TestDirectMCQRunnerRunOne:
    """Tests for DirectMCQRunner.run_one execution flow."""

    def test_correct_answer(self, runner_question_row):
        """Model returns the correct letter — should parse and score correct."""
        result = _make_runner(MockBackend(responses=["C"])).run_one(
            runner_question_row, sample_index=0
        )

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT
        assert result["parse_status"] == PARSE_OK

    def test_incorrect_answer(self, runner_question_row):
        """Model returns a wrong letter — should parse and score incorrect."""
        result = _make_runner(MockBackend(responses=["A"])).run_one(
            runner_question_row, sample_index=0
        )

        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_failed_response(self, runner_question_row):
        """Model call fails — parsed and score fields should be None."""
        backend = MockBackend(responses=[ProviderTimeoutError("Request timed out.")])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["score_status"] is None
        assert result["error_type"] == "ProviderTimeoutError"

    def test_unparseable_response(self, runner_question_row):
        """Model returns gibberish — should parse as missing, score unscorable."""
        backend = MockBackend(responses=["I'm not sure about this question"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    def test_result_row_metadata(self, runner_question_row):
        """Result row should carry all trace metadata correctly."""
        result = _make_runner(MockBackend(responses=["C"])).run_one(
            runner_question_row, sample_index=3
        )

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "baseline"
        assert result["split_name"] == "robustness"
        assert result["subject"] == "computer_security"
        assert result["provider"] == "openai"
        assert result["model_name"] == "gpt-4.1-mini"
        assert result["sample_index"] == 3

    def test_prompt_contains_question_and_options(self, runner_question_row):
        """The prompt sent to the model should include the question and all options."""
        result = _make_runner(MockBackend(responses=["C"])).run_one(
            runner_question_row, sample_index=0
        )

        assert "securely browse websites" in result["prompt"]
        assert "FTP" in result["prompt"]
        assert "HTTP" in result["prompt"]
        assert "HTTPS" in result["prompt"]
        assert "SMTP" in result["prompt"]

    def test_lowercase_answer_parsed(self, runner_question_row):
        """Model returns lowercase letter — should still parse correctly."""
        result = _make_runner(MockBackend(responses=["c"])).run_one(
            runner_question_row, sample_index=0
        )

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_missing_trailing_option_is_not_rendered_or_parsed(self, runner_question_row):
        row = dict(runner_question_row, choice_d="", correct_option="C")
        backend = MockBackend(responses=["D"])

        result = _make_runner(backend).run_one(row, sample_index=0)

        assert "D." not in result["prompt"]
        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE


class TestDirectMCQRunnerBuildPrompt:
    """Tests for DirectMCQRunner._build_prompt via a live runner."""

    def test_prompt_format(self, runner_question_row):
        """Prompt should be a non-empty string containing the question."""
        runner = _make_runner(MockBackend(responses=[]))
        prompt = runner._build_prompt(runner_question_row)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert runner_question_row["question_text"] in prompt

    def test_prompt_contains_all_options(self, runner_question_row):
        """Prompt should include all four option texts."""
        runner = _make_runner(MockBackend(responses=[]))
        prompt = runner._build_prompt(runner_question_row)

        assert "FTP" in prompt
        assert "HTTP" in prompt
        assert "HTTPS" in prompt
        assert "SMTP" in prompt

    def test_prompt_omits_missing_trailing_option(self, runner_question_row):
        runner = _make_runner(MockBackend(responses=[]))
        prompt = runner._build_prompt(dict(runner_question_row, choice_d=""))

        assert "A. FTP" in prompt
        assert "B. HTTP" in prompt
        assert "C. HTTPS" in prompt
        assert "D." not in prompt
