# tests/runners/test_visible_llm_matcher.py

from pathlib import Path

from choicebench.methods.library.visible_llm_matcher import VisibleLLMMatcherRunner
from choicebench.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend):
    return VisibleLLMMatcherRunner(
        backend=backend,
        method_name="visible_llm_matcher",
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
        seed=42,
        benchmark_name="mmlu",
    )


def _row_with_extracted_text(base_row, extracted_text):
    row = dict(base_row)
    row["extracted_text"] = extracted_text
    return row


class TestVisibleLLMMatcherRunOne:
    def test_one_new_call_per_question(self, runner_question_row):
        """Stage 1 (text_extraction) is reused, never re-called -- exactly
        one new call here, the Stage-2 LLM match."""
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        _make_runner(backend).run_one(row, sample_index=0)
        assert len(backend.requests_received) == 1

    def test_prompt_includes_the_reused_extracted_text_and_visible_options(self, runner_question_row):
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        _make_runner(backend).run_one(row, sample_index=0)
        prompt = backend.requests_received[0]
        assert "HTTPS" in prompt
        assert "A. FTP" in prompt
        assert "B. HTTP" in prompt
        assert "D. SMTP" in prompt

    def test_correct_match_scores_correct(self, runner_question_row):
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        result = _make_runner(backend).run_one(row, sample_index=0)
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_incorrect_match(self, runner_question_row):
        backend = MockBackend(responses=["A"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        result = _make_runner(backend).run_one(row, sample_index=0)
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_unparseable_response_is_unscorable(self, runner_question_row):
        backend = MockBackend(responses=["I cannot determine this."])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        result = _make_runner(backend).run_one(row, sample_index=0)
        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    def test_missing_extracted_text_column_raises(self, runner_question_row):
        import pytest
        with pytest.raises(KeyError, match="extracted_text"):
            _make_runner(MockBackend(responses=["C"])).run_one(runner_question_row, sample_index=0)

    def test_reused_extracted_text_is_persisted_on_the_row(self, runner_question_row):
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        result = _make_runner(backend).run_one(row, sample_index=0)
        assert result["reused_extracted_text"] == "HTTPS"
