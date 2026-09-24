# tests/scripts/test_paper_run_visible_llm_matcher.py

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from choicebench.config.schema import GenerationKwargsConfig, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "paper" / "run_visible_llm_matcher.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_visible_llm_matcher", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()

_DUMMY_MODEL_CONFIG = ModelConfig(
    backend="dummy", model_name_or_path="dummy-model",
    generation_kwargs=GenerationKwargsConfig(max_new_tokens=1024, temperature=0.0),
)


def _text_extraction_csv(tmp_path, rows) -> Path:
    choices_json = json.dumps([
        {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
        {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
    ])
    df = pd.DataFrame([
        dict(question_id=qid, benchmark_name="mmlu", subject="computer_security",
             question_text="Which protocol is primarily used to securely browse websites?",
             choices_json=choices_json, correct_option="C", raw_text=raw_text)
        for qid, raw_text in rows
    ])
    path = tmp_path / "text_extraction_result.csv"
    df.to_csv(path, index=False)
    return path


class TestRunVisibleLLMMatcher:
    def test_writes_one_row_per_question(self, tmp_path):
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS"), ("q2", "FTP")])
        output = tmp_path / "out.csv"

        n_written = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)

        assert n_written == 2
        result_df = pd.read_csv(output)
        assert len(result_df) == 2
        assert set(result_df["question_id"]) == {"q1", "q2"}

    def test_method_name_is_stamped_correctly(self, tmp_path):
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS")])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        result_df = pd.read_csv(output)
        assert result_df.iloc[0]["method_name"] == "visible_llm_matcher"

    def test_extracted_text_is_reused_not_regenerated(self, tmp_path):
        """The saved text_extraction raw_text must be threaded through as
        Stage 1's reused output, never re-elicited."""
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS")])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        result_df = pd.read_csv(output)
        assert result_df.iloc[0]["reused_extracted_text"] == "HTTPS"

    def test_resume_skips_already_completed_question_ids(self, tmp_path):
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS"), ("q2", "FTP"), ("q3", "HTTP")])
        output = tmp_path / "out.csv"

        n_first = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_first == 3

        # Simulate a fresh process restart on the same output file: nothing
        # new to do, must write zero additional rows and never duplicate.
        n_second = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_second == 0

        result_df = pd.read_csv(output)
        assert len(result_df) == 3  # not 6 -- no duplicates
        assert sorted(result_df["question_id"]) == ["q1", "q2", "q3"]

    def test_resume_after_partial_completion_only_does_remaining_work(self, tmp_path):
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS"), ("q2", "FTP")])
        output = tmp_path / "out.csv"

        # Simulate a crash after q1 completed: pre-seed the output file
        # with just that one row.
        existing = pd.DataFrame([{
            "question_id": "q1", "method_name": "visible_llm_matcher",
            "parsed_choice": "C", "is_correct": True,
        }])
        existing.to_csv(output, index=False)

        n_written = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_written == 1  # only q2 was actually new work

    def test_no_resume_flag_reprocesses_everything(self, tmp_path):
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS")])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        n_second = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42, resume=False)
        assert n_second == 1
        # Non-resumed run appends again -- this is a deliberate "start over"
        # escape hatch, not the default path, so duplicate rows here are
        # expected/intentional, not a bug.
        result_df = pd.read_csv(output)
        assert len(result_df) == 2

    def test_missing_raw_text_column_raises_clear_error(self, tmp_path):
        df = pd.DataFrame([{"question_id": "q1"}])
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="raw_text"):
            mod.run(path, _DUMMY_MODEL_CONFIG, "test_run", tmp_path / "out.csv", run_seed=42)
