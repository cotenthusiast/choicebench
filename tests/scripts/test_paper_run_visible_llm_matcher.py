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
        # Realistic partial state: an actual prior invocation of this same
        # script, not a hand-built fixture with a different column set --
        # a real resume's existing file always has the matching schema,
        # since only this script's own writer ever produced it.
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS"), ("q2", "FTP")])
        output = tmp_path / "out.csv"

        n_first = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_first == 2

        source2 = _text_extraction_csv(tmp_path, [("q1", "HTTPS"), ("q2", "FTP"), ("q3", "HTTP")])
        n_written = mod.run(source2, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_written == 1  # only q3 was actually new work

    def test_no_resume_flag_reprocesses_everything(self, tmp_path):
        """Corrected 2026-09-25 (confirmed conformance gap): --no-resume is
        a "start fresh" escape hatch -- it must truncate any pre-existing
        output first, never append scientifically duplicate rows onto it."""
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS")])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        n_second = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42, resume=False)
        assert n_second == 1
        result_df = pd.read_csv(output)
        assert len(result_df) == 1
        assert not result_df["question_id"].duplicated().any()

    def test_no_resume_against_existing_output_with_multiple_questions_does_not_duplicate(self, tmp_path):
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS"), ("q2", "FTP")])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42, resume=True)
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42, resume=False)
        result_df = pd.read_csv(output)
        assert not result_df["question_id"].duplicated().any(), (
            f"--no-resume duplicated rows: {result_df['question_id'].value_counts().to_dict()}"
        )

    def test_conflicting_duplicate_rows_in_existing_output_raise_on_resume(self, tmp_path):
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS"), ("q2", "FTP")])
        output = tmp_path / "out.csv"
        conflicting = pd.DataFrame([
            {"question_id": "q1", "parsed_choice": "A", "method_name": "visible_llm_matcher"},
            {"question_id": "q1", "parsed_choice": "B", "method_name": "visible_llm_matcher"},
        ])
        conflicting.to_csv(output, index=False)

        with pytest.raises(ValueError, match="conflicting"):
            mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42, resume=True)

    def test_missing_raw_text_column_raises_clear_error(self, tmp_path):
        df = pd.DataFrame([{"question_id": "q1"}])
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="raw_text"):
            mod.run(path, _DUMMY_MODEL_CONFIG, "test_run", tmp_path / "out.csv", run_seed=42)

    def test_wrong_model_text_extraction_csv_is_rejected(self, tmp_path):
        """--text-extraction-csv accidentally pointing at a DIFFERENT
        model's saved output (a real, plausible operational mistake) must
        be rejected, not silently mixed in as this run's Stage 1."""
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS")])
        df = pd.read_csv(source)
        df["model_name"] = "some-other-model-entirely"  # not "dummy-model"
        df.to_csv(source, index=False)
        output = tmp_path / "out.csv"

        with pytest.raises(ValueError, match="extracted_text_source_model"):
            mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)

    def test_a_schema_mismatched_existing_output_file_raises_clearly(self, tmp_path):
        """A stale output file from an older script version (fewer/
        different columns) must fail loudly, not silently have new-format
        rows appended into it, corrupting the CSV's per-row field count."""
        # q2 is genuinely pending (not in the stale file) so the run
        # actually reaches the point where it would append -- q1 alone
        # would look fully "completed" already and never get that far.
        source = _text_extraction_csv(tmp_path, [("q1", "HTTPS"), ("q2", "FTP")])
        output = tmp_path / "out.csv"
        stale = pd.DataFrame([{"question_id": "q1", "method_name": "visible_llm_matcher"}])
        stale.to_csv(output, index=False)

        with pytest.raises(ValueError, match="schema"):
            mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
