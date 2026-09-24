# tests/scripts/test_paper_run_two_stage_rotations.py

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from choicebench.config.schema import GenerationKwargsConfig, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "paper" / "run_two_stage_rotations.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_two_stage_rotations", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()

_DUMMY_MODEL_CONFIG = ModelConfig(
    backend="dummy", model_name_or_path="dummy-model",
    generation_kwargs=GenerationKwargsConfig(max_new_tokens=1024, temperature=0.0),
)


def _two_stage_csv(tmp_path, rows) -> Path:
    """rows: list of (question_id, free_text_response) -- free_text_response
    may be None to simulate a stage-1 failure in the source run."""
    choices_json = json.dumps([
        {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
        {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
    ])
    df = pd.DataFrame([
        dict(question_id=qid, benchmark_name="mmlu", subject="computer_security",
             question_text="Which protocol is primarily used to securely browse websites?",
             choices_json=choices_json, correct_option="C", free_text_response=free_text)
        for qid, free_text in rows
    ])
    path = tmp_path / "two_stage_result.csv"
    df.to_csv(path, index=False)
    return path


class TestRunTwoStageRotations:
    def test_writes_one_row_per_question(self, tmp_path):
        source = _two_stage_csv(tmp_path, [("q1", "HTTPS"), ("q2", "FTP")])
        output = tmp_path / "out.csv"

        n_written = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)

        assert n_written == 2
        result_df = pd.read_csv(output)
        assert len(result_df) == 2
        assert set(result_df["question_id"]) == {"q1", "q2"}

    def test_method_name_is_stamped_correctly(self, tmp_path):
        source = _two_stage_csv(tmp_path, [("q1", "HTTPS")])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        result_df = pd.read_csv(output)
        assert result_df.iloc[0]["method_name"] == "two_stage"

    def test_free_text_is_reused_not_regenerated(self, tmp_path):
        source = _two_stage_csv(tmp_path, [("q1", "HTTPS")])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        result_df = pd.read_csv(output)
        assert result_df.iloc[0]["free_text_response"] == "HTTPS"

    def test_per_rotation_choices_json_is_persisted(self, tmp_path):
        source = _two_stage_csv(tmp_path, [("q1", "HTTPS")])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        result_df = pd.read_csv(output)
        per_rotation = json.loads(result_df.iloc[0]["per_rotation_choices_json"])
        assert len(per_rotation) == 4

    def test_rows_with_missing_free_text_response_are_skipped(self, tmp_path):
        """A stage-1 failure in the source run leaves free_text_response
        empty -- there is no free-text answer to reuse, so this question
        cannot be rotation-rerun and must be skipped, not crash the run."""
        source = _two_stage_csv(tmp_path, [("q1", "HTTPS"), ("q2", None)])
        output = tmp_path / "out.csv"

        n_written = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)

        assert n_written == 1
        result_df = pd.read_csv(output)
        assert set(result_df["question_id"]) == {"q1"}

    def test_resume_skips_already_completed_question_ids(self, tmp_path):
        source = _two_stage_csv(tmp_path, [("q1", "HTTPS"), ("q2", "FTP"), ("q3", "HTTP")])
        output = tmp_path / "out.csv"

        n_first = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_first == 3

        n_second = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_second == 0

        result_df = pd.read_csv(output)
        assert len(result_df) == 3  # not 6 -- no duplicates
        assert sorted(result_df["question_id"]) == ["q1", "q2", "q3"]

    def test_resume_after_partial_completion_only_does_remaining_work(self, tmp_path):
        source = _two_stage_csv(tmp_path, [("q1", "HTTPS"), ("q2", "FTP")])
        output = tmp_path / "out.csv"

        existing = pd.DataFrame([{
            "question_id": "q1", "method_name": "two_stage",
            "parsed_choice": "C", "is_correct": True,
        }])
        existing.to_csv(output, index=False)

        n_written = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_written == 1  # only q2 was actually new work

    def test_missing_free_text_response_column_raises_clear_error(self, tmp_path):
        df = pd.DataFrame([{"question_id": "q1"}])
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="free_text_response"):
            mod.run(path, _DUMMY_MODEL_CONFIG, "test_run", tmp_path / "out.csv", run_seed=42)
