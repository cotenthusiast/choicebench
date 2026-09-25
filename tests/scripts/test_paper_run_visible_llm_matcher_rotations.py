# tests/scripts/test_paper_run_visible_llm_matcher_rotations.py

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from choicebench.config.schema import GenerationKwargsConfig, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "paper" / "run_visible_llm_matcher_rotations.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_visible_llm_matcher_rotations", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()

_DUMMY_MODEL_CONFIG = ModelConfig(
    backend="dummy", model_name_or_path="dummy-model",
    generation_kwargs=GenerationKwargsConfig(max_new_tokens=1024, temperature=0.0),
)


def _text_extraction_rotations_csv(tmp_path, rows) -> Path:
    """rows: list of (question_id, per_rotation_raw_text list)."""
    choices_json = json.dumps([
        {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
        {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
    ])
    df = pd.DataFrame([
        dict(question_id=qid, benchmark_name="mmlu", subject="computer_security",
             question_text="Which protocol is primarily used to securely browse websites?",
             choices_json=choices_json, correct_option="C",
             per_rotation_raw_text_json=json.dumps(per_rotation_text))
        for qid, per_rotation_text in rows
    ])
    path = tmp_path / "text_extraction_rotations_result.csv"
    df.to_csv(path, index=False)
    return path


class TestRunVisibleLLMMatcherRotations:
    def test_writes_one_row_per_question(self, tmp_path):
        source = _text_extraction_rotations_csv(tmp_path, [
            ("q1", ["HTTPS", "HTTPS", "HTTPS", "HTTPS"]),
            ("q2", ["FTP", "FTP", "FTP", "FTP"]),
        ])
        output = tmp_path / "out.csv"

        n_written = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)

        assert n_written == 2
        result_df = pd.read_csv(output)
        assert len(result_df) == 2
        assert set(result_df["question_id"]) == {"q1", "q2"}

    def test_method_name_is_stamped_correctly(self, tmp_path):
        source = _text_extraction_rotations_csv(tmp_path, [
            ("q1", ["HTTPS", "HTTPS", "HTTPS", "HTTPS"]),
        ])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        result_df = pd.read_csv(output)
        assert result_df.iloc[0]["method_name"] == "visible_llm_matcher"

    def test_per_rotation_choices_json_is_persisted(self, tmp_path):
        source = _text_extraction_rotations_csv(tmp_path, [
            ("q1", ["HTTPS", "HTTPS", "HTTPS", "HTTPS"]),
        ])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        result_df = pd.read_csv(output)
        assert len(json.loads(result_df.iloc[0]["per_rotation_choices_json"])) == 4

    def test_rows_with_all_null_extracted_text_still_produce_a_row(self, tmp_path):
        """Every rotation's upstream Stage-1 call failed -- no calls are
        made here, but the row must still be written (unscorable), not
        silently dropped."""
        source = _text_extraction_rotations_csv(tmp_path, [
            ("q1", [None, None, None, None]),
        ])
        output = tmp_path / "out.csv"
        n_written = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_written == 1
        result_df = pd.read_csv(output)
        assert pd.isna(result_df.iloc[0]["parsed_choice"])

    def test_resume_skips_already_completed_question_ids(self, tmp_path):
        source = _text_extraction_rotations_csv(tmp_path, [
            ("q1", ["HTTPS"] * 4), ("q2", ["FTP"] * 4), ("q3", ["HTTP"] * 4),
        ])
        output = tmp_path / "out.csv"

        n_first = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_first == 3

        n_second = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_second == 0

        result_df = pd.read_csv(output)
        assert len(result_df) == 3
        assert sorted(result_df["question_id"]) == ["q1", "q2", "q3"]

    def test_resume_after_partial_completion_only_does_remaining_work(self, tmp_path):
        # Realistic partial state: an actual prior invocation of this same
        # script, not a hand-built fixture with a different column set --
        # a real resume's existing file always has the matching schema,
        # since only this script's own writer ever produced it.
        source = _text_extraction_rotations_csv(tmp_path, [
            ("q1", ["HTTPS"] * 4), ("q2", ["FTP"] * 4),
        ])
        output = tmp_path / "out.csv"

        n_first = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_first == 2

        source2 = _text_extraction_rotations_csv(tmp_path, [
            ("q1", ["HTTPS"] * 4), ("q2", ["FTP"] * 4), ("q3", ["HTTP"] * 4),
        ])
        n_written = mod.run(source2, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_written == 1

    def test_missing_per_rotation_raw_text_json_column_raises_clear_error(self, tmp_path):
        df = pd.DataFrame([{"question_id": "q1"}])
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="per_rotation_raw_text_json"):
            mod.run(path, _DUMMY_MODEL_CONFIG, "test_run", tmp_path / "out.csv", run_seed=42)

    def test_a_schema_mismatched_existing_output_file_raises_clearly(self, tmp_path):
        """A stale output file from an older script version (fewer/
        different columns) must fail loudly, not silently have new-format
        rows appended into it, corrupting the CSV's per-row field count."""
        source = _text_extraction_rotations_csv(tmp_path, [
            ("q1", ["HTTPS"] * 4), ("q2", ["FTP"] * 4),
        ])
        output = tmp_path / "out.csv"
        stale = pd.DataFrame([{"question_id": "q1", "method_name": "visible_llm_matcher"}])
        stale.to_csv(output, index=False)

        with pytest.raises(ValueError, match="schema"):
            mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
