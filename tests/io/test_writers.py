# tests/io/test_writers.py

import pandas as pd
import pytest

from choicebench.io.writers import (
    write_run_results,
)


@pytest.fixture
def identity_metadata() -> dict:
    """Identity columns required by the condition_id-provided write path."""
    return {
        "experiment_id": "exp_1", "condition_id": "cond_1", "dataset_artifact_id": "ds_1",
        "dataset_selection_id": "sel_1", "model_id": "model_1", "method_id": "method_1",
        "prompt_id": "prompt_1", "benchmark_split": "test",
    }


@pytest.fixture
def sample_results(identity_metadata) -> list[dict]:
    """Minimal result rows mimicking runner output, with identity columns."""
    return [
        {
            "question_id": "q_001",
            "parsed_choice": "C",
            "is_correct": True,
            **identity_metadata,
        },
        {
            "question_id": "q_002",
            "parsed_choice": "A",
            "is_correct": False,
            **identity_metadata,
        },
    ]


@pytest.fixture
def sample_two_stage_results(identity_metadata) -> list[dict]:
    """Result rows with extra two-stage fields."""
    return [
        {
            "question_id": "q_001",
            "parsed_choice": "C",
            "is_correct": True,
            "free_text_response": "HTTPS",
            "free_text_prompt": "Answer the question...",
            "free_text_latency": 0.5,
            **identity_metadata,
        },
    ]


class TestWriteRunResults:
    """Tests for write_run_results."""

    def test_creates_csv_file(self, tmp_path, sample_results):
        """Should create a CSV file at the expected condition-ID path."""
        path = write_run_results(sample_results, tmp_path, "cond_1")
        assert path.exists()
        assert path == tmp_path / "results" / "cond_1.csv"

    def test_writes_integrity_sidecar(self, tmp_path, sample_results):
        """Should write a companion artifact.json metadata sidecar."""
        path = write_run_results(sample_results, tmp_path, "cond_1")
        assert path.with_suffix(".artifact.json").exists()

    def test_csv_contains_all_rows(self, tmp_path, sample_results):
        """Written CSV should have one row per result."""
        path = write_run_results(sample_results, tmp_path, "cond_1")
        df = pd.read_csv(path)
        assert len(df) == 2

    def test_csv_contains_all_columns(self, tmp_path, sample_results):
        """Written CSV should have all keys from the result dicts."""
        path = write_run_results(sample_results, tmp_path, "cond_1")
        df = pd.read_csv(path)
        for key in sample_results[0]:
            assert key in df.columns

    def test_empty_results_raises(self, tmp_path):
        """Empty results list should be refused, not written as an empty CSV."""
        with pytest.raises(RuntimeError, match="empty"):
            write_run_results([], tmp_path, "cond_1")

    def test_creates_output_dir(self, tmp_path, sample_results):
        """Should create the output directory if it doesn't exist."""
        nested = tmp_path / "deep" / "nested"
        path = write_run_results(sample_results, nested, "cond_1")
        assert path.exists()

    def test_two_stage_extra_columns(self, tmp_path, sample_two_stage_results):
        """Two-stage results with extra fields should write correctly."""
        path = write_run_results(sample_two_stage_results, tmp_path, "cond_1")
        df = pd.read_csv(path)
        assert "free_text_response" in df.columns
        assert df.iloc[0]["free_text_response"] == "HTTPS"
