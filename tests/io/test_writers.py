# tests/io/test_writers.py

import pandas as pd
import pytest

from choicebench.io.writers import (
    write_run_results,
)


@pytest.fixture
def sample_results() -> list[dict]:
    """Minimal result rows mimicking runner output."""
    return [
        {
            "run_id": "run_001",
            "question_id": "q_001",
            "method_name": "baseline",
            "model_name": "gpt-5-mini",
            "parsed_choice": "C",
            "is_correct": True,
        },
        {
            "run_id": "run_001",
            "question_id": "q_002",
            "method_name": "baseline",
            "model_name": "gpt-5-mini",
            "parsed_choice": "A",
            "is_correct": False,
        },
    ]


@pytest.fixture
def sample_two_stage_results() -> list[dict]:
    """Result rows with extra two-stage fields."""
    return [
        {
            "run_id": "run_001",
            "question_id": "q_001",
            "method_name": "two_stage",
            "model_name": "gpt-5-mini",
            "parsed_choice": "C",
            "is_correct": True,
            "free_text_response": "HTTPS",
            "free_text_prompt": "Answer the question...",
            "free_text_latency": 0.5,
        },
    ]


class TestWriteRunResults:
    """Tests for write_run_results."""

    def test_creates_csv_file(self, tmp_path, sample_results):
        """Should create a CSV file at the expected path."""
        path = write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        assert path.exists()
        assert path.suffix == ".csv"

    def test_filename_encodes_identifiers(self, tmp_path, sample_results):
        """Filename should contain run_id, method, and model."""
        path = write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        assert "run_001" in path.name
        assert "baseline" in path.name
        assert "gpt-5-mini" in path.name

    def test_csv_contains_all_rows(self, tmp_path, sample_results):
        """Written CSV should have one row per result."""
        path = write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        df = pd.read_csv(path)
        assert len(df) == 2

    def test_csv_contains_all_columns(self, tmp_path, sample_results):
        """Written CSV should have all keys from the result dicts."""
        path = write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        df = pd.read_csv(path)
        for key in sample_results[0]:
            assert key in df.columns

    def test_empty_results(self, tmp_path):
        """Empty results list should create an empty CSV."""
        path = write_run_results([], tmp_path, "run_001", "baseline", "gpt-5-mini")
        assert path.exists()

    def test_creates_output_dir(self, tmp_path, sample_results):
        """Should create the output directory if it doesn't exist."""
        nested = tmp_path / "deep" / "nested"
        path = write_run_results(
            sample_results, nested, "run_001", "baseline", "gpt-5-mini"
        )
        assert path.exists()

    def test_two_stage_extra_columns(self, tmp_path, sample_two_stage_results):
        """Two-stage results with extra fields should write correctly."""
        path = write_run_results(
            sample_two_stage_results,
            tmp_path,
            "run_001",
            "two_stage",
            "gpt-5-mini",
        )
        df = pd.read_csv(path)
        assert "free_text_response" in df.columns
        assert df.iloc[0]["free_text_response"] == "HTTPS"
