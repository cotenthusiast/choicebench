# tests/io/test_readers.py

import pandas as pd
import pytest

from choicebench.io.readers import (
    _infer_benchmark_from_stem,
    read_all_run_results,
    read_run_results,
)
from choicebench.io.writers import write_run_results


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


class TestReadRunResults:
    """Tests for read_run_results."""

    def test_roundtrip(self, tmp_path, sample_results):
        """Write then read should return equivalent data."""
        path = write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        df = read_run_results(path)
        assert len(df) == 2
        assert "question_id" in df.columns

    def test_returns_dataframe(self, tmp_path, sample_results):
        """Should return a pandas DataFrame."""
        path = write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        df = read_run_results(path)
        assert isinstance(df, pd.DataFrame)

    def test_preserves_values(self, tmp_path, sample_results):
        """Read values should match what was written."""
        path = write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        df = read_run_results(path)
        assert df.iloc[0]["question_id"] == "q_001"
        assert df.iloc[1]["question_id"] == "q_002"

    def test_two_stage_extra_fields(self, tmp_path, sample_two_stage_results):
        """Should preserve two-stage extra fields through roundtrip."""
        path = write_run_results(
            sample_two_stage_results,
            tmp_path,
            "run_001",
            "two_stage",
            "gpt-5-mini",
        )
        df = read_run_results(path)
        assert df.iloc[0]["free_text_response"] == "HTTPS"


class TestReadAllRunResults:
    """Tests for read_all_run_results."""

    def test_reads_multiple_files(self, tmp_path, sample_results):
        """Should combine results from multiple CSV files."""
        write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        write_run_results(
            sample_results, tmp_path, "run_001", "two_stage", "gpt-5-mini"
        )
        df = read_all_run_results(tmp_path)
        assert len(df) == 4

    def test_filter_by_method(self, tmp_path, sample_results):
        """Should return only files matching the method filter."""
        write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        write_run_results(
            sample_results, tmp_path, "run_001", "two_stage", "gpt-5-mini"
        )
        df = read_all_run_results(tmp_path, method_name="baseline")
        assert len(df) == 2

    def test_filter_by_model(self, tmp_path, sample_results):
        """Should return only files matching the model filter."""
        write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gemini-2.5-flash"
        )
        df = read_all_run_results(tmp_path, model_name="gpt-5-mini")
        assert len(df) == 2

    def test_filter_by_run_id(self, tmp_path, sample_results):
        """Should return only files matching the run ID filter."""
        write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        write_run_results(
            sample_results, tmp_path, "run_002", "baseline", "gpt-5-mini"
        )
        df = read_all_run_results(tmp_path, run_id="run_001")
        assert len(df) == 2

    def test_empty_directory(self, tmp_path):
        """Empty directory should return empty DataFrame."""
        df = read_all_run_results(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_no_matching_files(self, tmp_path, sample_results):
        """No files match filter — should return empty DataFrame."""
        write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        df = read_all_run_results(tmp_path, method_name="nonexistent")
        assert len(df) == 0


def test_legacy_filename_benchmark_inference_knows_registered_benchmarks():
    """Older CSVs without benchmark_name should still infer current built-ins."""
    for benchmark in [
        "arc_challenge",
        "hellaswag",
        "huggingface",
        "mmlu",
        "mmlu_pro",
        "truthful_qa",
        "toy",
    ]:
        stem = f"run_001_direct_mcq_dummy_model_{benchmark}"
        assert _infer_benchmark_from_stem(stem) == benchmark
