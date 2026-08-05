# tests/io/test_readers.py

import pandas as pd
import pytest

from choicebench.io.writers import write_run_results


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
        {"question_id": "q_001", "parsed_choice": "C", "is_correct": True, **identity_metadata},
        {"question_id": "q_002", "parsed_choice": "A", "is_correct": False, **identity_metadata},
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


class TestWriteRunResultsRoundtrip:
    """Round-trip tests for write_run_results (read back with plain pandas)."""

    def test_roundtrip(self, tmp_path, sample_results):
        """Write then read should return equivalent data."""
        path = write_run_results(sample_results, tmp_path, "cond_1")
        df = pd.read_csv(path)
        assert len(df) == 2
        assert "question_id" in df.columns

    def test_returns_dataframe(self, tmp_path, sample_results):
        """Should return a pandas DataFrame."""
        path = write_run_results(sample_results, tmp_path, "cond_1")
        df = pd.read_csv(path)
        assert isinstance(df, pd.DataFrame)

    def test_preserves_values(self, tmp_path, sample_results):
        """Read values should match what was written."""
        path = write_run_results(sample_results, tmp_path, "cond_1")
        df = pd.read_csv(path)
        assert df.iloc[0]["question_id"] == "q_001"
        assert df.iloc[1]["question_id"] == "q_002"

    def test_two_stage_extra_fields(self, tmp_path, sample_two_stage_results):
        """Should preserve two-stage extra fields through roundtrip."""
        path = write_run_results(sample_two_stage_results, tmp_path, "cond_1")
        df = pd.read_csv(path)
        assert df.iloc[0]["free_text_response"] == "HTTPS"
