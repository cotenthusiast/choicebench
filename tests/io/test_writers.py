# tests/io/test_writers.py

import json

import pandas as pd
import pytest

from mcq_eval.io.writers import (
    write_normalized_questions,
    write_run_results,
    write_split_ids,
    write_split_metadata,
    write_group_splits,
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
            "method_name": "two_prompt",
            "model_name": "gpt-5-mini",
            "parsed_choice": "C",
            "is_correct": True,
            "free_text_response": "HTTPS",
            "free_text_prompt": "Answer the question...",
            "free_text_latency": 0.5,
        },
    ]


class TestWriteNormalizedQuestions:
    """Tests for write_normalized_questions."""

    def test_write_normalized_questions(self, sample_raw_dataframe, sample_normalized_dataframe, tmp_path):
        raw_csv_path = tmp_path / "raw.csv"
        normalized_csv_path = tmp_path / "normalized.csv"

        sample_raw_dataframe.to_csv(raw_csv_path, index=False)

        write_normalized_questions(
            raw_questions_path=raw_csv_path,
            normalized_questions_path=normalized_csv_path,
        )

        actual_df = pd.read_csv(normalized_csv_path)
        pd.testing.assert_frame_equal(actual_df, sample_normalized_dataframe)


class TestWriteSplitArtifacts:
    """Tests for write_split_ids, write_split_metadata, and write_group_splits."""

    def test_write_split_ids_writes_json_id_list(self, tmp_path):
        split_ids = ["q_001", "q_014", "q_203", "q_417"]
        write_split_ids(split_ids, "robustness", tmp_path, "benchmark")
        with open(tmp_path / "benchmark" / "robustness_ids.json", "r", encoding="utf-8") as f:
            actual = json.load(f)
        assert actual == split_ids

    def test_write_split_metadata_writes_json_metadata_dict(self, tmp_path):
        split_ids = ["q_001", "q_014", "q_203", "q_417"]
        split_metadata = {
            "split_name": "robustness",
            "split_ids": split_ids,
            "subjects": ["anatomy", "economics"],
            "per_subject": 2,
            "seed": 42,
            "strategy": "balanced_subject_sample",
            "actual_size": 4,
            "actual_subject_counts": {
                "anatomy": 2,
                "economics": 2,
            },
            "eligible_pool_size": 120,
            "excluded_id_count": 10,
        }
        write_split_metadata(split_metadata, "robustness", tmp_path, "benchmark")
        with open(tmp_path / "benchmark" / "robustness_metadata.json", "r", encoding="utf-8") as f:
            actual = json.load(f)
        assert actual == split_metadata

    def test_write_group_splits_writes_all_expected_split_artifacts(self, tmp_path):
        split_ids = ["q_001", "q_014", "q_203", "q_417"]
        split_metadata = {
            "split_name": "robustness",
            "split_ids": split_ids,
            "subjects": ["anatomy", "economics"],
            "per_subject": 2,
            "seed": 42,
            "strategy": "balanced_subject_sample",
            "actual_size": 4,
            "actual_subject_counts": {
                "anatomy": 2,
                "economics": 2,
            },
            "eligible_pool_size": 120,
            "excluded_id_count": 10,
        }

        split_artifacts = {
            "robustness": {
                "ids": split_ids,
                "metadata": split_metadata,
            }
        }

        write_group_splits(split_artifacts, tmp_path, "benchmark")

        with open(tmp_path / "benchmark" / "robustness_ids.json", "r", encoding="utf-8") as f:
            actual_ids = json.load(f)

        with open(tmp_path / "benchmark" / "robustness_metadata.json", "r", encoding="utf-8") as f:
            actual_metadata = json.load(f)

        assert actual_ids == split_ids
        assert actual_metadata == split_metadata


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
            "two_prompt",
            "gpt-5-mini",
        )
        df = pd.read_csv(path)
        assert "free_text_response" in df.columns
        assert df.iloc[0]["free_text_response"] == "HTTPS"
