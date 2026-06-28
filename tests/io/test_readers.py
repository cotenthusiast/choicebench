# tests/io/test_readers.py

import json

import pandas as pd
import pytest

from choicebench.io.readers import (
    read_all_run_results,
    read_group_splits,
    read_normalized_questions,
    read_raw_questions,
    read_run_results,
    read_split_ids,
    read_split_metadata,
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
            "method_name": "two_prompt",
            "model_name": "gpt-5-mini",
            "parsed_choice": "C",
            "is_correct": True,
            "free_text_response": "HTTPS",
            "free_text_prompt": "Answer the question...",
            "free_text_latency": 0.5,
        },
    ]


class TestReadRawAndNormalizedQuestions:
    """Tests for read_raw_questions and read_normalized_questions."""

    def test_read_raw_questions(self, tmp_path, sample_raw_dataframe):
        raw_csv_path = tmp_path / "raw.csv"
        data = (
            "subject,question,choices,answer\n"
            'computer_security,Which protocol is primarily used to securely browse websites?,"[""FTP"", ""HTTP"", ""HTTPS"", ""SMTP""]",2\n'
            'high_school_physics,What is the SI unit of force?,"[""Joule"", ""Newton"", ""Watt"", ""Pascal""]",1'
        )
        raw_csv_path.write_text(data)
        pd.testing.assert_frame_equal(read_raw_questions("raw.csv", raw_dir=tmp_path), sample_raw_dataframe)

    def test_read_normalized_questions(self, tmp_path, sample_normalized_dataframe):
        normalized_csv_path = tmp_path / "normalized.csv"
        data = (
            "question_id,subject,question_text,choice_a,choice_b,choice_c,choice_d,correct_option,correct_answer_text\n"
            "4865890d7f0efae8,computer_security,Which protocol is primarily used to securely browse websites?,FTP,HTTP,HTTPS,SMTP,C,HTTPS\n"
            "5e9876049bf053f9,high_school_physics,What is the SI unit of force?,Joule,Newton,Watt,Pascal,B,Newton"
        )
        normalized_csv_path.write_text(data)
        pd.testing.assert_frame_equal(read_normalized_questions("normalized.csv", processed_dir=tmp_path), sample_normalized_dataframe)


class TestReadSplitArtifacts:
    """Tests for read_split_ids, read_split_metadata, and read_group_splits."""

    def test_read_split_ids_returns_id_list(self, tmp_path):
        split_ids = ["q_001", "q_014", "q_203", "q_417"]
        group_dir = tmp_path / "benchmark"
        group_dir.mkdir()

        with open(group_dir / "robustness_ids.json", "w", encoding="utf-8") as f:
            json.dump(split_ids, f)

        actual_ids = read_split_ids("robustness", tmp_path, "benchmark")
        assert actual_ids == split_ids

    def test_read_split_metadata_returns_metadata_dict(self, tmp_path):
        split_metadata = {
            "split_name": "robustness",
            "split_ids": ["q_001", "q_014", "q_203", "q_417"],
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
        group_dir = tmp_path / "benchmark"
        group_dir.mkdir()

        with open(group_dir / "robustness_metadata.json", "w", encoding="utf-8") as f:
            json.dump(split_metadata, f)

        actual_ids = read_split_metadata("robustness", tmp_path, "benchmark")
        assert actual_ids == split_metadata

    def test_read_group_splits_returns_nested_payload_for_benchmark(self, tmp_path):
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

        group_dir = tmp_path / "benchmark"
        group_dir.mkdir()

        with open(group_dir / "robustness_ids.json", "w", encoding="utf-8") as f:
            json.dump(split_ids, f)

        with open(group_dir / "robustness_metadata.json", "w", encoding="utf-8") as f:
            json.dump(split_metadata, f)

        actual_data = read_group_splits("benchmark", tmp_path)

        expected_data = {
            "robustness": {
                "ids": split_ids,
                "metadata": split_metadata,
            }
        }

        assert actual_data == expected_data


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
            "two_prompt",
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
            sample_results, tmp_path, "run_001", "two_prompt", "gpt-5-mini"
        )
        df = read_all_run_results(tmp_path)
        assert len(df) == 4

    def test_filter_by_method(self, tmp_path, sample_results):
        """Should return only files matching the method filter."""
        write_run_results(
            sample_results, tmp_path, "run_001", "baseline", "gpt-5-mini"
        )
        write_run_results(
            sample_results, tmp_path, "run_001", "two_prompt", "gpt-5-mini"
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
