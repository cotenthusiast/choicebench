# tests/test_preflight.py

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from choicebench.config.schema import BenchmarkConfig, MethodConfig, PreflightConfig
from choicebench.preflight import load_preflight


def _make_method(preflight: PreflightConfig | None = None) -> MethodConfig:
    return MethodConfig(name="direct_mcq", preflight=preflight)


def _make_benchmark(split: str = "test") -> BenchmarkConfig:
    return BenchmarkConfig(name="toy", split=split)


def _make_questions(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question_id": f"q{i:03d}",
                "subject": "general",
                "question_text": f"Question {i}?",
                "choice_a": "A text",
                "choice_b": "B text",
                "choice_c": "C text",
                "choice_d": "D text",
                "correct_option": "A",
                "correct_answer_text": "A text",
            }
            for i in range(n)
        ]
    )


class TestLoadPreflightNone:
    def test_no_preflight_block_returns_none(self):
        """No preflight block → load_preflight returns None."""
        method = _make_method(preflight=None)
        bench = _make_benchmark()
        result = load_preflight(method, bench, run_seed=42)
        assert result is None


class TestLoadPreflightBenchmarkSource:
    def test_valid_split_returns_n_records(self):
        """source: benchmark with valid (different) split returns n dicts."""
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=5))
        bench = _make_benchmark(split="test")

        fake_df = _make_questions(20)
        with patch("choicebench.preflight._load_benchmark_df", return_value=fake_df):
            result = load_preflight(method, bench, run_seed=42)

        assert isinstance(result, list)
        assert len(result) == 5
        for record in result:
            assert "question_id" in record

    def test_split_matches_eval_split_raises(self):
        """source: benchmark, preflight split == eval split → ValueError."""
        method = _make_method(PreflightConfig(source="benchmark", split="test", n=10))
        bench = _make_benchmark(split="test")

        with pytest.raises(ValueError, match="disjoint"):
            load_preflight(method, bench, run_seed=42)

    def test_n_larger_than_dataset_clamps(self):
        """Requesting more rows than available returns all rows."""
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=50))
        bench = _make_benchmark(split="test")

        fake_df = _make_questions(10)
        with patch("choicebench.preflight._load_benchmark_df", return_value=fake_df):
            result = load_preflight(method, bench, run_seed=42)

        assert len(result) == 10

    def test_reproducibility_same_seed(self):
        """Same seed must return the same sample across two calls."""
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=5))
        bench = _make_benchmark(split="test")
        fake_df = _make_questions(20)

        with patch("choicebench.preflight._load_benchmark_df", return_value=fake_df):
            first = load_preflight(method, bench, run_seed=7)
        with patch("choicebench.preflight._load_benchmark_df", return_value=fake_df):
            second = load_preflight(method, bench, run_seed=7)

        assert [r["question_id"] for r in first] == [r["question_id"] for r in second]

    def test_reproducibility_different_seeds_differ(self):
        """Different seeds should (almost always) return different samples."""
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=5))
        bench = _make_benchmark(split="test")
        fake_df = _make_questions(20)

        with patch("choicebench.preflight._load_benchmark_df", return_value=fake_df):
            first = load_preflight(method, bench, run_seed=1)
        with patch("choicebench.preflight._load_benchmark_df", return_value=fake_df):
            second = load_preflight(method, bench, run_seed=999)

        assert [r["question_id"] for r in first] != [r["question_id"] for r in second]


class TestLoadPreflightFilePath:
    def test_jsonl_source_returns_records(self, tmp_path: Path):
        """source: JSONL file path → returns parsed records."""
        records = [
            {"question_id": "j1", "question_text": "Q1?", "choice_a": "a"},
            {"question_id": "j2", "question_text": "Q2?", "choice_a": "b"},
        ]
        jsonl_path = tmp_path / "preflight.jsonl"
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        method = _make_method(PreflightConfig(source=str(jsonl_path), split="validation", n=100))
        bench = _make_benchmark(split="test")
        result = load_preflight(method, bench, run_seed=42)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["question_id"] == "j1"
        assert result[1]["question_id"] == "j2"

    def test_csv_source_returns_records(self, tmp_path: Path):
        """source: CSV file path → returns parsed records."""
        df = _make_questions(3)
        csv_path = tmp_path / "preflight.csv"
        df.to_csv(csv_path, index=False)

        method = _make_method(PreflightConfig(source=str(csv_path), split="validation", n=100))
        bench = _make_benchmark(split="test")
        result = load_preflight(method, bench, run_seed=42)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_unsupported_extension_raises(self, tmp_path: Path):
        """Unsupported file type → ValueError."""
        bad_path = tmp_path / "data.parquet"
        bad_path.write_text("not real parquet")

        method = _make_method(PreflightConfig(source=str(bad_path), split="validation", n=10))
        bench = _make_benchmark(split="test")
        with pytest.raises(ValueError, match="Unsupported preflight source"):
            load_preflight(method, bench, run_seed=42)
