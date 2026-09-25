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


def _make_prepared_dataset(df: pd.DataFrame, artifact_id: str = "art_bench"):
    """Fake PreparedDataset for mocking _load_benchmark_artifact."""
    from choicebench.config.paths import PROCESSED_DIR
    from choicebench.datasets import PreparedDataset

    metadata = {"artifact_id": artifact_id, "content_digest": f"{artifact_id}_digest"}
    return PreparedDataset(
        path=PROCESSED_DIR / artifact_id / "normalized.csv",
        metadata_path=PROCESSED_DIR / artifact_id / "artifact.json",
        metadata=metadata,
        dataframe=df,
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

        fake_artifact = _make_prepared_dataset(_make_questions(20))
        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            result = load_preflight(method, bench, run_seed=42)

        assert isinstance(result, list)
        assert len(result) == 5
        for record in result:
            assert "question_id" in record

    def test_same_prepared_artifact_as_eval_raises(self):
        """source: benchmark resolving to the same prepared artifact as eval_artifact_id → ValueError."""
        method = _make_method(PreflightConfig(source="benchmark", split="test", n=10))
        bench = _make_benchmark(split="test")

        fake_artifact = _make_prepared_dataset(_make_questions(20), artifact_id="art_shared")
        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            with pytest.raises(ValueError, match="same prepared artifact"):
                load_preflight(method, bench, run_seed=42, eval_artifact_id="art_shared")

    def test_split_matches_eval_split_raises_without_eval_artifact_id(self):
        """No eval_artifact_id supplied: falls back to the split-label check,
        so preflight split == eval split still raises instead of silently
        allowing overlap."""
        method = _make_method(PreflightConfig(source="benchmark", split="test", n=10))
        bench = _make_benchmark(split="test")

        fake_artifact = _make_prepared_dataset(_make_questions(20))
        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            with pytest.raises(ValueError, match="disjoint"):
                load_preflight(method, bench, run_seed=42)

    def test_n_larger_than_dataset_clamps(self):
        """Requesting more rows than available returns all rows."""
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=50))
        bench = _make_benchmark(split="test")

        fake_artifact = _make_prepared_dataset(_make_questions(10))
        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            result = load_preflight(method, bench, run_seed=42)

        assert len(result) == 10

    def test_reproducibility_same_seed(self):
        """Same seed must return the same sample across two calls."""
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=5))
        bench = _make_benchmark(split="test")
        fake_artifact = _make_prepared_dataset(_make_questions(20))

        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            first = load_preflight(method, bench, run_seed=7)
        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            second = load_preflight(method, bench, run_seed=7)

        assert [r["question_id"] for r in first] == [r["question_id"] for r in second]

    def test_excludes_eval_question_ids(self):
        """eval_question_ids must never appear in the returned calibration sample."""
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=15))
        bench = _make_benchmark(split="test")
        fake_artifact = _make_prepared_dataset(_make_questions(20))
        eval_ids = {f"q{i:03d}" for i in range(15)}  # same seed/pool would otherwise overlap

        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            result = load_preflight(method, bench, run_seed=42, eval_question_ids=eval_ids)

        returned_ids = {r["question_id"] for r in result}
        assert returned_ids.isdisjoint(eval_ids)
        assert returned_ids == {f"q{i:03d}" for i in range(15, 20)}

    def test_reproducibility_different_seeds_differ(self):
        """Different seeds should (almost always) return different samples."""
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=5))
        bench = _make_benchmark(split="test")
        fake_artifact = _make_prepared_dataset(_make_questions(20))

        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            first = load_preflight(method, bench, run_seed=1)
        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            second = load_preflight(method, bench, run_seed=999)

        assert [r["question_id"] for r in first] != [r["question_id"] for r in second]


def _make_questions_with_n_choices(rows: list[tuple[str, int]]) -> pd.DataFrame:
    """rows: list of (question_id, n_choices)."""
    return pd.DataFrame(
        [
            {
                "question_id": qid,
                "subject": "general",
                "question_text": f"Question {qid}?",
                "correct_option": "A",
                "correct_answer_text": "A text",
                "n_choices": n_choices,
            }
            for qid, n_choices in rows
        ]
    )


class TestLoadPreflightEligibilityBeforeSampling:
    """Confirmed audit finding, corrected: sampling n raw rows and
    filtering to n_choices==k AFTERWARD offers no a priori guarantee of
    yielding n eligible rows if the raw pool contains any ineligible
    ones (ARC-Challenge validation's few 3-/5-option rows). Frozen
    protocol: filter to the eligible pool FIRST, then sample exactly n
    from THAT pool."""

    def test_discriminates_sample_before_filter_from_filter_before_sample(self):
        """This exact fixture/seed pair is chosen so the OLD (sample n raw
        rows, then filter) approach would have returned only 10/15
        eligible rows -- empirically verified by directly reproducing that
        old code path below -- while the fixed (filter first, then sample
        n) approach returns exactly 15."""
        rows = (
            [(f"q{i:03d}", 4) for i in range(15)]  # 15 eligible
            + [(f"x{i:03d}", 3) for i in range(5)]  # 5 ineligible
        )
        df = _make_questions_with_n_choices(rows)

        # Reproduce the OLD sample-then-filter behavior directly, to prove
        # this fixture/seed genuinely would have broken it (not just
        # asserting the NEW behavior in isolation).
        old_style_sample = df.sample(n=15, random_state=42)
        old_style_eligible_count = int((old_style_sample["n_choices"] == 4).sum())
        assert old_style_eligible_count < 15, (
            "fixture must exercise the old sample-before-filter bug; "
            f"got {old_style_eligible_count}/15 eligible in the naive sample"
        )

        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=15, n_choices=4))
        bench = _make_benchmark(split="test")
        fake_artifact = _make_prepared_dataset(df)

        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            result = load_preflight(method, bench, run_seed=42)

        assert len(result) == 15
        assert all(r["n_choices"] == 4 for r in result)
        assert all(r["question_id"].startswith("q") for r in result)  # never one of the 5 ineligible rows

    def test_deterministic_selection_under_seed_42(self):
        rows = [(f"q{i:03d}", 4) for i in range(15)] + [(f"x{i:03d}", 3) for i in range(5)]
        df = _make_questions_with_n_choices(rows)
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=10, n_choices=4))
        bench = _make_benchmark(split="test")
        fake_artifact = _make_prepared_dataset(df)

        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            first = load_preflight(method, bench, run_seed=42)
        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            second = load_preflight(method, bench, run_seed=42)

        assert [r["question_id"] for r in first] == [r["question_id"] for r in second]

    def test_exact_mmlu_style_selection_77_of_77_eligible(self):
        """MMLU: validation pool = 1,531, all modal-k=4 eligible -- select
        exactly 77."""
        rows = [(f"q{i:04d}", 4) for i in range(1531)]
        df = _make_questions_with_n_choices(rows)
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=77, n_choices=4))
        bench = _make_benchmark(split="test")
        fake_artifact = _make_prepared_dataset(df)

        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            result = load_preflight(method, bench, run_seed=42)

        assert len(result) == 77
        assert len({r["question_id"] for r in result}) == 77  # no duplicates
        assert all(r["n_choices"] == 4 for r in result)

    def test_exact_arc_style_selection_15_of_295_eligible(self):
        """ARC: raw validation = 299, eligible four-option = 295 -- select
        exactly 15 from the eligible pool, never from the ineligible 4."""
        rows = (
            [(f"q{i:04d}", 4) for i in range(295)]
            + [(f"x{i:04d}", 3) for i in range(2)]
            + [(f"y{i:04d}", 5) for i in range(2)]
        )
        df = _make_questions_with_n_choices(rows)
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=15, n_choices=4))
        bench = _make_benchmark(split="test")
        fake_artifact = _make_prepared_dataset(df)

        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            result = load_preflight(method, bench, run_seed=42)

        assert len(result) == 15
        assert all(r["n_choices"] == 4 for r in result)
        assert all(r["question_id"].startswith("q") for r in result)

    def test_calibration_evaluation_disjointness_preserved_with_n_choices(self):
        """The pre-existing eval_question_ids exclusion must still apply
        BEFORE eligibility filtering -- an eligible row that's also in the
        evaluation set must never be selected for calibration."""
        rows = [(f"q{i:03d}", 4) for i in range(20)]
        df = _make_questions_with_n_choices(rows)
        eval_ids = {f"q{i:03d}" for i in range(10)}  # first 10 are "in eval"

        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=10, n_choices=4))
        bench = _make_benchmark(split="test")
        fake_artifact = _make_prepared_dataset(df)

        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            result = load_preflight(method, bench, run_seed=42, eval_question_ids=eval_ids)

        returned_ids = {r["question_id"] for r in result}
        assert returned_ids.isdisjoint(eval_ids)
        assert returned_ids == {f"q{i:03d}" for i in range(10, 20)}  # only the 10 non-eval-overlapping ones

    def test_insufficient_eligible_pool_fails_loudly(self):
        """Only 12 eligible rows exist but n=15 is requested -- must raise
        immediately, at load_preflight() itself, not silently return 12
        or silently include ineligible rows."""
        rows = [(f"q{i:03d}", 4) for i in range(12)] + [(f"x{i:03d}", 3) for i in range(8)]
        df = _make_questions_with_n_choices(rows)
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=15, n_choices=4))
        bench = _make_benchmark(split="test")
        fake_artifact = _make_prepared_dataset(df)

        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            with pytest.raises(ValueError, match="eligible"):
                load_preflight(method, bench, run_seed=42)

    def test_missing_n_choices_column_raises_clearly(self):
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=5, n_choices=4))
        bench = _make_benchmark(split="test")
        fake_artifact = _make_prepared_dataset(_make_questions(20))  # no n_choices column

        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            with pytest.raises(ValueError, match="n_choices"):
                load_preflight(method, bench, run_seed=42)

    def test_n_choices_none_preserves_prior_unfiltered_behavior(self):
        """Backward compatibility: when n_choices is not set (the default,
        and every non-PriDe preflight consumer), behavior is completely
        unchanged -- rows of any n_choices value may be sampled."""
        rows = [(f"q{i:03d}", 4) for i in range(10)] + [(f"x{i:03d}", 3) for i in range(10)]
        df = _make_questions_with_n_choices(rows)
        method = _make_method(PreflightConfig(source="benchmark", split="validation", n=15))  # n_choices unset
        bench = _make_benchmark(split="test")
        fake_artifact = _make_prepared_dataset(df)

        with patch("choicebench.preflight._load_benchmark_artifact", return_value=fake_artifact):
            result = load_preflight(method, bench, run_seed=42)

        assert len(result) == 15  # never raises even though only 10 are "eligible" -- filter is opt-in


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
