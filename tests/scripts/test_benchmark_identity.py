# tests/scripts/test_benchmark_identity.py
#
# Regression for FCD-3 / MF-C: two distinct `name: huggingface` datasets must
# keep separate identities all the way to the written artifacts — CSV filename,
# checkpoint key, and benchmark_name column — instead of colliding under the
# literal label "huggingface".

from pathlib import Path

import pandas as pd
import pytest
import yaml

from choicebench.config.schema import (
    BenchmarkConfig,
    ConfigError,
    benchmark_write_label,
    load_config,
)
from choicebench.infra.checkpoint import CheckpointManager
from choicebench.io.writers import write_run_results


def _row(qid: str, condition_id: str) -> dict:
    return {
        "question_id": qid,
        "correct_option": "A",
        "parsed_choice": "A",
        "experiment_id": "exp", "condition_id": condition_id, "dataset_artifact_id": "ds",
        "dataset_selection_id": "sel", "model_id": "model", "method_id": "method",
        "prompt_id": "prompt", "benchmark_split": "test",
    }


def test_two_huggingface_datasets_get_distinct_written_identity(tmp_path: Path):
    bench_a = BenchmarkConfig(name="huggingface", hf_path="allenai/ai2_arc", hf_subset="ARC-Easy")
    bench_b = BenchmarkConfig(name="huggingface", hf_path="cais/mmlu", hf_subset="all")

    label_a = benchmark_write_label(bench_a)
    label_b = benchmark_write_label(bench_b)
    assert label_a != label_b
    assert label_a == "ai2_arc"
    assert label_b == "mmlu"

    # Distinct CSV filenames (via condition_id) + benchmark_name column.
    out_a = write_run_results([_row("a1", "cond_a")], tmp_path, "cond_a", label_a)
    out_b = write_run_results([_row("b1", "cond_b")], tmp_path, "cond_b", label_b)
    assert out_a != out_b
    assert pd.read_csv(out_a)["benchmark_name"].iloc[0] == "ai2_arc"
    assert pd.read_csv(out_b)["benchmark_name"].iloc[0] == "mmlu"

    # Distinct checkpoint keys.
    ckpt_a = CheckpointManager(tmp_path, "run", "direct_mcq", "m", label_a)
    ckpt_b = CheckpointManager(tmp_path, "run", "direct_mcq", "m", label_b)
    assert ckpt_a._path != ckpt_b._path


def test_output_name_disambiguates_same_hf_stem(tmp_path: Path):
    """Two HF datasets whose hf_path stems collide are distinguished by output_name."""
    a = BenchmarkConfig(name="huggingface", hf_path="orgA/bench", output_name="bench_a")
    b = BenchmarkConfig(name="huggingface", hf_path="orgB/bench", output_name="bench_b")
    assert benchmark_write_label(a) == "bench_a"
    assert benchmark_write_label(b) == "bench_b"


def test_loadconfig_rejects_two_hf_with_same_resolved_label(tmp_path: Path):
    data = {
        "experiment": {"name": "dup"},
        "models": [{"backend": "dummy", "model_name_or_path": "d1", "device": "cpu"}],
        "benchmarks": [
            {"name": "huggingface", "hf_path": "cais/mmlu"},
            {"name": "huggingface", "hf_path": "cais/mmlu"},
        ],
        "methods": [{"name": "direct_mcq"}],
        "metrics": ["accuracy"],
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="Duplicate benchmark key"):
        load_config(str(p))


def test_loadconfig_allows_two_distinct_hf_datasets(tmp_path: Path):
    data = {
        "experiment": {"name": "ok"},
        "models": [{"backend": "dummy", "model_name_or_path": "d1", "device": "cpu"}],
        "benchmarks": [
            {"name": "huggingface", "hf_path": "allenai/ai2_arc", "hf_subset": "ARC-Easy"},
            {"name": "huggingface", "hf_path": "cais/mmlu", "hf_subset": "all"},
        ],
        "methods": [{"name": "direct_mcq"}],
        "metrics": ["accuracy"],
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    cfg = load_config(str(p))  # must not raise
    assert len(cfg.benchmarks) == 2
