"""End-to-end proof that the PriDe reproduction config's full pipeline —
permutation filter -> pride_repro prompts -> direct_logprob/cyclic_logprob ->
accuracy/recall_rstd/mad -- executes with the dummy backend, before any GPU
time is spent on the real huggyllama/llama-13b run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from choicebench.config.schema import load_config
from choicebench.metrics import BUILTIN_METRICS
from choicebench.methods.library.cyclic_logprob import CyclicLogprobRunner
from choicebench.methods.library.direct_logprob import DirectLogprobRunner
from choicebench.permutation_filter import filter_permutation_unsafe

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROMPTS_DIR = _REPO_ROOT / "prompts"


def _mmlu_style_rows() -> pd.DataFrame:
    """A handful of MMLU-shaped rows, one of which is permutation-unsafe."""
    from choicebench.benchmarks.base import make_normalized_row

    rows = [
        make_normalized_row("abstract_algebra", "1 + 1 = ?", ["1", "2", "3", "4"], 1),
        make_normalized_row("anatomy", "Which is a bone?", ["Femur", "Liver", "A and B", "Skin"], 0),
        make_normalized_row("astronomy", "Closest planet to the sun?", ["Venus", "Mercury", "Earth", "Mars"], 1),
    ]
    return pd.DataFrame(rows)


def test_config_loads_with_dummy_backend(tmp_path):
    raw = yaml.safe_load((_REPO_ROOT / "examples" / "pride_reproduction.yaml").read_text())
    raw["models"][0]["backend"] = "dummy"
    cfg_path = tmp_path / "dummy_pride_repro.yaml"
    cfg_path.write_text(yaml.dump(raw))

    cfg = load_config(str(cfg_path))

    assert [m.name for m in cfg.methods] == ["direct_logprob", "cyclic_logprob"]
    assert cfg.run.prompt_version == "pride_repro"
    assert cfg.metrics == ["accuracy", "recall_rstd", "mad"]


def test_full_pipeline_dummy_backend_smoke():
    from choicebench.backends.dummy_backend import DummyBackend

    df = _mmlu_style_rows()
    filtered, exclusions = filter_permutation_unsafe(df)
    assert len(exclusions) == 1
    assert len(filtered) == 2

    backend = DummyBackend(fixed_scores={"A": -0.1, "B": -1.5, "C": -2.0, "D": -2.5})

    direct_runner = DirectLogprobRunner(
        backend=backend, method_name="direct_logprob", split_name="test",
        prompt_version="pride_repro", prompts_dir=_PROMPTS_DIR, run_id="dry_run",
    )
    cyclic_runner = CyclicLogprobRunner(
        backend=backend, method_name="cyclic_logprob", split_name="test",
        prompt_version="pride_repro", prompts_dir=_PROMPTS_DIR, run_id="dry_run",
    )

    direct_rows = direct_runner.run_many(filtered)
    cyclic_rows = cyclic_runner.run_many(filtered)

    assert len(direct_rows) == len(filtered)
    assert len(cyclic_rows) == len(filtered)
    for row in direct_rows + cyclic_rows:
        assert "about " in row["prompt"]  # pride_repro subject sentence rendered
        assert row["prompt"].rstrip("\n").endswith("Answer:")
        assert "_" not in row["prompt"].split("about ", 1)[1].split(".", 1)[0]

    results_df = pd.DataFrame(direct_rows)
    for metric_name in ("accuracy", "recall_rstd", "mad"):
        metric = BUILTIN_METRICS[metric_name]()
        computed = metric.compute(results_df)
        assert isinstance(computed, dict)
        assert computed  # non-empty — every metric produced *something*
