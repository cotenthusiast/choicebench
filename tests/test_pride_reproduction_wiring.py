"""End-to-end proof that the PriDe reproduction config's full pipeline —
permutation filter -> pride_repro prompts -> direct_logprob/cyclic_logprob ->
accuracy/recall_rstd/mad -- executes with the dummy backend, before any GPU
time is spent on the real huggyllama/llama-13b run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
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

    # Combine BOTH runners' rows so the metrics layer is exercised on
    # direct_logprob's and cyclic_logprob's output together, not just one.
    #
    # Both surviving questions (abstract_algebra, astronomy) have
    # correct_option == "B" (see _mmlu_style_rows/make_normalized_row). The
    # dummy backend's fixed_scores are keyed purely by letter and ignore
    # prompt content, so direct_logprob's argmax is deterministically "A" for
    # every question — a pure positional bias that never matches "B".
    # cyclic_logprob's debiasing (Eq. 1, averaged over cyclic permutations of
    # this same content-blind backend) also converges on "A" here; both were
    # confirmed by running the pipeline directly (see task report), not
    # guessed.
    results_df = pd.DataFrame(direct_rows + cyclic_rows)
    assert len(results_df) == 4  # 2 questions x 2 methods

    accuracy_result = BUILTIN_METRICS["accuracy"]().compute(results_df)
    assert accuracy_result["accuracy"] == 0.0
    assert accuracy_result["accuracy_conditional"] == 0.0

    recall_rstd_result = BUILTIN_METRICS["recall_rstd"]().compute(results_df)
    # Gold answers all sit at "B"; the backend's positional bias means every
    # row is parsed as "A", so recall at "B" is 0 out of 4 and there is no
    # dispersion to measure (a single non-NaN per-letter recall).
    assert recall_rstd_result["n_per_letter_B"] == 4.0
    assert recall_rstd_result["recall_B"] == 0.0
    assert recall_rstd_result["rstd"] == 0.0

    mad_result = BUILTIN_METRICS["mad"]().compute(results_df)
    # Predictions are 100% "A" (0% gold) and 0% "B" (100% gold) -> |100-0|
    # and |0-100| average to a maximal 100.0 marginal skew.
    assert mad_result["mad"] == 100.0
    assert mad_result["mad_std"] == 0.0
