"""Section L: stochasticity (repeated observations).

Frozen invariant: observation 0 REUSES the main accuracy run's own saved
result (scripts/paper/run_stochasticity_repeats.py::_load_canonical_obs0_rows)
-- zero new target-model calls -- while observations 1..3 are fully fresh,
independent calls, each under a distinct model_identity (hence a distinct
cache/batch-state bucket). two_stage/reasoning_two_stage stay synchronous
(no multi-wave batching is built for their per-repetition Stage1->Stage2
dependency); direct_mcq/reasoning_mcq are batch-eligible. provider_seed is
None throughout (the same model_config object is reused for every
repetition, never varied).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PAPER_DIR = REPO_ROOT / "scripts" / "paper"
if str(SCRIPTS_PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PAPER_DIR))

import run_stochasticity_repeats as stoch  # noqa: E402  (path shim above)

from choicebench.analysis.agreement import compute_agreement_rate
from choicebench.config.schema import GenerationKwargsConfig, ModelConfig

from fakes.spy_backend import SpyBackend

PROMPTS_DIR = REPO_ROOT / "prompts"


def _model_config() -> ModelConfig:
    return ModelConfig(
        backend="api", provider="spy-provider", model_name_or_path="spy-model",
        generation_kwargs=GenerationKwargsConfig(temperature=0.0, max_new_tokens=1024),
        provider_seed=None,
    )


def _questions_csv(tmp_path: Path, question_id: str = "diag-n4-paris-001") -> Path:
    df = pd.DataFrame([{
        "question_id": question_id,
        "subject": "geography",
        "question_text": "What is the capital of France?",
        "choices_json": '[{"text": "Paris", "source_index": 0}, {"text": "London", "source_index": 1}, {"text": "Berlin", "source_index": 2}, {"text": "Madrid", "source_index": 3}]',
        "correct_option": "A",
        "benchmark_name": "diag_bench",
    }])
    path = tmp_path / "questions.csv"
    df.to_csv(path, index=False)
    return path


def _canonical_obs0_csv(
    tmp_path: Path, *, question_id: str = "diag-n4-paris-001", method_name: str = "direct_mcq",
    provider: str = "spy-provider", model_name: str = "spy-model",
    benchmark_name: str = "diag_bench", prompt_version: str = "v1",
    temperature: float = 0.0, max_tokens: int = 1024, filename: str = "obs0.csv",
    extra_rows: list[dict] | None = None,
) -> Path:
    rows = [{
        "question_id": question_id, "method_name": method_name,
        "provider": provider, "model_name": model_name,
        "benchmark_name": benchmark_name, "prompt_version": prompt_version,
        "temperature": temperature, "max_tokens": max_tokens,
        "parsed_choice": "A", "is_correct": True,
    }]
    if extra_rows:
        rows.extend(extra_rows)
    path = tmp_path / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _spy_factory(monkeypatch, built: dict[str, SpyBackend], *, per_rep_marker: bool = False):
    def fake_build_backend(model_config, run_id, run_seed, execution_mode, model_identity):
        if per_rep_marker:
            rep = model_identity.rsplit("rep", 1)[1]
            responder = lambda prompt, rep=rep: f"FREE_TEXT_REP_{rep}"  # noqa: E731
            spy = SpyBackend(responder=responder, provider=model_config.provider, model_name=model_config.model_name_or_path)
        else:
            spy = SpyBackend(default_text="The answer is A.", provider=model_config.provider, model_name=model_config.model_name_or_path)
        built[model_identity] = spy
        return spy

    monkeypatch.setattr(stoch, "build_backend", fake_build_backend)
    return fake_build_backend


# --- L1/L2/L3 ---------------------------------------------------------------

def test_l1_l2_l3_obs0_reused_zero_calls_and_three_fresh_canonical_calls(tmp_path, monkeypatch):
    built: dict[str, SpyBackend] = {}
    _spy_factory(monkeypatch, built)

    questions_csv = _questions_csv(tmp_path)
    obs0_csv = _canonical_obs0_csv(tmp_path)
    output_path = tmp_path / "out.csv"

    n_written = stoch.run(
        questions_csv, _model_config(), run_id="t", output_path=output_path,
        method_name="direct_mcq", prompt_version="v1", canonical_obs0_csv=obs0_csv,
        n_repetitions=4, run_seed=42, resume=True, execution_mode="batch",
    )
    assert n_written == 4

    # L1: obs0 caused zero SpyBackend constructions (pure CSV reuse).
    # L2: exactly 3 fresh SpyBackends were built (repetitions 1-3).
    assert len(built) == 3
    for identity, spy in built.items():
        assert spy.call_count == 1, f"{identity} made {spy.call_count} calls, expected 1"

    out_df = pd.read_csv(output_path)
    assert len(out_df) == 4
    assert set(out_df["repetition_index"]) == {0, 1, 2, 3}
    assert set(out_df["question_id"].astype(str)) == {"diag-n4-paris-001"}

    # L3: fresh repetitions' prompts use canonical (unrotated) option order.
    for identity, spy in built.items():
        prompt = spy.calls[0].prompt
        assert prompt.index("Paris") < prompt.index("London") < prompt.index("Berlin") < prompt.index("Madrid")


# --- L4/L5 ------------------------------------------------------------------

def test_l4_l5_distinct_model_identity_per_repetition(tmp_path, monkeypatch):
    built: dict[str, SpyBackend] = {}
    _spy_factory(monkeypatch, built)
    questions_csv = _questions_csv(tmp_path)
    obs0_csv = _canonical_obs0_csv(tmp_path)
    stoch.run(
        questions_csv, _model_config(), run_id="t", output_path=tmp_path / "out.csv",
        method_name="direct_mcq", prompt_version="v1", canonical_obs0_csv=obs0_csv,
        n_repetitions=4, run_seed=42, resume=True, execution_mode="batch",
    )
    identities = set(built.keys())
    assert identities == {
        "stochasticity_direct_mcq_rep1", "stochasticity_direct_mcq_rep2", "stochasticity_direct_mcq_rep3",
    }
    # L5 (no cross-repetition cache bleed) follows structurally from L4: each
    # identity maps to build_backend's own distinct cache_dir/batch_state_dir
    # (see cli/run_experiment.py::build_backend) -- the full cache layer
    # itself is exercised separately in test_cache_identity.py.


# --- L6 -----------------------------------------------------------------

def test_l6_resume_does_not_create_obs4_or_duplicate(tmp_path, monkeypatch):
    built: dict[str, SpyBackend] = {}
    _spy_factory(monkeypatch, built)
    questions_csv = _questions_csv(tmp_path)
    obs0_csv = _canonical_obs0_csv(tmp_path)
    output_path = tmp_path / "out.csv"

    first = stoch.run(
        questions_csv, _model_config(), run_id="t", output_path=output_path,
        method_name="direct_mcq", prompt_version="v1", canonical_obs0_csv=obs0_csv,
        n_repetitions=4, run_seed=42, resume=True, execution_mode="batch",
    )
    assert first == 4

    second = stoch.run(
        questions_csv, _model_config(), run_id="t", output_path=output_path,
        method_name="direct_mcq", prompt_version="v1", canonical_obs0_csv=obs0_csv,
        n_repetitions=4, run_seed=42, resume=True, execution_mode="batch",
    )
    assert second == 0
    out_df = pd.read_csv(output_path)
    assert len(out_df) == 4
    assert set(out_df["repetition_index"]) == {0, 1, 2, 3}


# --- L7-L11 -------------------------------------------------------------

@pytest.mark.parametrize(
    "mutation,column",
    [
        ({"provider": "wrong-provider"}, "provider"),
        ({"model_name": "wrong-model"}, "model_name"),
        ({"benchmark_name": "wrong-benchmark"}, "benchmark_name"),
        ({"prompt_version": "v99"}, "prompt_version"),
        ({"temperature": 1.0}, "temperature"),
        ({"max_tokens": 4096}, "max_tokens"),
    ],
    ids=["provider", "model_name", "benchmark_name", "prompt_version", "temperature", "max_tokens"],
)
def test_l7_l11_wrong_identity_obs0_hard_failure(tmp_path, monkeypatch, mutation, column):
    built: dict[str, SpyBackend] = {}
    _spy_factory(monkeypatch, built)
    questions_csv = _questions_csv(tmp_path)
    obs0_csv = _canonical_obs0_csv(tmp_path, **mutation, filename=f"obs0_{column}.csv")

    with pytest.raises(ValueError):
        stoch.run(
            questions_csv, _model_config(), run_id="t", output_path=tmp_path / f"out_{column}.csv",
            method_name="direct_mcq", prompt_version="v1", canonical_obs0_csv=obs0_csv,
            n_repetitions=4, run_seed=42, resume=True, execution_mode="batch",
        )


# --- L12 ------------------------------------------------------------------

def test_l12_duplicate_obs0_rows_hard_failure(tmp_path, monkeypatch):
    built: dict[str, SpyBackend] = {}
    _spy_factory(monkeypatch, built)
    questions_csv = _questions_csv(tmp_path)
    duplicate_row = {
        "question_id": "diag-n4-paris-001", "method_name": "direct_mcq",
        "provider": "spy-provider", "model_name": "spy-model",
        "benchmark_name": "diag_bench", "prompt_version": "v1",
        "temperature": 0.0, "max_tokens": 1024, "parsed_choice": "B", "is_correct": False,
    }
    obs0_csv = _canonical_obs0_csv(tmp_path, extra_rows=[duplicate_row], filename="obs0_dup.csv")

    with pytest.raises(ValueError, match="more than one row"):
        stoch.run(
            questions_csv, _model_config(), run_id="t", output_path=tmp_path / "out_dup.csv",
            method_name="direct_mcq", prompt_version="v1", canonical_obs0_csv=obs0_csv,
            n_repetitions=4, run_seed=42, resume=True, execution_mode="batch",
        )


# --- L13/L14 ----------------------------------------------------------------

def test_l13_l14_two_stage_stochasticity_no_cross_repetition_mixing(tmp_path, monkeypatch):
    built: dict[str, SpyBackend] = {}
    _spy_factory(monkeypatch, built, per_rep_marker=True)
    questions_csv = _questions_csv(tmp_path)
    obs0_csv = _canonical_obs0_csv(tmp_path, method_name="two_stage")

    stoch.run(
        questions_csv, _model_config(), run_id="t", output_path=tmp_path / "out_ts.csv",
        method_name="two_stage", prompt_version="v1", canonical_obs0_csv=obs0_csv,
        n_repetitions=4, run_seed=42, resume=True, execution_mode="sync",
    )

    assert len(built) == 3
    for identity, spy in built.items():
        assert spy.call_count == 2, f"{identity}: expected 1 Stage1 + 1 Stage2 = 2 calls, got {spy.call_count}"

    total_stage1 = sum(1 for spy in built.values())  # each spy's call[0] is its own Stage1
    total_stage2 = sum(1 for spy in built.values())  # each spy's call[1] is its own Stage2
    assert total_stage1 == 3
    assert total_stage2 == 3

    for identity, spy in built.items():
        rep = identity.rsplit("rep", 1)[1]
        own_marker = f"FREE_TEXT_REP_{rep}"
        stage2_prompt = spy.calls[1].prompt
        assert own_marker in stage2_prompt, f"{identity}'s Stage2 prompt missing its own Stage1 marker {own_marker!r}"
        for other_rep in {"1", "2", "3"} - {rep}:
            assert f"FREE_TEXT_REP_{other_rep}" not in stage2_prompt, (
                f"{identity}'s Stage2 prompt leaked repetition {other_rep}'s Stage1 marker -- cross-repetition mixing"
            )


# --- L15/L16 ----------------------------------------------------------------

def test_l15_l16_two_stage_and_reasoning_two_stage_must_be_sync():
    with pytest.raises(ValueError, match="execution_mode='sync'"):
        stoch._validate_execution_mode("two_stage", "batch")
    with pytest.raises(ValueError, match="execution_mode='sync'"):
        stoch._validate_execution_mode("reasoning_two_stage", "batch")
    stoch._validate_execution_mode("two_stage", "sync")  # must not raise


def test_l15_batch_rejected_before_any_backend_is_built(tmp_path, monkeypatch):
    built: dict[str, SpyBackend] = {}
    _spy_factory(monkeypatch, built)
    questions_csv = _questions_csv(tmp_path)
    obs0_csv = _canonical_obs0_csv(tmp_path, method_name="two_stage")
    with pytest.raises(ValueError):
        stoch.run(
            questions_csv, _model_config(), run_id="t", output_path=tmp_path / "out.csv",
            method_name="two_stage", prompt_version="v1", canonical_obs0_csv=obs0_csv,
            n_repetitions=4, run_seed=42, resume=True, execution_mode="batch",
        )
    assert built == {}, "a backend was constructed despite the sync-only violation"


# --- L17/L18 ----------------------------------------------------------------

def test_l17_l18_baseline_and_reasoning_mcq_are_batch_eligible():
    stoch._validate_execution_mode("direct_mcq", "batch")
    stoch._validate_execution_mode("reasoning_mcq", "batch")


# --- L19 ------------------------------------------------------------------

def test_l19_provider_seed_none_throughout(tmp_path, monkeypatch):
    seen_model_configs = []

    def fake_build_backend(model_config, run_id, run_seed, execution_mode, model_identity):
        seen_model_configs.append(model_config)
        return SpyBackend(default_text="The answer is A.")

    monkeypatch.setattr(stoch, "build_backend", fake_build_backend)
    model_config = _model_config()
    assert model_config.provider_seed is None

    questions_csv = _questions_csv(tmp_path)
    obs0_csv = _canonical_obs0_csv(tmp_path)
    stoch.run(
        questions_csv, model_config, run_id="t", output_path=tmp_path / "out.csv",
        method_name="direct_mcq", prompt_version="v1", canonical_obs0_csv=obs0_csv,
        n_repetitions=4, run_seed=42, resume=True, execution_mode="batch",
    )
    assert len(seen_model_configs) == 3
    for mc in seen_model_configs:
        assert mc is model_config
        assert mc.provider_seed is None


# --- L20 ------------------------------------------------------------------

def test_l20_agreement_metric_arithmetic():
    all_agree = pd.DataFrame({"question_id": ["q1"] * 4, "parsed_choice": ["A", "A", "A", "A"]})
    assert compute_agreement_rate(all_agree)["agreement_rate"] == pytest.approx(1.0)

    disagree = pd.DataFrame({"question_id": ["q1"] * 4, "parsed_choice": ["A", "A", "A", "B"]})
    assert compute_agreement_rate(disagree)["agreement_rate"] == pytest.approx(0.0)


# --- L21 ------------------------------------------------------------------

def test_l21_missing_or_duplicate_observation_not_silently_valid():
    """Frozen invariant: an agreement-rate consumer must require EXACTLY
    observations {0,1,2,3} -- a group missing one repetition_index, or with
    a duplicate, must never be silently treated as a valid four-observation
    group.

    FIXED (verified against commit 651137768dcad640de28f124cff3d50837fe7d7c):
    compute_agreement_rate now checks, whenever a repetition_index column is
    present, that each counted group's repetition_index values form a
    complete, duplicate-free 0..N-1 range, raising ValueError otherwise.
    Previously xfail."""
    missing_one = pd.DataFrame({
        "question_id": ["q1"] * 3, "repetition_index": [0, 1, 3],
        "parsed_choice": ["A", "A", "A"],
    })
    duplicate_three = pd.DataFrame({
        "question_id": ["q1"] * 5, "repetition_index": [0, 1, 2, 3, 3],
        "parsed_choice": ["A", "A", "A", "A", "B"],
    })
    for df in (missing_one, duplicate_three):
        with pytest.raises(ValueError):
            compute_agreement_rate(df, group_by="question_id")
