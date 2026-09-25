"""Section R: backward compatibility.

Frozen invariant: a representative pre-v2 config still parses through the
current schema; the new provider_seed default (None) does not alter the
experiment seed's own independent behavior; unpinned clients keep their
pre-pinning cache key; PriDe's strict full-calibration mode is opt-in and
does not break a generic historical caller that never asked for it.

R5 (conformance tests never touch real historical outputs) is a suite-
hygiene property verified by code review + tmp_path discipline throughout
this suite, not an automated grep-based check here (a static grep for
"runs/"/"cache/" would also match legitimate read-only references in
docstrings/comments and produce false positives) -- see README.md.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from choicebench.backends.dummy_backend import DummyBackend
from choicebench.cli.run_experiment import build_backend
from choicebench.clients.openrouter_client import OpenRouterClient
from choicebench.config.schema import ExperimentConfig, load_config
from choicebench.infra.cache import _cache_key, client_extra_identity
from choicebench.clients.types import ModelRequest
from choicebench.methods.base import ExperimentRunner
from choicebench.methods.library.pride import PriDeRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "prompts"

_PRE_V2_TOY_CONFIG = """\
experiment:
  name: toy_experiment

models:
  - backend: dummy
    model_name_or_path: dummy_1
    device: cpu
    generation_kwargs:
      max_new_tokens: 16
      temperature: 0.0
      do_sample: false
  - backend: dummy
    model_name_or_path: dummy_2
    device: cpu
    generation_kwargs:
      max_new_tokens: 16
      temperature: 0.0
      do_sample: false

benchmarks:
  - name: toy
    split: test
    n_samples: 10
    subject_filter: null

methods:
  - name: direct_mcq

metrics:
  - accuracy

run:
  seed: 42
  resume: false
  dry_run: false
  checkpoint_every_n: 50
  prompt_version: "v1"
  concurrency_limit: 10
"""


# --- R1 -------------------------------------------------------------------

def test_r1_pre_v2_config_still_parses_and_builds_a_backend(tmp_path):
    """config/toy_experiment.yaml as it existed at pre-v2 anchor commit
    549c9bcc2f9f4db3c74b3bc2394cfea145fb2c7b (git show), copied verbatim --
    never written back into the tracked working tree."""
    config_path = tmp_path / "pre_v2_toy_experiment.yaml"
    config_path.write_text(_PRE_V2_TOY_CONFIG)

    config = load_config(str(config_path))
    assert isinstance(config, ExperimentConfig)
    assert len(config.models) == 2
    assert config.models[0].provider_seed is None  # field didn't exist pre-v2; default applies cleanly

    backend = build_backend(config.models[0], run_id="conformance-r1", run_seed=config.run.seed)
    assert isinstance(backend, DummyBackend)


# --- R2 -------------------------------------------------------------------

def test_r2_provider_seed_default_none_does_not_alter_experiment_seed_layer():
    """ExperimentRunner.seed (the experiment seed) and ModelConfig.provider_seed
    live on entirely separate objects/layers -- ExperimentRunner's constructor
    has no provider_seed parameter at all, so a model's provider_seed default
    (None) can never influence experiment-seed-driven behavior (tie-breaking,
    bootstrap, benchmark sampling)."""
    params = inspect.signature(ExperimentRunner.__init__).parameters
    assert "seed" in params
    assert "provider_seed" not in params


# --- R3 -------------------------------------------------------------------

def test_r3_legacy_unpinned_cache_key_matches_pre_pinning_formula(monkeypatch):
    monkeypatch.setattr("choicebench.config.providers.OPENROUTER_API_KEY", "sk-test")
    unpinned = OpenRouterClient(model_name="x")
    assert client_extra_identity(unpinned) is None

    request = ModelRequest(provider="openrouter", model_name="x", payload="hi", temperature=0.0, max_tokens=1024, seed=None)

    import hashlib
    import json
    pre_pinning_key_data = {
        "provider": request.provider, "model_name": request.model_name,
        "prompt": request.payload, "temperature": request.temperature,
        "max_tokens": request.max_tokens, "seed": request.seed,
    }
    pre_pinning_key = hashlib.sha256(json.dumps(pre_pinning_key_data, sort_keys=True).encode()).hexdigest()

    assert _cache_key(request, extra_identity=client_extra_identity(unpinned)) == pre_pinning_key


# --- R4 -------------------------------------------------------------------

def test_r4_pride_strict_calibration_is_opt_in_generic_caller_unaffected(tmp_path):
    """A generic historical caller who never set require_full_calibration
    (default False) and has a calibration pool smaller than calibration_n
    must NOT be broken by the new strict-calibration feature -- it degrades
    gracefully (smaller/uniform prior), exactly as it always has."""
    pool = [
        {
            "question_id": f"legacy-{i}", "subject": "s", "question_text": "q?",
            "choices_json": [{"text": t, "source_index": j} for j, t in enumerate(["a", "b", "c", "d"])],
            "correct_option": "A",
        }
        for i in range(3)  # far fewer than calibration_n=15
    ]
    runner = PriDeRunner(
        backend=DummyBackend(), method_name="pride", split_name="test",
        prompt_version="v1", prompts_dir=PROMPTS_DIR, run_id="conformance-r4",
        seed=42, benchmark_name="diag_bench",
        calibration_n=15, calibration_seed=42,
        calibration_questions=pool, modal_k=4,
        calibration_runs_dir=tmp_path,
        # require_full_calibration omitted -- defaults to False.
    )
    runner._ensure_calibration()  # must not raise
    assert len(runner._calibration_state.estimation_question_ids) == 3

    strict_runner = PriDeRunner(
        backend=DummyBackend(), method_name="pride", split_name="test",
        prompt_version="v1", prompts_dir=PROMPTS_DIR, run_id="conformance-r4-strict",
        seed=42, benchmark_name="diag_bench",
        calibration_n=15, calibration_seed=42, require_full_calibration=True,
        calibration_questions=pool, modal_k=4,
        calibration_runs_dir=tmp_path,
    )
    import pytest
    with pytest.raises(RuntimeError):
        strict_runner._ensure_calibration()
