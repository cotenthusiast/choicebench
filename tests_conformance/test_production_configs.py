"""Section A: production config -> runtime propagation.

Frozen invariant: every config/paper/*.yaml condition is temperature=0,
max_new_tokens=1024, provider_seed=None, and the experiment seed (42) never
leaks into a provider's own request-level `seed` (ModelRequest.seed) --
that seed's own class default happens to be 42 (config/providers.py's
SEED=42, threaded through clients.types.ModelRequest), so a call site that
forgets to pass seed=None explicitly would silently "leak" the experiment
seed into the provider request without any test ever exercising the real
construction path catching it.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest

from choicebench.config.schema import ConfigError, load_config, model_supports_logprobs
from choicebench.cli.run_experiment import build_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG_DIR = REPO_ROOT / "config" / "paper"
PAPER_CONFIGS = sorted(PAPER_CONFIG_DIR.glob("*.yaml"))

# The two export-manifest configs are not experiment conditions at all --
# their single model is backend=dummy with a "unused-placeholder" name,
# present only to satisfy load_config()'s non-empty-models/-methods
# validation for scripts/paper/export_benchmark_questions.py. Excluded from
# every per-model generation-settings/seed assertion below (A2-A8), but
# still covered by A1 (must parse).
_EXPORT_ONLY_CONFIGS = {"mmlu_stochasticity_export.yaml", "arc_stochasticity_export.yaml"}

_REAL_CONFIGS = [p for p in PAPER_CONFIGS if p.name not in _EXPORT_ONLY_CONFIGS]

_SIX_FROZEN_CONDITIONS = {
    ("api", "openai", "gpt-4.1-mini-2025-04-14"),
    ("api", "anthropic", "claude-haiku-4-5-20251001"),
    ("api", "together", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    ("huggingface", None, "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"),
    ("huggingface", None, "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic"),
}
# Llama-3.1-8B-Instruct targeting the intended DeepInfra FP8 deployment is
# ONE frozen condition realized through two different routings depending on
# whether the consuming method runs sync or batch (see
# mmlu_core_methods_cyclic_batch.yaml's header): OpenRouter pinned to
# upstream_provider=deepinfra for sync methods, or the direct `deepinfra`
# client for batch/IHS/reasoning methods. Either satisfies the condition.
_LLAMA_DEEPINFRA_VARIANTS = {
    ("api", "openrouter", "meta-llama/llama-3.1-8b-instruct"),
    ("api", "deepinfra", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
}


def _real_generating_model_entries():
    """Yield (path, model_config) for every non-dummy model in every real config."""
    for path in _REAL_CONFIGS:
        config = load_config(str(path))
        for model in config.models:
            if model.backend == "dummy":
                continue
            yield path, model


# --- A1 ---------------------------------------------------------------

def test_a1_paper_config_dir_is_nonempty():
    assert len(PAPER_CONFIGS) >= 1, "config/paper/*.yaml is empty -- nothing to conform-test"


@pytest.mark.parametrize("path", PAPER_CONFIGS, ids=lambda p: p.name)
def test_a1_paper_config_parses(path):
    config = load_config(str(path))
    assert config.models, f"{path} has no models"
    assert config.benchmarks, f"{path} has no benchmarks"


# --- A2 -----------------------------------------------------------------

@pytest.mark.parametrize(
    "path,model",
    list(_real_generating_model_entries()),
    ids=[f"{p.name}::{m.model_name_or_path}" for p, m in _real_generating_model_entries()],
)
def test_a2_frozen_temperature_and_max_tokens(path, model):
    assert model.generation_kwargs.temperature == 0.0, (
        f"{path.name}: {model.model_name_or_path} temperature "
        f"{model.generation_kwargs.temperature!r} != 0.0"
    )
    assert model.generation_kwargs.max_new_tokens == 1024, (
        f"{path.name}: {model.model_name_or_path} max_new_tokens "
        f"{model.generation_kwargs.max_new_tokens!r} != 1024 "
        "(library default is 512 -- config must set this explicitly)"
    )


# --- A3 -------------------------------------------------------------------

@pytest.mark.parametrize(
    "path,model",
    list(_real_generating_model_entries()),
    ids=[f"{p.name}::{m.model_name_or_path}" for p, m in _real_generating_model_entries()],
)
def test_a3_provider_seed_is_none(path, model):
    assert model.provider_seed is None, (
        f"{path.name}: {model.model_name_or_path} has provider_seed="
        f"{model.provider_seed!r}, expected None for every frozen paper condition"
    )


# --- A4 ---------------------------------------------------------------

def test_a4_real_backend_construction_never_leaks_experiment_seed_openrouter(tmp_path, monkeypatch):
    monkeypatch.setattr("choicebench.config.providers.OPENROUTER_API_KEY", "sk-test-key")
    # build_backend() computes cache_dir under the real RUNS_DIR by default
    # (cache_scope="per_run") -- monkeypatch the module-level RUNS_DIR name
    # cli/run_experiment.py itself binds, so APIBackend's ResponseCache
    # creates its directory under tmp_path instead of the tracked repo tree.
    monkeypatch.setattr("choicebench.cli.run_experiment.RUNS_DIR", tmp_path)
    config = load_config(str(PAPER_CONFIG_DIR / "mmlu_core_methods.yaml"))
    model = next(m for m in config.models if m.provider == "openrouter")
    backend = build_backend(model, run_id="conformance-a4", run_seed=42, model_identity="a4-openrouter")
    request = backend._make_request("dummy prompt")
    assert request.seed is None, (
        "ModelRequest.seed leaked the experiment seed (42) instead of staying "
        f"None: got {request.seed!r}. ModelRequest's own class default for "
        "seed is 42 (config/providers.py SEED=42) -- a call site that forgot "
        "to pass provider_seed explicitly would silently fall back to it."
    )
    assert request.provider == "openrouter"


def test_a4_real_backend_construction_never_leaks_experiment_seed_together(tmp_path, monkeypatch):
    monkeypatch.setattr("choicebench.config.providers.TOGETHER_API_KEY", "sk-test-key")
    monkeypatch.setattr("choicebench.cli.run_experiment.RUNS_DIR", tmp_path)
    config = load_config(str(PAPER_CONFIG_DIR / "mmlu_core_methods.yaml"))
    model = next(m for m in config.models if m.provider == "together")
    backend = build_backend(model, run_id="conformance-a4", run_seed=42, model_identity="a4-together")
    request = backend._make_request("dummy prompt")
    assert request.seed is None
    assert request.provider == "together"


# --- A5 -------------------------------------------------------------------

def test_a5_generic_provider_seed_still_propagates(tmp_path, monkeypatch):
    """A synthetic (non-paper) config that DOES set provider_seed must still
    have it reach ModelRequest.seed -- proving the None-by-default fix did
    not simply delete provider_seed support for callers who want it."""
    monkeypatch.setattr("choicebench.config.providers.OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setattr("choicebench.cli.run_experiment.RUNS_DIR", tmp_path)
    config_path = tmp_path / "synthetic_provider_seed.yaml"
    config_path.write_text(
        """
experiment:
  name: synthetic_provider_seed_test
models:
  - backend: api
    provider: openai
    model_name_or_path: gpt-4.1-mini-2025-04-14
    provider_seed: 7
    generation_kwargs:
      temperature: 0.0
      max_new_tokens: 1024
benchmarks:
  - name: toy
methods:
  - name: direct_mcq
metrics:
  - accuracy
run:
  seed: 42
"""
    )
    config = load_config(str(config_path))
    model = config.models[0]
    assert model.provider_seed == 7
    backend = build_backend(model, run_id="conformance-a5", run_seed=42, model_identity="a5-openai")
    request = backend._make_request("dummy prompt")
    assert request.seed == 7


# --- A6 ---------------------------------------------------------------

def test_a6_paper_configs_cover_exactly_the_six_frozen_conditions():
    seen: set[tuple[str, str | None, str]] = set()
    for path, model in _real_generating_model_entries():
        seen.add((model.backend, model.provider, model.model_name_or_path))

    llama_variants_present = seen & _LLAMA_DEEPINFRA_VARIANTS
    assert llama_variants_present, (
        "No config uses either frozen Llama-3.1-8B/DeepInfra routing "
        f"(expected one of {_LLAMA_DEEPINFRA_VARIANTS}); saw {sorted(seen)}"
    )
    non_llama_seen = seen - _LLAMA_DEEPINFRA_VARIANTS
    assert non_llama_seen == _SIX_FROZEN_CONDITIONS, (
        f"Non-Llama condition set mismatch.\nExpected: {sorted(_SIX_FROZEN_CONDITIONS)}\n"
        f"Actual:   {sorted(non_llama_seen)}"
    )
    # Every real config's own model list, individually, must resolve to
    # exactly six (backend, provider, model) entries (the nominal six
    # conditions), never fewer/more within one file (PriDe's own configs
    # only run the two local models -- exempted, since PriDe is
    # local-only by design per its own header comment).
    for path in _REAL_CONFIGS:
        config = load_config(str(path))
        if path.name in {"mmlu_pride.yaml", "arc_pride.yaml"}:
            assert len(config.models) == 2
            continue
        assert len(config.models) == 6, f"{path.name} has {len(config.models)} models, expected 6"


# --- A7 -------------------------------------------------------------------

@pytest.mark.parametrize("path", PAPER_CONFIGS, ids=lambda p: p.name)
def test_a7_no_gemini_in_paper_configs(path):
    config = load_config(str(path))
    for model in config.models:
        assert model.provider != "gemini", f"{path.name}: found provider=gemini"
        assert "gemini" not in model.model_name_or_path.lower(), (
            f"{path.name}: found gemini-named model {model.model_name_or_path!r}"
        )


# --- A8 -------------------------------------------------------------------

_EXPECTED_REDHAT_IDS = {
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
    "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic",
}


@pytest.mark.parametrize("path", _REAL_CONFIGS, ids=lambda p: p.name)
def test_a8_local_configs_use_huggingface_and_exact_redhat_ids(path):
    config = load_config(str(path))
    for model in config.models:
        if model.backend != "huggingface":
            continue
        assert model.model_name_or_path in _EXPECTED_REDHAT_IDS, (
            f"{path.name}: unexpected local checkpoint {model.model_name_or_path!r}, "
            f"expected one of {_EXPECTED_REDHAT_IDS}"
        )


# --- Review-Focus regression guard: model_supports_logprobs structural gate ---

def test_pride_requires_logprobs_rejected_for_api_backend(tmp_path, monkeypatch):
    """J10-adjacent / A-section structural guard: an api backend can never
    satisfy a requires_logprobs: true method -- config validation itself
    must refuse this combination before any backend/network call."""
    monkeypatch.setattr("choicebench.config.providers.OPENAI_API_KEY", "sk-test-key")
    config_path = tmp_path / "synthetic_pride_api.yaml"
    config_path.write_text(
        """
experiment:
  name: synthetic_pride_on_api_backend
models:
  - backend: api
    provider: openai
    model_name_or_path: gpt-4.1-mini-2025-04-14
    generation_kwargs:
      temperature: 0.0
      max_new_tokens: 1024
benchmarks:
  - name: toy
methods:
  - name: pride
    requires_logprobs: true
metrics:
  - accuracy
run:
  seed: 42
"""
    )
    with pytest.raises(ConfigError):
        load_config(str(config_path))


def test_model_supports_logprobs_structural_gate():
    from dataclasses import fields
    from choicebench.config.schema import GenerationKwargsConfig, ModelConfig

    api_model = ModelConfig(
        backend="api", provider="openai", model_name_or_path="gpt-4.1-mini-2025-04-14",
        generation_kwargs=GenerationKwargsConfig(),
    )
    dummy_model = ModelConfig(
        backend="dummy", model_name_or_path="unused",
        generation_kwargs=GenerationKwargsConfig(),
    )
    assert model_supports_logprobs(api_model) is False
    # dummy is the offline stand-in for a logprob-capable (huggingface) backend
    # -- exercising the real huggingface path would require loading a tokenizer.
    assert model_supports_logprobs(dummy_model) is True


# --- J11 structural cross-check (no alpha parameter anywhere in PriDe) -----

def test_pride_has_no_alpha_parameter():
    from choicebench.config.schema import PriDeConfig
    from choicebench.methods.library.pride import PriDeRunner

    field_names = {f.name for f in dataclasses.fields(PriDeConfig)}
    assert "alpha" not in field_names
    init_params = inspect.signature(PriDeRunner.__init__).parameters
    assert not any("alpha" in name for name in init_params), (
        f"PriDeRunner.__init__ has an alpha-shaped parameter: {list(init_params)}"
    )
