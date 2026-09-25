# tests/config/test_paper_frozen_configs.py
#
# Paper-specific (eacl-2026-revision): regression protection for the
# frozen production config values themselves -- until now, none of the 12
# config/paper/*.yaml files had ANY test coverage at all. A future
# accidental edit (e.g. someone "cleaning up" a model ID, or copy-pasting
# a models: block between a sync and a batch config) could silently
# change a frozen scientific parameter with nothing catching it. This
# also directly guards against regressing the DeepInfra Llama model-ID
# bug found and fixed 2026-09-25 (a plain, deprecated, bfloat16 ID would
# have made the batch path silently serve a different deployment than
# the synchronous OpenRouter->DeepInfra path).

from pathlib import Path

import pytest

from choicebench.config.schema import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config" / "paper"

_ALL_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))

# The frozen fp8 model IDs -- see the batch configs' own header comments
# for the verification method (DeepInfra's live catalog + OpenRouter's
# endpoint listing, 2026-09-25).
_LLAMA_SYNC_MODEL_ID = "meta-llama/llama-3.1-8b-instruct"  # via openrouter + upstream_provider pin
_LLAMA_BATCH_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"  # via direct deepinfra client
_LLAMA_DEPRECATED_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # bfloat16, deprecated -- must never appear
_LLAMA_LOCAL_MODEL_ID = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
_QWEN_API_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-Turbo"
_QWEN_LOCAL_MODEL_ID = "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic"
_GPT_MODEL_ID = "gpt-4.1-mini-2025-04-14"
_CLAUDE_MODEL_ID = "claude-haiku-4-5-20251001"

_FULL_ROSTER_CONFIGS = [
    "mmlu_core_methods.yaml", "arc_core_methods.yaml",
    "mmlu_core_methods_cyclic_batch.yaml", "arc_core_methods_cyclic_batch.yaml",
    "mmlu_independent_hypothesis.yaml", "arc_independent_hypothesis.yaml",
    "mmlu_reasoning.yaml", "arc_reasoning.yaml",
    "mmlu_reasoning_batch.yaml", "arc_reasoning_batch.yaml",
]

_BATCH_CONFIGS_AND_METHODS = {
    "mmlu_core_methods_cyclic_batch.yaml": {"cyclic_permutation"},
    "arc_core_methods_cyclic_batch.yaml": {"cyclic_permutation"},
    "mmlu_independent_hypothesis.yaml": {"independent_hypothesis"},
    "arc_independent_hypothesis.yaml": {"independent_hypothesis"},
    "mmlu_reasoning_batch.yaml": {"reasoning_mcq", "reasoning_cyclic"},
    "arc_reasoning_batch.yaml": {"reasoning_mcq", "reasoning_cyclic"},
}

_SYNC_ONLY_CONFIGS = {
    "mmlu_core_methods.yaml": {"direct_mcq", "two_stage", "text_extraction"},
    "arc_core_methods.yaml": {"direct_mcq", "two_stage", "text_extraction"},
    "mmlu_reasoning.yaml": {"reasoning_two_stage"},
    "arc_reasoning.yaml": {"reasoning_two_stage"},
}

_PRIDE_CONFIGS = {"mmlu_pride.yaml": 77, "arc_pride.yaml": 15}


@pytest.mark.parametrize("path", _ALL_CONFIGS, ids=lambda p: p.name)
def test_every_paper_config_loads(path):
    load_config(str(path))


@pytest.mark.parametrize("path", _ALL_CONFIGS, ids=lambda p: p.name)
def test_every_paper_config_uses_the_frozen_generation_settings(path):
    """1024-token ceiling, temperature 0, uniform across every method --
    no config may silently drift from these."""
    config = load_config(str(path))
    for model in config.models:
        assert model.generation_kwargs.max_new_tokens == 1024, f"{path.name}: {model.model_name_or_path}"
        assert model.generation_kwargs.temperature == 0.0, f"{path.name}: {model.model_name_or_path}"


@pytest.mark.parametrize("filename", _FULL_ROSTER_CONFIGS)
def test_full_roster_configs_have_the_exact_frozen_six_models(filename):
    config = load_config(str(CONFIG_DIR / filename))
    ids_by_provider = {
        (m.provider, m.backend): m.model_name_or_path for m in config.models
    }
    assert len(config.models) == 6, filename

    gpt = next(m for m in config.models if m.provider == "openai")
    assert gpt.model_name_or_path == _GPT_MODEL_ID, filename

    claude = next(m for m in config.models if m.provider == "anthropic")
    assert claude.model_name_or_path == _CLAUDE_MODEL_ID, filename

    together = next(m for m in config.models if m.provider == "together")
    assert together.model_name_or_path == _QWEN_API_MODEL_ID, filename

    local_models = {m.model_name_or_path for m in config.models if m.backend == "huggingface"}
    assert local_models == {_LLAMA_LOCAL_MODEL_ID, _QWEN_LOCAL_MODEL_ID}, filename


@pytest.mark.parametrize("filename", [
    "mmlu_core_methods.yaml", "arc_core_methods.yaml",
    "mmlu_reasoning.yaml", "arc_reasoning.yaml",
])
def test_synchronous_configs_route_llama_through_openrouter_deepinfra_pin(filename):
    config = load_config(str(CONFIG_DIR / filename))
    llama = next(m for m in config.models if m.provider == "openrouter")
    assert llama.model_name_or_path == _LLAMA_SYNC_MODEL_ID, filename
    assert llama.upstream_provider == "deepinfra", filename
    assert llama.allow_fallbacks is False, filename


@pytest.mark.parametrize("filename", [
    "mmlu_core_methods_cyclic_batch.yaml", "arc_core_methods_cyclic_batch.yaml",
    "mmlu_independent_hypothesis.yaml", "arc_independent_hypothesis.yaml",
    "mmlu_reasoning_batch.yaml", "arc_reasoning_batch.yaml",
])
def test_batch_configs_route_llama_through_direct_deepinfra_fp8_turbo(filename):
    """Regression guard for the bug found/fixed 2026-09-25: the plain
    (deprecated, bfloat16) Llama ID must never reappear in a batch config."""
    config = load_config(str(CONFIG_DIR / filename))
    llama = next(m for m in config.models if m.provider == "deepinfra")
    assert llama.model_name_or_path == _LLAMA_BATCH_MODEL_ID, filename
    assert llama.model_name_or_path != _LLAMA_DEPRECATED_MODEL_ID, filename


def test_deprecated_llama_model_id_appears_in_no_paper_config():
    """Belt-and-suspenders: scan every config's raw model list, not just
    the ones this file already parametrizes over -- a new config added
    later without updating this test file's lists still gets caught."""
    for path in _ALL_CONFIGS:
        config = load_config(str(path))
        offenders = [
            m.model_name_or_path for m in config.models
            if m.model_name_or_path == _LLAMA_DEPRECATED_MODEL_ID
        ]
        assert not offenders, f"{path.name} uses the deprecated Llama model ID"


@pytest.mark.parametrize("filename,expected_methods", _BATCH_CONFIGS_AND_METHODS.items())
def test_batch_safe_methods_run_under_execution_mode_batch(filename, expected_methods):
    config = load_config(str(CONFIG_DIR / filename))
    batch_methods = {m.name for m in config.methods if m.execution_mode == "batch"}
    assert batch_methods == expected_methods, filename


@pytest.mark.parametrize("filename,expected_methods", _SYNC_ONLY_CONFIGS.items())
def test_dependent_methods_never_run_under_execution_mode_batch(filename, expected_methods):
    config = load_config(str(CONFIG_DIR / filename))
    method_names = {m.name for m in config.methods}
    assert method_names == expected_methods, filename
    for method in config.methods:
        assert method.execution_mode == "sync", f"{filename}: {method.name}"


@pytest.mark.parametrize("filename,expected_n", _PRIDE_CONFIGS.items())
def test_pride_configs_have_the_frozen_calibration_n(filename, expected_n):
    config = load_config(str(CONFIG_DIR / filename))
    assert len(config.methods) == 1
    pride = config.methods[0]
    assert pride.name == "pride"
    assert pride.preflight is not None
    assert pride.preflight.n == expected_n, filename
    assert pride.preflight.source == "benchmark", filename
    assert pride.preflight.split == "validation", filename


@pytest.mark.parametrize("filename", _PRIDE_CONFIGS)
def test_pride_configs_are_local_only(filename):
    """PriDe never runs against an API model in this paper -- both
    models in these configs must be huggingface-backed."""
    config = load_config(str(CONFIG_DIR / filename))
    assert len(config.models) == 2
    assert all(m.backend == "huggingface" for m in config.models), filename
    local_models = {m.model_name_or_path for m in config.models}
    assert local_models == {_LLAMA_LOCAL_MODEL_ID, _QWEN_LOCAL_MODEL_ID}, filename


def test_mmlu_configs_use_the_frozen_evaluation_manifest():
    for filename in [
        "mmlu_core_methods.yaml", "mmlu_core_methods_cyclic_batch.yaml",
        "mmlu_independent_hypothesis.yaml", "mmlu_reasoning.yaml",
        "mmlu_reasoning_batch.yaml",
    ]:
        config = load_config(str(CONFIG_DIR / filename))
        assert len(config.benchmarks) == 1
        assert config.benchmarks[0].question_id_manifest == "data/manifests/mmlu_eval_v2.csv", filename


def test_arc_configs_use_the_frozen_evaluation_manifest():
    for filename in [
        "arc_core_methods.yaml", "arc_core_methods_cyclic_batch.yaml",
        "arc_independent_hypothesis.yaml", "arc_reasoning.yaml",
        "arc_reasoning_batch.yaml",
    ]:
        config = load_config(str(CONFIG_DIR / filename))
        assert len(config.benchmarks) == 1
        assert config.benchmarks[0].question_id_manifest == "data/manifests/arc_challenge_eval_v2.csv", filename
