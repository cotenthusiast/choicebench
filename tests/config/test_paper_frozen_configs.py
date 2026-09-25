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

import shutil
from pathlib import Path

import pytest

from choicebench.config.schema import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config" / "paper"


@pytest.fixture(scope="module")
def real_prepared_mmlu_arc_data():
    """Copies this repo's REAL, already-verified prepared MMLU/ARC-Challenge
    artifacts (data/processed/{mmlu,arc_challenge}/) into this test session's
    isolated CHOICEBENCH_HOME (see conftest.py's workspace isolation), so
    tests can exercise load_preflight() against genuine prepared data instead
    of only asserting on YAML values. Scoped to this module only -- the rest
    of the suite stays hermetic/independent of repo working-tree state."""
    from choicebench.config.paths import PROCESSED_DIR

    for name in ("mmlu", "arc_challenge"):
        src = REPO_ROOT / "data" / "processed" / name
        if not src.exists():
            pytest.skip(f"real prepared artifact data/processed/{name}/ not present in this checkout")
        dst = PROCESSED_DIR / name
        if not dst.exists():
            shutil.copytree(src, dst)

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
    # direct_mcq/reasoning_mcq are NOT method entries in the cyclic/
    # reasoning-batch configs -- the frozen protocol requires baseline to
    # BE cyclic_permutation's/reasoning_cyclic's own rotation-0
    # observation, derived (zero new calls) via
    # scripts/paper/derive_baseline_from_cyclic.py rather than run as a
    # second independent method here. See that script's/module's own
    # header for the full rationale.
    "mmlu_core_methods_cyclic_batch.yaml": {"cyclic_permutation"},
    "arc_core_methods_cyclic_batch.yaml": {"cyclic_permutation"},
    "mmlu_independent_hypothesis.yaml": {"independent_hypothesis"},
    "arc_independent_hypothesis.yaml": {"independent_hypothesis"},
    "mmlu_reasoning_batch.yaml": {"reasoning_cyclic"},
    "arc_reasoning_batch.yaml": {"reasoning_cyclic"},
}

_SYNC_ONLY_CONFIGS = {
    # direct_mcq is NOT a method entry here -- see _BATCH_CONFIGS_AND_METHODS's note.
    "mmlu_core_methods.yaml": {"two_stage", "text_extraction"},
    "arc_core_methods.yaml": {"two_stage", "text_extraction"},
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


@pytest.mark.parametrize("filename", [
    "mmlu_core_methods.yaml", "arc_core_methods.yaml",
    "mmlu_core_methods_cyclic_batch.yaml", "arc_core_methods_cyclic_batch.yaml",
])
def test_direct_mcq_is_never_an_independent_method_entry(filename):
    """Regression: baseline must BE cyclic_permutation's own rotation-0
    observation (derived, zero new calls), never a second independent
    live method -- confirmed audit finding, fixed 2026-09-25. A
    reintroduced 'direct_mcq' entry anywhere in these configs would mean
    two target-model observations for the same canonical-order prompt."""
    config = load_config(str(CONFIG_DIR / filename))
    method_names = {m.name for m in config.methods}
    assert "direct_mcq" not in method_names, filename


@pytest.mark.parametrize("filename", [
    "mmlu_reasoning.yaml", "arc_reasoning.yaml",
    "mmlu_reasoning_batch.yaml", "arc_reasoning_batch.yaml",
])
def test_reasoning_mcq_is_never_an_independent_method_entry(filename):
    """Same bug, same fix, reasoning variant: reasoning_mcq must BE
    reasoning_cyclic's own rotation-0 observation, derived via
    scripts/paper/derive_baseline_from_cyclic.py --method-name
    reasoning_mcq, never run as its own live method alongside
    reasoning_cyclic in the same batch config."""
    config = load_config(str(CONFIG_DIR / filename))
    method_names = {m.name for m in config.methods}
    assert "reasoning_mcq" not in method_names, filename


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


@pytest.mark.parametrize("filename,expected_n", _PRIDE_CONFIGS.items())
def test_pride_configs_actually_thread_calibration_n_into_the_runner(
        filename, expected_n, real_prepared_mmlu_arc_data,
):
    """Regression for a confirmed bug: preflight.n controls how many rows
    are LOADED as the calibration pool, but PriDeRunner.__init__'s own
    calibration_n keyword (default 50) separately controls how many of
    those it actually calibrates on. The frozen configs previously set
    only preflight.n (77/15), leaving calibration_n at its silent default
    of 50 -- MMLU would have calibrated on 50 questions, not the frozen
    77. This instantiates the REAL production config through the REAL
    orchestrator path (load_config -> load_preflight -> instantiate_runner),
    not just asserting on the YAML value, so a regression here would be
    caught even if someone "fixes" only the YAML comment.
    """
    from choicebench.backends.dummy_backend import DummyBackend
    from choicebench.cli.run_experiment import instantiate_runner
    from choicebench.preflight import load_preflight

    config = load_config(str(CONFIG_DIR / filename))
    model_config = config.models[0]
    method_config = config.methods[0]
    benchmark_cfg = config.benchmarks[0]

    preflight_questions = load_preflight(method_config, benchmark_cfg, run_seed=config.run.seed)
    assert preflight_questions, f"{filename}: preflight produced no rows against real prepared data"

    runner = instantiate_runner(
        config, model_config, method_config, DummyBackend(), "test_run", benchmark_cfg,
        preflight_questions=preflight_questions,
    )

    assert runner._calibration_n == expected_n, (
        f"{filename}: runner._calibration_n={runner._calibration_n} != frozen {expected_n} "
        "-- calibration_n was not actually threaded from config into the runner."
    )
    assert runner._require_full_calibration is True, filename


@pytest.mark.parametrize("filename,expected_n", _PRIDE_CONFIGS.items())
def test_pride_configs_actually_calibrate_on_the_full_frozen_n(
        filename, expected_n, real_prepared_mmlu_arc_data,
):
    """Beyond instantiation: actually runs calibration (real prepared
    validation data, DummyBackend for score_options so no network/GPU is
    used) and asserts the resulting calibration state was fit on exactly
    expected_n questions -- the real end-to-end behavioral check the
    audit asked for, not just a constructor-argument check."""
    from choicebench.backends.dummy_backend import DummyBackend
    from choicebench.cli.run_experiment import instantiate_runner
    from choicebench.preflight import load_preflight

    config = load_config(str(CONFIG_DIR / filename))
    model_config = config.models[0]
    method_config = config.methods[0]
    benchmark_cfg = config.benchmarks[0]

    preflight_questions = load_preflight(method_config, benchmark_cfg, run_seed=config.run.seed)
    runner = instantiate_runner(
        config, model_config, method_config, DummyBackend(), "test_run", benchmark_cfg,
        preflight_questions=preflight_questions,
    )

    runner._ensure_calibration()  # must not raise -- confirms >= expected_n eligible today

    assert len(runner._calibration_state.estimation_question_ids) == expected_n, filename


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
