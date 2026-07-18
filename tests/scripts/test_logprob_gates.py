# tests/scripts/test_logprob_gates.py
#
# Regression for FCD-2 / PF-2: config validation (schema.load_config) and the
# pre-run gate (run_experiment.validate_logprob_compatibility) must agree on
# logprob capability. They previously used two hardcoded sets that disagreed on
# `dummy`, so a pride+dummy+requires_logprobs config passed schema validation
# then crashed at the runtime gate. Both now consult model_supports_logprobs().

import importlib
from pathlib import Path

import pytest
import yaml

from choicebench.config.schema import (
    ConfigError,
    load_config,
    model_supports_logprobs,
    ModelConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_run_experiment():
    import choicebench.cli.run_experiment as module
    return importlib.reload(module)


def _write(tmp_path, data) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return str(p)


def _pride_dummy_config():
    return {
        "experiment": {"name": "pride_dummy"},
        "models": [{"backend": "dummy", "model_name_or_path": "d1", "device": "cpu"}],
        "benchmarks": [{"name": "toy"}],
        "methods": [{"name": "pride", "requires_logprobs": True}],
        "metrics": ["accuracy"],
    }


def test_dummy_is_logprob_capable():
    assert model_supports_logprobs(
        ModelConfig(backend="dummy", model_name_or_path="d1", device="cpu")
    ) is True


def test_no_api_provider_is_logprob_capable():
    # No API provider client (including vLLM) implements score_options() — all
    # are generate-only, so no api provider is accepted for config-driven
    # logprob methods (see _LOGPROB_API_PROVIDERS in config/schema.py).
    assert model_supports_logprobs(
        ModelConfig(backend="api", model_name_or_path="m", provider="vllm")
    ) is False
    assert model_supports_logprobs(
        ModelConfig(backend="api", model_name_or_path="m", provider="openai")
    ) is False


def test_pride_dummy_passes_both_validators(tmp_path):
    # Validator 1: schema cross-field check (requires_logprobs).
    cfg = load_config(_write(tmp_path, _pride_dummy_config()))
    # Validator 2: run_experiment pre-run gate (requires_score_options).
    run_exp = _load_run_experiment()
    run_exp.validate_logprob_compatibility(cfg)  # must not raise


def test_pride_openai_rejected_by_both_validators(tmp_path):
    data = _pride_dummy_config()
    data["models"] = [
        {"backend": "api", "model_name_or_path": "gpt-4.1-mini", "provider": "openai"}
    ]
    # Validator 1 rejects at load (requires_logprobs + incapable backend).
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, data))

    # Validator 2 also rejects: build a config without requires_logprobs so it
    # passes schema, then the runtime gate (keyed on requires_score_options)
    # must still reject pride on an openai backend.
    data["methods"] = [{"name": "pride"}]
    cfg = load_config(_write(tmp_path, data))
    run_exp = _load_run_experiment()
    with pytest.raises(run_exp.ConfigurationError):
        run_exp.validate_logprob_compatibility(cfg)


def test_pride_vllm_rejected_by_both_validators(tmp_path):
    # vLLM is generate-only (no score_options), so config-driven
    # pride/cyclic_logprob + api+vllm is rejected the same way as any other
    # logprob-incapable api provider.
    data = _pride_dummy_config()
    data["models"] = [
        {"backend": "api", "model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct", "provider": "vllm"}
    ]
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, data))

    data["methods"] = [{"name": "pride"}]
    cfg = load_config(_write(tmp_path, data))
    run_exp = _load_run_experiment()
    with pytest.raises(run_exp.ConfigurationError):
        run_exp.validate_logprob_compatibility(cfg)
