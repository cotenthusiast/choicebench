# tests/config/test_schema.py
#
# Validation coverage for load_config() — the gate every run passes through.
# Several pre-release bugs (singular `model:`, logprob/backend mismatch) lived
# precisely in this layer, so it gets a dedicated suite.

import textwrap

import pytest
import yaml

from choicebench.config.schema import (
    BenchmarkConfig,
    ConfigError,
    ExperimentConfig,
    benchmark_normalized_stem,
    load_config,
)


def _write_config(tmp_path, data: dict) -> str:
    """Dump a config dict to a YAML file and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return str(p)


def _valid_config() -> dict:
    return {
        "experiment": {"name": "unit_test"},
        "models": [
            {"backend": "dummy", "model_name_or_path": "d1", "device": "cpu"},
        ],
        "benchmarks": [{"name": "toy", "split": "test", "n_samples": 5}],
        "methods": [{"name": "direct_mcq"}],
        "metrics": ["accuracy"],
        "run": {"seed": 7, "concurrency_limit": 3},
    }


# --- happy path ------------------------------------------------------------

def test_valid_config_loads(tmp_path):
    cfg = load_config(_write_config(tmp_path, _valid_config()))
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "unit_test"
    assert len(cfg.models) == 1
    assert cfg.run.seed == 7
    assert cfg.run.concurrency_limit == 3


def test_run_defaults_applied(tmp_path):
    data = _valid_config()
    del data["run"]
    cfg = load_config(_write_config(tmp_path, data))
    assert cfg.run.seed == 42
    assert cfg.run.prompt_version == "v1"
    assert cfg.run.concurrency_limit == 10
    assert cfg.run.resume is True


def test_generation_kwargs_parsed(tmp_path):
    data = _valid_config()
    data["models"][0]["generation_kwargs"] = {
        "max_new_tokens": 16,
        "temperature": 0.5,
        "do_sample": True,
    }
    cfg = load_config(_write_config(tmp_path, data))
    gk = cfg.models[0].generation_kwargs
    assert gk.max_new_tokens == 16
    assert gk.temperature == 0.5
    assert gk.do_sample is True


def test_method_params_default_to_empty_dict(tmp_path):
    cfg = load_config(_write_config(tmp_path, _valid_config()))
    assert cfg.methods[0].params == {}


def test_method_params_default_dicts_are_independent(tmp_path):
    data = _valid_config()
    data["methods"] = [{"name": "direct_mcq"}, {"name": "two_stage"}]
    cfg = load_config(_write_config(tmp_path, data))

    cfg.methods[0].params["custom"] = True

    assert cfg.methods[1].params == {}


def test_method_params_preserved(tmp_path):
    data = _valid_config()
    data["methods"] = [
        {
            "name": "pride",
            "requires_logprobs": True,
            "params": {"calibration_n": 100, "calibration_seed": 42},
        }
    ]
    cfg = load_config(_write_config(tmp_path, data))
    assert cfg.methods[0].params == {"calibration_n": 100, "calibration_seed": 42}


# --- missing / empty required sections -------------------------------------

def test_missing_experiment_name_raises(tmp_path):
    data = _valid_config()
    data["experiment"] = {}
    with pytest.raises(ConfigError, match="name"):
        load_config(_write_config(tmp_path, data))


def test_missing_models_raises(tmp_path):
    data = _valid_config()
    del data["models"]
    with pytest.raises(ConfigError, match="models must be a non-empty list"):
        load_config(_write_config(tmp_path, data))


def test_empty_benchmarks_raises(tmp_path):
    data = _valid_config()
    data["benchmarks"] = []
    with pytest.raises(ConfigError, match="benchmarks must be a non-empty list"):
        load_config(_write_config(tmp_path, data))


def test_empty_methods_raises(tmp_path):
    data = _valid_config()
    data["methods"] = []
    with pytest.raises(ConfigError, match="methods must be a non-empty list"):
        load_config(_write_config(tmp_path, data))


def test_empty_metrics_raises(tmp_path):
    data = _valid_config()
    data["metrics"] = []
    with pytest.raises(ConfigError, match="metrics must be a non-empty list"):
        load_config(_write_config(tmp_path, data))


# --- per-field validation --------------------------------------------------

def test_invalid_backend_raises(tmp_path):
    data = _valid_config()
    data["models"][0]["backend"] = "not_a_backend"
    with pytest.raises(ConfigError, match="model.backend must be one of"):
        load_config(_write_config(tmp_path, data))


def test_invalid_device_raises(tmp_path):
    data = _valid_config()
    data["models"][0]["device"] = "tpu"
    with pytest.raises(ConfigError, match="model.device must be one of"):
        load_config(_write_config(tmp_path, data))


def test_api_backend_requires_provider(tmp_path):
    data = _valid_config()
    data["models"][0] = {"backend": "api", "model_name_or_path": "gpt-4.1-mini"}
    with pytest.raises(ConfigError, match="model.provider is required"):
        load_config(_write_config(tmp_path, data))


def test_unknown_benchmark_raises(tmp_path):
    data = _valid_config()
    data["benchmarks"] = [{"name": "nonexistent"}]
    with pytest.raises(ConfigError, match="name must be one of"):
        load_config(_write_config(tmp_path, data))


def test_huggingface_benchmark_requires_hf_path(tmp_path):
    data = _valid_config()
    data["benchmarks"] = [{"name": "huggingface"}]
    with pytest.raises(ConfigError, match="hf_path is required"):
        load_config(_write_config(tmp_path, data))


def test_n_samples_must_be_positive(tmp_path):
    data = _valid_config()
    data["benchmarks"][0]["n_samples"] = 0
    with pytest.raises(ConfigError, match="n_samples must be a positive integer"):
        load_config(_write_config(tmp_path, data))


def test_unknown_metric_raises(tmp_path):
    data = _valid_config()
    data["metrics"] = ["not_a_metric"]
    with pytest.raises(ConfigError, match="Unknown metric"):
        load_config(_write_config(tmp_path, data))


def test_method_params_must_be_mapping(tmp_path):
    data = _valid_config()
    data["methods"] = [{"name": "direct_mcq", "params": ["not", "a", "mapping"]}]
    with pytest.raises(ConfigError, match=r"methods\[0\]\.params must be a YAML mapping"):
        load_config(_write_config(tmp_path, data))


def test_external_metric_path_allowed(tmp_path):
    data = _valid_config()
    data["metrics"] = ["my_pkg.metrics:MyMetric"]  # dotted path bypasses builtin check
    cfg = load_config(_write_config(tmp_path, data))
    assert cfg.metrics == ["my_pkg.metrics:MyMetric"]


# --- cross-field validation ------------------------------------------------

def test_logprob_method_with_api_backend_raises(tmp_path):
    data = _valid_config()
    data["models"][0] = {
        "backend": "api",
        "provider": "openai",
        "model_name_or_path": "gpt-4.1-mini",
    }
    data["methods"] = [{"name": "pride", "requires_logprobs": True}]
    with pytest.raises(ConfigError, match="requires score_options"):
        load_config(_write_config(tmp_path, data))


def test_logprob_method_with_dummy_backend_ok(tmp_path):
    data = _valid_config()  # dummy backend supports logprobs
    data["methods"] = [{"name": "pride", "requires_logprobs": True}]
    cfg = load_config(_write_config(tmp_path, data))
    assert cfg.methods[0].requires_logprobs is True


def test_duplicate_benchmark_key_raises(tmp_path):
    data = _valid_config()
    data["benchmarks"] = [{"name": "toy"}, {"name": "toy"}]
    with pytest.raises(ConfigError, match="Duplicate benchmark key"):
        load_config(_write_config(tmp_path, data))


def test_duplicate_benchmark_disambiguated_by_output_name(tmp_path):
    data = _valid_config()
    data["benchmarks"] = [
        {"name": "huggingface", "hf_path": "cais/mmlu", "output_name": "mmlu_a"},
        {"name": "huggingface", "hf_path": "cais/mmlu", "output_name": "mmlu_b"},
    ]
    cfg = load_config(_write_config(tmp_path, data))
    assert len(cfg.benchmarks) == 2


# --- helpers / misc --------------------------------------------------------

def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(str(tmp_path / "does_not_exist.yaml"))


def test_non_mapping_yaml_raises(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(str(p))


def test_malformed_yaml_raises(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent("key: [unclosed\n"))
    with pytest.raises(ConfigError, match="Could not parse"):
        load_config(str(p))


def test_benchmark_normalized_stem_prefers_output_name():
    cfg = BenchmarkConfig(name="huggingface", hf_path="cais/mmlu", output_name="custom")
    assert benchmark_normalized_stem(cfg) == "custom"


def test_benchmark_normalized_stem_derives_from_hf_path():
    cfg = BenchmarkConfig(name="huggingface", hf_path="allenai/ai2-arc")
    assert benchmark_normalized_stem(cfg) == "ai2_arc"


def test_benchmark_normalized_stem_requires_a_source():
    cfg = BenchmarkConfig(name="toy")
    with pytest.raises(ConfigError, match="requires either output_name or hf_path"):
        benchmark_normalized_stem(cfg)
