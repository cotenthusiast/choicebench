# tests/scripts/test_build_backend.py
#
# Regression coverage for build_backend() in scripts/run_experiment.py.
# scripts/ is an entry-point directory, not an importable package, so the
# module is loaded from its file path.

import importlib.util
import pathlib

import pytest

from choicebench.backends.api_backend import APIBackend
from choicebench.backends.dummy_backend import DummyBackend
from choicebench.config.schema import GenerationKwargsConfig, ModelConfig

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_run_experiment():
    spec = importlib.util.spec_from_file_location(
        "run_experiment_under_test", _REPO_ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(backend: str, **kwargs) -> ModelConfig:
    return ModelConfig(
        backend=backend,
        model_name_or_path=kwargs.pop("model_name_or_path", "m"),
        provider=kwargs.pop("provider", None),
        device=kwargs.pop("device", "cpu"),
        generation_kwargs=GenerationKwargsConfig(),
        **kwargs,
    )


def test_dummy_backend_built():
    run_exp = _load_run_experiment()
    backend = run_exp.build_backend(_model("dummy"), "rid", run_seed=99)
    assert isinstance(backend, DummyBackend)


def test_unknown_backend_raises():
    run_exp = _load_run_experiment()
    with pytest.raises(ValueError, match="Unsupported backend type"):
        run_exp.build_backend(_model("not_a_backend"), "rid", run_seed=1)


def test_api_backend_receives_run_seed(tmp_path, monkeypatch):
    """Regression: build_backend must thread the run seed into APIBackend.

    The original code read a nonexistent model_config.run.seed and crashed
    with AttributeError on every API run.
    """
    run_exp = _load_run_experiment()
    monkeypatch.setattr(run_exp, "RUNS_DIR", tmp_path)

    class _FakeClient:
        def __init__(self, model_name):
            self.model_name = model_name
            self.provider = "fake"

    monkeypatch.setitem(run_exp.CLIENT_REGISTRY, "fake", _FakeClient)

    backend = run_exp.build_backend(
        _model("api", provider="fake", model_name_or_path="fake-model"),
        "rid",
        run_seed=1234,
    )
    assert isinstance(backend, APIBackend)
    assert backend._seed == 1234


def test_api_backend_unknown_provider_raises():
    run_exp = _load_run_experiment()
    with pytest.raises(ValueError, match="Unknown provider"):
        run_exp.build_backend(
            _model("api", provider="nope"), "rid", run_seed=1
        )


def test_huggingface_backend_is_loaded_before_return(monkeypatch):
    """Regression: build_backend must call .load() so generate() works.

    The original code returned an unloaded HuggingFaceBackend, so the first
    generate()/score_options() raised "Call load() before ...".
    """
    run_exp = _load_run_experiment()

    class _FakeHF:
        def __init__(self, model_name_or_path, device):
            self.model_name_or_path = model_name_or_path
            self.device = device
            self.loaded = False

        def load(self):
            self.loaded = True

    monkeypatch.setattr(run_exp, "HuggingFaceBackend", _FakeHF)

    backend = run_exp.build_backend(
        _model("huggingface", device="cpu"), "rid", run_seed=1
    )
    assert backend.loaded is True
