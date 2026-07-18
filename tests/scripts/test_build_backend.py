# tests/scripts/test_build_backend.py
#
# Regression coverage for build_backend() in scripts/run_experiment.py.
# scripts/ is an entry-point directory, not an importable package, so the
# module is loaded from its file path.

import asyncio
import importlib
import pathlib
import sys
import types

import pandas as pd
import pytest

from choicebench.backends.api_backend import APIBackend
from choicebench.backends.dummy_backend import DummyBackend
from choicebench.config.schema import (
    BenchmarkConfig,
    ExperimentConfig,
    GenerationKwargsConfig,
    MethodConfig,
    ModelConfig,
    RunConfig,
)
from choicebench.methods import DirectMCQRunner

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_run_experiment():
    import choicebench.cli.run_experiment as module
    return importlib.reload(module)


def _model(backend: str, **kwargs) -> ModelConfig:
    return ModelConfig(
        backend=backend,
        model_name_or_path=kwargs.pop("model_name_or_path", "m"),
        provider=kwargs.pop("provider", None),
        device=kwargs.pop("device", "cpu"),
        generation_kwargs=kwargs.pop("generation_kwargs", GenerationKwargsConfig()),
        **kwargs,
    )


def _experiment(method: MethodConfig) -> ExperimentConfig:
    return ExperimentConfig(
        name="unit",
        models=[_model("dummy")],
        benchmarks=[BenchmarkConfig(name="toy")],
        methods=[method],
        metrics=["accuracy"],
        run=RunConfig(seed=123, prompt_version="v1"),
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
        def __init__(self, model_name, concurrency_limit=10, **kwargs):
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


def test_model_inherits_run_concurrency_limit_when_unset(tmp_path, monkeypatch):
    """PF-12: a model that does not set concurrency_limit inherits the
    run-level default passed to build_backend()."""
    run_exp = _load_run_experiment()
    monkeypatch.setattr(run_exp, "RUNS_DIR", tmp_path)

    class _FakeClient:
        def __init__(self, model_name, concurrency_limit=10, **kwargs):
            self.model_name = model_name
            self.provider = "fake"
            self.concurrency_limit = concurrency_limit

    monkeypatch.setitem(run_exp.CLIENT_REGISTRY, "fake", _FakeClient)

    backend = run_exp.build_backend(
        _model("api", provider="fake", model_name_or_path="fake-model"),
        "rid",
        run_seed=1,
        default_concurrency_limit=3,
    )
    assert backend._concurrency_limit == 3
    assert backend._raw_client.concurrency_limit == 3


def test_per_model_concurrency_limit_overrides_run_default(tmp_path, monkeypatch):
    """PF-12: an explicit per-model concurrency_limit wins over the run default."""
    run_exp = _load_run_experiment()
    monkeypatch.setattr(run_exp, "RUNS_DIR", tmp_path)

    class _FakeClient:
        def __init__(self, model_name, concurrency_limit=10, **kwargs):
            self.model_name = model_name
            self.provider = "fake"
            self.concurrency_limit = concurrency_limit

    monkeypatch.setitem(run_exp.CLIENT_REGISTRY, "fake", _FakeClient)

    backend = run_exp.build_backend(
        _model("api", provider="fake", model_name_or_path="fake-model", concurrency_limit=7),
        "rid",
        run_seed=1,
        default_concurrency_limit=3,
    )
    assert backend._concurrency_limit == 7
    assert backend._raw_client.concurrency_limit == 7


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
        def __init__(self, model_name_or_path, device, **generation_kwargs):
            self.model_name_or_path = model_name_or_path
            self.device = device
            self.generation_kwargs = generation_kwargs
            self.loaded = False

        def load(self):
            self.loaded = True

    monkeypatch.setattr(run_exp, "HuggingFaceBackend", _FakeHF)

    backend = run_exp.build_backend(
        _model("huggingface", device="cpu"), "rid", run_seed=1
    )
    assert backend.loaded is True


def test_huggingface_backend_receives_generation_kwargs(monkeypatch):
    run_exp = _load_run_experiment()

    class _FakeHF:
        def __init__(self, model_name_or_path, device, **generation_kwargs):
            self.model_name_or_path = model_name_or_path
            self.device = device
            self.generation_kwargs = generation_kwargs

        def load(self):
            pass

    monkeypatch.setattr(run_exp, "HuggingFaceBackend", _FakeHF)

    backend = run_exp.build_backend(
        _model(
            "huggingface",
            device="cpu",
            generation_kwargs=GenerationKwargsConfig(
                max_new_tokens=17,
                temperature=0.25,
                do_sample=True,
            ),
        ),
        "rid",
        run_seed=1,
    )

    assert backend.generation_kwargs == {
        "add_bos_token": True,
        "max_new_tokens": 17,
        "temperature": 0.25,
        "do_sample": True,
    }


def test_instantiate_runner_passes_params_to_registry_method(monkeypatch):
    run_exp = _load_run_experiment()

    class _ParamRunner:
        def __init__(self, custom_threshold, **kwargs):
            self.custom_threshold = custom_threshold
            self.framework_kwargs = kwargs

    monkeypatch.setitem(run_exp.METHOD_REGISTRY, "param_method", _ParamRunner)

    method = MethodConfig(name="param_method", params={"custom_threshold": 0.75})
    runner = run_exp.instantiate_runner(
        _experiment(method),
        _model("dummy"),
        method,
        DummyBackend(),
        "rid",
        BenchmarkConfig(name="toy", split="test"),
    )

    assert runner.custom_threshold == 0.75
    assert runner.framework_kwargs["method_name"] == "param_method"


def test_instantiate_runner_without_params_still_works():
    run_exp = _load_run_experiment()
    method = MethodConfig(name="direct_mcq")

    runner = run_exp.instantiate_runner(
        _experiment(method),
        _model("dummy"),
        method,
        DummyBackend(),
        "rid",
        BenchmarkConfig(name="toy", split="test"),
    )

    assert isinstance(runner, DirectMCQRunner)


def test_instantiate_runner_passes_params_to_external_method(monkeypatch):
    run_exp = _load_run_experiment()

    class _ExternalRunner:
        def __init__(self, calibration_n, **kwargs):
            self.calibration_n = calibration_n
            self.framework_kwargs = kwargs

    module = types.ModuleType("choicebench_test_methods")
    module.ExternalRunner = _ExternalRunner
    monkeypatch.setitem(sys.modules, "choicebench_test_methods", module)

    method = MethodConfig(
        name="choicebench_test_methods:ExternalRunner",
        params={"calibration_n": 100},
    )
    runner = run_exp.instantiate_runner(
        _experiment(method),
        _model("dummy"),
        method,
        DummyBackend(),
        "rid",
        BenchmarkConfig(name="toy", split="test"),
    )

    assert runner.calibration_n == 100
    assert runner.framework_kwargs["method_name"] == "choicebench_test_methods:ExternalRunner"


def test_instantiate_runner_unsupported_param_has_clear_error(monkeypatch):
    run_exp = _load_run_experiment()

    class _NoParamRunner:
        def __init__(self, **kwargs):
            if "unsupported" in kwargs:
                raise TypeError("unexpected keyword argument 'unsupported'")

    monkeypatch.setitem(run_exp.METHOD_REGISTRY, "no_param_method", _NoParamRunner)

    method = MethodConfig(name="no_param_method", params={"unsupported": True})
    with pytest.raises(TypeError, match="no_param_method.*configured params.*YAML should remove"):
        run_exp.instantiate_runner(
            _experiment(method),
            _model("dummy"),
            method,
            DummyBackend(),
            "rid",
            BenchmarkConfig(name="toy", split="test"),
        )


def test_yaml_dry_run_is_honored(monkeypatch):
    run_exp = _load_run_experiment()
    cfg = _experiment(MethodConfig(name="direct_mcq"))
    cfg.run.dry_run = True

    monkeypatch.setattr(run_exp, "ensure_dirs", lambda: None)
    monkeypatch.setattr(
        run_exp,
        "parse_args",
        lambda: types.SimpleNamespace(
            config="config.yaml",
            dry_run=False,
            run_id="rid",
            yes=True,
        ),
    )
    monkeypatch.setattr(run_exp, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        run_exp,
        "load_benchmark",
        lambda *args, **kwargs: pytest.fail("dry run should not load benchmarks"),
    )

    run_exp.main()


class _RecordingRunner:
    run_id = "rid"

    def __init__(self):
        self.seen_ids: list[str] = []

    def run_many(self, rows):
        records = rows.to_dict(orient="records")
        self.seen_ids.extend(row["question_id"] for row in records)
        return [{"question_id": row["question_id"], "fresh": True} for row in records]


class _CheckpointDouble:
    def __init__(self, state):
        self.state = state
        self.load_called = False
        self.delete_count = 0
        self.saved_states = []

    def load(self):
        self.load_called = True
        return self.state

    def save(self, completed_ids, results, started_at):
        self.saved_states.append((list(completed_ids), list(results), started_at))

    def delete(self):
        self.delete_count += 1


def test_run_method_resume_true_loads_checkpoint(tmp_path, monkeypatch):
    run_exp = _load_run_experiment()
    questions = pd.DataFrame(
        [
            {"question_id": "q1"},
            {"question_id": "q2"},
        ]
    )
    checkpoint = _CheckpointDouble(
        {
            "completed_ids": ["q1"],
            "results": [{"question_id": "q1", "old": True}],
            "started_at": "2026-01-01T00:00:00+00:00",
        }
    )
    runner = _RecordingRunner()
    written = {}
    monkeypatch.setattr(
        run_exp,
        "write_run_results",
        lambda **kwargs: written.setdefault("results", kwargs["results"]) or tmp_path / "out.csv",
    )

    asyncio.run(run_exp.run_method(
        method_name="direct_mcq",
        runner=runner,
        questions=questions,
        checkpoint_mgr=checkpoint,
        output_dir=tmp_path,
        checkpoint_every_n=10,
        resume=True,
    ))

    assert checkpoint.load_called is True
    assert runner.seen_ids == ["q2"]
    assert written["results"] == [
        {"question_id": "q1", "old": True},
        {"question_id": "q2", "fresh": True},
    ]


def test_run_method_resume_false_starts_fresh_without_loading(tmp_path, monkeypatch):
    run_exp = _load_run_experiment()
    questions = pd.DataFrame(
        [
            {"question_id": "q1"},
            {"question_id": "q2"},
        ]
    )
    checkpoint = _CheckpointDouble(
        {
            "completed_ids": ["q1"],
            "results": [{"question_id": "q1", "old": True}],
            "started_at": "2026-01-01T00:00:00+00:00",
        }
    )
    runner = _RecordingRunner()
    written = {}
    monkeypatch.setattr(
        run_exp,
        "write_run_results",
        lambda **kwargs: written.setdefault("results", kwargs["results"]) or tmp_path / "out.csv",
    )

    asyncio.run(run_exp.run_method(
        method_name="direct_mcq",
        runner=runner,
        questions=questions,
        checkpoint_mgr=checkpoint,
        output_dir=tmp_path,
        checkpoint_every_n=10,
        resume=False,
    ))

    assert checkpoint.load_called is False
    assert checkpoint.delete_count >= 1
    assert runner.seen_ids == ["q1", "q2"]
    assert written["results"] == [
        {"question_id": "q1", "fresh": True},
        {"question_id": "q2", "fresh": True},
    ]


def test_load_benchmark_applies_subject_filter(monkeypatch):
    run_exp = _load_run_experiment()
    artifact = types.SimpleNamespace(
        dataframe=pd.DataFrame([
            {"question_id": "q1", "subject": "math"},
            {"question_id": "q2", "subject": "history"},
        ]), artifact_id="ds", content_digest="digest",
    )
    monkeypatch.setattr(run_exp, "load_prepared_dataset", lambda *a, **k: artifact)

    questions = run_exp.load_benchmark(
        BenchmarkConfig(name="toy", subject_filter=["math"]),
        run_seed=123,
    )

    assert questions["question_id"].tolist() == ["q1"]


def test_load_benchmark_subject_filter_rejects_empty_result(monkeypatch):
    run_exp = _load_run_experiment()
    artifact = types.SimpleNamespace(
        dataframe=pd.DataFrame([{"question_id": "q1", "subject": "math"}]),
        artifact_id="ds", content_digest="digest",
    )
    monkeypatch.setattr(run_exp, "load_prepared_dataset", lambda *a, **k: artifact)

    with pytest.raises(ValueError, match="subject_filter.*removed all rows"):
        run_exp.load_benchmark(
            BenchmarkConfig(name="toy", subject_filter=["history"]),
            run_seed=123,
        )


# ---------------------------------------------------------------------------
# validate_logprob_compatibility tests
# ---------------------------------------------------------------------------

def _cfg(method_name: str, backend: str, provider: str | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        name="unit",
        models=[_model(backend, provider=provider)],
        benchmarks=[BenchmarkConfig(name="toy")],
        methods=[MethodConfig(name=method_name)],
        metrics=["accuracy"],
        run=RunConfig(seed=0, prompt_version="v1"),
    )


def test_validate_pride_openai_raises_configuration_error():
    """pride + openai API backend → ConfigurationError before any run starts."""
    run_exp = _load_run_experiment()
    with pytest.raises(run_exp.ConfigurationError, match="requires logprob access"):
        run_exp.validate_logprob_compatibility(_cfg("pride", "api", provider="openai"))


def test_validate_pride_gemini_raises_configuration_error():
    """pride + gemini API backend → ConfigurationError."""
    run_exp = _load_run_experiment()
    with pytest.raises(run_exp.ConfigurationError, match="provider 'gemini'"):
        run_exp.validate_logprob_compatibility(_cfg("pride", "api", provider="gemini"))


def test_validate_pride_dummy_passes():
    """pride + dummy backend → no error.

    PF-2: DummyBackend.supports_logprobs is True, so the unified capability
    check (model_supports_logprobs) classifies dummy as logprob-capable in both
    the schema validator and this pre-run gate. (Previously the two gates
    disagreed and this combination passed schema then crashed here — FCD-2.)
    """
    run_exp = _load_run_experiment()
    run_exp.validate_logprob_compatibility(_cfg("pride", "dummy"))


def test_validate_pride_vllm_raises_configuration_error():
    """pride + vllm API backend → ConfigurationError.

    vLLM's client is generate-only — it does not implement score_options —
    so it is not accepted for config-driven logprob methods (see
    _LOGPROB_API_PROVIDERS in config/schema.py).
    """
    run_exp = _load_run_experiment()
    with pytest.raises(run_exp.ConfigurationError, match="provider 'vllm'"):
        run_exp.validate_logprob_compatibility(_cfg("pride", "api", provider="vllm"))


def test_validate_pride_huggingface_passes():
    """pride + huggingface backend → no error."""
    run_exp = _load_run_experiment()
    run_exp.validate_logprob_compatibility(_cfg("pride", "huggingface"))


def test_validate_direct_mcq_openai_passes():
    """direct_mcq does not require logprobs → any backend is fine."""
    run_exp = _load_run_experiment()
    run_exp.validate_logprob_compatibility(_cfg("direct_mcq", "api", provider="openai"))


def test_validate_direct_mcq_dummy_passes():
    """direct_mcq + dummy backend → no error."""
    run_exp = _load_run_experiment()
    run_exp.validate_logprob_compatibility(_cfg("direct_mcq", "dummy"))


def test_validate_error_message_names_provider():
    """ConfigurationError message must include the exact provider string."""
    run_exp = _load_run_experiment()
    with pytest.raises(run_exp.ConfigurationError, match="provider 'anthropic'"):
        run_exp.validate_logprob_compatibility(
            _cfg("pride", "api", provider="anthropic")
        )


# ---------------------------------------------------------------------------
# Safety-net catch: APIBackend.generate() called directly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_method_catches_api_generate_direct_call(tmp_path, monkeypatch):
    """run_method() must convert the APIBackend.generate() RuntimeError into
    a ConfigurationError with a message about the async path."""
    run_exp = _load_run_experiment()

    class _DirectCallRunner:
        """Simulates a runner that incorrectly calls backend.generate() directly."""
        run_id = "rid"
        backend = None  # makes use_async=False so the sync path is taken

        def run_many(self, rows):
            raise RuntimeError(
                "APIBackend.generate() cannot be called inside a running event loop. "
                "Use await backend.generate_batch([prompt]) from an async runner method."
            )

    checkpoint = _CheckpointDouble(None)
    monkeypatch.setattr(
        run_exp, "write_run_results",
        lambda **kwargs: tmp_path / "out.csv",
    )
    questions = pd.DataFrame([{"question_id": "q1"}])

    with pytest.raises(run_exp.ConfigurationError, match="async path"):
        await run_exp.run_method(
            method_name="direct_mcq",
            runner=_DirectCallRunner(),
            questions=questions,
            checkpoint_mgr=checkpoint,
            output_dir=tmp_path,
            checkpoint_every_n=10,
            resume=False,
        )
