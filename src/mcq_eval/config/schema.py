# src/mcq_eval/config/schema.py
#
# Validated schema for the unified experiment config (config/experiment_template.yaml).
# Describes a single (model, benchmark, methods, metrics) experiment, loaded
# and validated by load_config() before scripts/run_experiment.py uses it.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

_VALID_BACKENDS = {"huggingface", "api", "dummy"}
# Backends known (in this codebase) to implement score_options() — i.e.
# supports_logprobs=True. "dummy" is included deliberately: DummyBackend
# returns fixed scores precisely so it can stand in for a logprob-capable
# backend in tests and onboarding configs (see config/toy_experiment.yaml).
_LOGPROB_CAPABLE_BACKENDS = {"huggingface", "dummy"}
_VALID_DEVICES = {"cuda", "cpu", "auto"}
_VALID_BENCHMARKS = {"mmlu", "arc_challenge", "custom"}
_VALID_METRICS = {"accuracy", "mad"}


class ConfigError(Exception):
    """Raised when an experiment config is missing required fields or
    contains values that are individually valid but mutually inconsistent
    (e.g. a logprob-requiring method paired with an API backend)."""


@dataclass
class GenerationKwargsConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False


@dataclass
class ModelConfig:
    backend: str
    model_name_or_path: str
    device: str = "cuda"
    generation_kwargs: GenerationKwargsConfig = field(default_factory=GenerationKwargsConfig)


@dataclass
class BenchmarkConfig:
    name: str
    split: str = "test"
    n_samples: int | None = None
    subject_filter: list[str] | None = None


@dataclass
class MethodConfig:
    name: str
    requires_logprobs: bool = False


@dataclass
class RunConfig:
    seed: int = 42
    resume: bool = True
    dry_run: bool = False


@dataclass
class ExperimentConfig:
    """Top-level validated experiment config.

    Construct via load_config(path), not directly — load_config() is what
    enforces the cross-field rules (logprob/backend compatibility, writable
    output_dir) that a bare dataclass constructor can't.
    """

    name: str
    output_dir: str
    model: ModelConfig
    benchmark: BenchmarkConfig
    methods: list[MethodConfig]
    metrics: list[str]
    run: RunConfig


def _require(d: Mapping, key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required field '{key}' in {where}.")
    return d[key]


def _build_generation_kwargs(raw: dict | None) -> GenerationKwargsConfig:
    raw = raw or {}
    return GenerationKwargsConfig(
        max_new_tokens=int(raw.get("max_new_tokens", 512)),
        temperature=float(raw.get("temperature", 0.0)),
        do_sample=bool(raw.get("do_sample", False)),
    )


def _build_model(raw: dict) -> ModelConfig:
    backend = _require(raw, "backend", "model")
    if backend not in _VALID_BACKENDS:
        raise ConfigError(
            f"model.backend must be one of {sorted(_VALID_BACKENDS)}; got {backend!r}."
        )
    device = raw.get("device", "cuda")
    if device not in _VALID_DEVICES:
        raise ConfigError(
            f"model.device must be one of {sorted(_VALID_DEVICES)}; got {device!r}."
        )
    return ModelConfig(
        backend=backend,
        model_name_or_path=_require(raw, "model_name_or_path", "model"),
        device=device,
        generation_kwargs=_build_generation_kwargs(raw.get("generation_kwargs")),
    )


def _build_benchmark(raw: dict) -> BenchmarkConfig:
    name = _require(raw, "name", "benchmark")
    if name not in _VALID_BENCHMARKS:
        raise ConfigError(
            f"benchmark.name must be one of {sorted(_VALID_BENCHMARKS)}; got {name!r}."
        )
    n_samples = raw.get("n_samples")
    if n_samples is not None and (not isinstance(n_samples, int) or n_samples <= 0):
        raise ConfigError(
            f"benchmark.n_samples must be a positive integer or null; got {n_samples!r}."
        )
    return BenchmarkConfig(
        name=name,
        split=raw.get("split", "test"),
        n_samples=n_samples,
        subject_filter=raw.get("subject_filter"),
    )


def _build_methods(raw: list | None) -> list[MethodConfig]:
    if not raw:
        raise ConfigError("methods must be a non-empty list of {name, ...} entries.")
    methods = []
    for i, entry in enumerate(raw):
        name = _require(entry, "name", f"methods[{i}]")
        methods.append(
            MethodConfig(name=name, requires_logprobs=bool(entry.get("requires_logprobs", False)))
        )
    return methods


def _build_metrics(raw: list | None) -> list[str]:
    if not raw:
        raise ConfigError("metrics must be a non-empty list of metric names.")
    for m in raw:
        if not isinstance(m, str):
            raise ConfigError(f"Each metrics entry must be a string; got {m!r}.")
    return list(raw)


def _build_run(raw: dict | None) -> RunConfig:
    raw = raw or {}
    return RunConfig(
        seed=int(raw.get("seed", 42)),
        resume=bool(raw.get("resume", True)),
        dry_run=bool(raw.get("dry_run", False)),
    )


def _validate_cross_field(config: ExperimentConfig) -> None:
    """Rules that span more than one section — can't be checked per-field."""
    logprob_methods = [m.name for m in config.methods if m.requires_logprobs]
    if logprob_methods and config.model.backend not in _LOGPROB_CAPABLE_BACKENDS:
        raise ConfigError(
            f"Method(s) {logprob_methods} require score_options() / logprobs, "
            f"but model.backend is {config.model.backend!r}. API backends are "
            f"closed/opaque by design (generate() only) — only "
            f"{sorted(_LOGPROB_CAPABLE_BACKENDS)} currently support "
            f"score_options(). Either drop these methods or switch backends."
        )

    output_dir = Path(config.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"output_dir {config.output_dir!r} could not be created: {exc}") from exc
    if not os.access(output_dir, os.W_OK):
        raise ConfigError(f"output_dir {config.output_dir!r} is not writable.")


def load_config(path: str) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML path.

    Raises:
        ConfigError: with a human-readable message if the file is missing
            required fields, has individually-invalid field values, or has
            fields that are valid alone but mutually inconsistent (e.g. a
            logprob-requiring method paired with an API backend, or a
            non-writable output_dir).
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path} as YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level.")

    experiment = raw.get("experiment") or {}
    config = ExperimentConfig(
        name=_require(experiment, "name", "experiment"),
        output_dir=_require(experiment, "output_dir", "experiment"),
        model=_build_model(_require(raw, "model", "top level")),
        benchmark=_build_benchmark(_require(raw, "benchmark", "top level")),
        methods=_build_methods(raw.get("methods")),
        metrics=_build_metrics(raw.get("metrics")),
        run=_build_run(raw.get("run")),
    )
    _validate_cross_field(config)
    return config
