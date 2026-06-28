# src/choicebench/config/schema.py
#
# Validated schema for the unified experiment config (config/experiment_template.yaml).
# Describes a single (model, benchmark, methods, metrics) experiment, loaded
# and validated by load_config() before scripts/run_experiment.py uses it.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from choicebench.metrics import BUILTIN_METRICS

DEFAULT_MAX_NEW_TOKENS = 512

_VALID_BACKENDS = {"huggingface", "api", "dummy"}
# Backends known (in this codebase) to implement score_options() — i.e.
# supports_logprobs=True. "dummy" is included deliberately: DummyBackend
# returns fixed scores precisely so it can stand in for a logprob-capable
# backend in tests and onboarding configs (see config/toy_experiment.yaml).
_LOGPROB_CAPABLE_BACKENDS = {"huggingface", "dummy"}
_VALID_DEVICES = {"cuda", "cpu", "auto"}

BENCHMARK_MMLU = "mmlu"
BENCHMARK_ARC_CHALLENGE = "arc_challenge"
BENCHMARK_TOY = "toy"
BENCHMARK_HUGGINGFACE = "huggingface"
VALID_BENCHMARKS = {BENCHMARK_MMLU, BENCHMARK_ARC_CHALLENGE, BENCHMARK_TOY, BENCHMARK_HUGGINGFACE}


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
    provider: str | None = None  # Only required for API backends
    device: str = "cuda"
    generation_kwargs: GenerationKwargsConfig = field(default_factory=GenerationKwargsConfig)


@dataclass
class BenchmarkConfig:
    name: str
    split: str = "test"
    n_samples: int | None = None
    subject_filter: list[str] | None = None
    hf_path: str | None = None       # e.g. "cais/mmlu"
    hf_subset: str | None = None     # e.g. "all" or "ARC-Challenge"
    output_name: str | None = None   # override for normalized CSV filename stem


@dataclass
class MethodConfig:
    name: str
    requires_logprobs: bool = False


@dataclass
class RunConfig:
    seed: int = 42
    resume: bool = True
    dry_run: bool = False
    checkpoint_every_n: int = 50
    prompt_version: str = "v1"


@dataclass
class ExperimentConfig:
    """Top-level validated experiment config.

    Construct via load_config(path), not directly — load_config() is what
    enforces the cross-field rules (e.g. logprob/backend compatibility)
    that a bare dataclass constructor can't.
    """

    name: str
    models: list[ModelConfig]
    benchmarks: list[BenchmarkConfig]
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


def _build_models(raw: dict) -> list[ModelConfig]:
    models = []
    for i, entry in enumerate(raw.get("models", [])):
        backend = _require(entry, "backend", f"models[{i}]")
        if backend not in _VALID_BACKENDS:
            raise ConfigError(
                f"model.backend must be one of {sorted(_VALID_BACKENDS)}; got {backend!r}."
            )
        device = entry.get("device", "cuda")
        if device not in _VALID_DEVICES:
            raise ConfigError(
                f"model.device must be one of {sorted(_VALID_DEVICES)}; got {device!r}."
            )
        if backend == "api" and "provider" not in entry:
            raise ConfigError("model.provider is required for API backends.")
        models.append(
            ModelConfig(
                backend=backend,
                model_name_or_path=_require(entry, "model_name_or_path", f"models[{i}]"),
                provider=entry.get("provider"),
                device=device,
                generation_kwargs=_build_generation_kwargs(entry.get("generation_kwargs")),
            )
        )
    return models


def _build_benchmark_entry(raw: dict, index: int) -> BenchmarkConfig:
    where = f"benchmarks[{index}]"
    name = _require(raw, "name", where)
    if name not in VALID_BENCHMARKS:
        raise ConfigError(
            f"{where}.name must be one of {sorted(VALID_BENCHMARKS)}; got {name!r}."
        )
    n_samples = raw.get("n_samples")
    if n_samples is not None and (not isinstance(n_samples, int) or n_samples <= 0):
        raise ConfigError(
            f"{where}.n_samples must be a positive integer or null; got {n_samples!r}."
        )
    if name == BENCHMARK_HUGGINGFACE and not raw.get("hf_path"):
        raise ConfigError(f"{where}.hf_path is required when name is 'huggingface'.")
    return BenchmarkConfig(
        name=name,
        split=raw.get("split", "test"),
        n_samples=n_samples,
        subject_filter=raw.get("subject_filter"),
        hf_path=raw.get("hf_path"),
        hf_subset=raw.get("hf_subset"),
        output_name=raw.get("output_name"),
    )


def _build_benchmarks(raw: dict) -> list[BenchmarkConfig]:
    entries = raw.get("benchmarks", [])
    if not entries:
        raise ConfigError("benchmarks must be a non-empty list.")
    return [_build_benchmark_entry(entry, i) for i, entry in enumerate(entries)]


def benchmark_normalized_stem(config: BenchmarkConfig) -> str:
    """Return the normalized CSV filename stem for a benchmark config.

    Uses output_name if set; otherwise derives from hf_path by taking the
    portion after the last '/', lowercasing, and replacing hyphens with
    underscores (e.g. "cais/mmlu" → "mmlu", "org/my-bench" → "my_bench").
    """
    if config.output_name:
        return config.output_name
    if not config.hf_path:
        raise ConfigError(
            "benchmark_normalized_stem() requires either output_name or hf_path to be set."
        )
    stem = config.hf_path.rsplit("/", 1)[-1]
    return stem.lower().replace("-", "_")


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
        # Dotted importlib paths ("module.path:ClassName") are loaded at runtime.
        if ":" not in m and m not in BUILTIN_METRICS:
            raise ConfigError(
                f"Unknown metric {m!r}. Built-ins: {sorted(BUILTIN_METRICS)}. "
                f"For external metrics use 'module.path:ClassName'."
            )
    return list(raw)


def _build_run(raw: dict | None) -> RunConfig:
    raw = raw or {}
    return RunConfig(
        seed=int(raw.get("seed", 42)),
        resume=bool(raw.get("resume", True)),
        dry_run=bool(raw.get("dry_run", False)),
        checkpoint_every_n=int(raw.get("checkpoint_every_n", 50)),
        prompt_version=str(raw.get("prompt_version", "v1")),
    )


def _validate_cross_field(config: ExperimentConfig) -> None:
    """Rules that span more than one section — can't be checked per-field."""
    for m in config.methods:
        for model in config.models:
            if m.requires_logprobs and model.backend not in _LOGPROB_CAPABLE_BACKENDS:
                raise ConfigError(
                    f"Method {m.name!r} requires score_options() / logprobs, "
                    f"but model.backend is {model.backend!r}. API backends are "
                    f"closed/opaque by design (generate() only) — only "
                    f"{sorted(_LOGPROB_CAPABLE_BACKENDS)} currently support "
                    f"score_options(). Either drop this method or switch backends."
                )
    seen_benchmark_keys: set[str] = set()
    for bench in config.benchmarks:
        key = bench.output_name or bench.name
        if key in seen_benchmark_keys:
            raise ConfigError(
                f"Duplicate benchmark key {key!r} in benchmarks list — "
                f"two entries would write to the same output CSV. "
                f"Set output_name on one of them to disambiguate."
            )
        seen_benchmark_keys.add(key)


def load_config(path: str) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML path."""
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
    models = _build_models(raw)
    if not models:
        raise ConfigError("models must be a non-empty list.")

    config = ExperimentConfig(
        name=_require(experiment, "name", "experiment"),
        models=models,
        benchmarks=_build_benchmarks(raw),
        methods=_build_methods(raw.get("methods")),
        metrics=_build_metrics(raw.get("metrics")),
        run=_build_run(raw.get("run")),
    )
    _validate_cross_field(config)
    return config
