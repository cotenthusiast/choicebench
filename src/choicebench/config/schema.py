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

from choicebench.backends.dummy_backend import DummyBackend
from choicebench.backends.hf_backend import HuggingFaceBackend
from choicebench.benchmarks.registry import BENCHMARK_REGISTRY
from choicebench.metrics import BUILTIN_METRICS

DEFAULT_MAX_NEW_TOKENS = 512

_VALID_BACKENDS = {"huggingface", "api", "dummy"}
# No API provider is treated as logprob-capable for config-driven runs: vLLM's
# score_options only sees the top-20 generation logprobs (missing options are
# floored to -100.0), a degraded prior relative to HuggingFace's true
# full-vocabulary logit, so pride/cyclic_logprob + api+vllm is rejected here
# rather than silently producing debiased answers from a truncated
# distribution. This is the only place api-provider logprob capability is
# encoded. (The underlying capability — APIBackend.supports_logprobs /
# VLLMClient.score_options_async — is untouched, so a PriDeRunner/
# CyclicLogprobRunner built directly in Python, bypassing load_config(), can
# still use it; see the README "Logprob methods on vLLM vs HuggingFace"
# section.)
_LOGPROB_API_PROVIDERS: set[str] = set()
# Non-api backend classes, consulted for their declared supports_logprobs so we
# never maintain a second hardcoded "logprob-capable" name list (FCD-2 / PF-2).
_NONAPI_BACKEND_CLASSES = {
    "dummy": DummyBackend,
    "huggingface": HuggingFaceBackend,
}
_VALID_DEVICES = {"cuda", "cpu", "auto"}

BENCHMARK_TOY = "toy"
BENCHMARK_HUGGINGFACE = "huggingface"


def get_valid_benchmarks() -> set[str]:
    """Return the set of valid benchmark names (registry + special cases)."""
    return set(BENCHMARK_REGISTRY.keys()) | {BENCHMARK_TOY, BENCHMARK_HUGGINGFACE}


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
    # Max in-flight requests for this model (API backends only). None means
    # "inherit run.concurrency_limit", resolved when the backend is built.
    concurrency_limit: int | None = None
    base_url: str | None = None  # vLLM server address; ignored for all other providers


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
class PreflightConfig:
    source: str = "benchmark"   # "benchmark" or a file path
    split: str = "validation"   # split label; used for disjointness check
    n: int = 100                # number of preflight questions to load


@dataclass
class MethodConfig:
    name: str
    requires_logprobs: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    preflight: PreflightConfig | None = None


@dataclass
class RunConfig:
    seed: int = 42
    resume: bool = True
    dry_run: bool = False
    checkpoint_every_n: int = 50
    prompt_version: str = "v1"
    concurrency_limit: int = 10


@dataclass
class PriDeConfig:
    """PriDe-specific settings.

    modal_k_threshold: minimum proportion of evaluation questions that must
    share the benchmark's modal choice count for a PriDe run to proceed. PriDe
    has a global calibration step that assumes a single option count, so a
    benchmark with heterogeneous option counts is rejected before the run.
    Must lie in (0, 1].
    """

    modal_k_threshold: float = 0.95


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
    pride: PriDeConfig = field(default_factory=PriDeConfig)


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
                concurrency_limit=(
                    int(entry["concurrency_limit"])
                    if entry.get("concurrency_limit") is not None
                    else None
                ),
                base_url=entry.get("base_url"),
            )
        )
    return models


def _build_benchmark_entry(raw: dict, index: int) -> BenchmarkConfig:
    where = f"benchmarks[{index}]"
    name = _require(raw, "name", where)
    valid = get_valid_benchmarks()
    if name not in valid:
        raise ConfigError(
            f"{where}.name must be one of {sorted(valid)}; got {name!r}."
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


def benchmark_write_label(config: BenchmarkConfig) -> str:
    """Return the benchmark identity written to result CSVs and checkpoint keys.

    For the generic ``huggingface`` path, two distinct datasets both carry
    name ``"huggingface"``, so writing under the literal name would collide
    (overwrite + blend). The normalized stem (from output_name/hf_path) is the
    real per-dataset identity and is used instead. Every other benchmark type
    uses its ``name`` verbatim.
    """
    if config.name == BENCHMARK_HUGGINGFACE:
        return benchmark_normalized_stem(config)
    return config.name


def _build_methods(raw: list | None) -> list[MethodConfig]:
    if not raw:
        raise ConfigError("methods must be a non-empty list of {name, ...} entries.")
    methods = []
    for i, entry in enumerate(raw):
        where = f"methods[{i}]"
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{where} must be a YAML mapping; got {entry!r}.")
        name = _require(entry, "name", where)
        params = entry.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ConfigError(f"{where}.params must be a YAML mapping; got {params!r}.")
        preflight_raw = entry.get("preflight")
        preflight: PreflightConfig | None = None
        if preflight_raw is not None:
            if not isinstance(preflight_raw, dict):
                raise ConfigError(
                    f"{where}.preflight must be a YAML mapping; got {preflight_raw!r}."
                )
            preflight = PreflightConfig(
                source=str(preflight_raw.get("source", "benchmark")),
                split=str(preflight_raw.get("split", "validation")),
                n=int(preflight_raw.get("n", 100)),
            )
        methods.append(
            MethodConfig(
                name=name,
                requires_logprobs=bool(entry.get("requires_logprobs", False)),
                params=dict(params),
                preflight=preflight,
            )
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
        concurrency_limit=int(raw.get("concurrency_limit", 10)),
    )


def _build_pride(raw: dict | None) -> PriDeConfig:
    raw = raw or {}
    threshold = raw.get("modal_k_threshold", 0.95)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ConfigError(
            f"pride.modal_k_threshold must be a number in (0, 1]; got {threshold!r}."
        )
    if not (0.0 < threshold <= 1.0):
        raise ConfigError(
            f"pride.modal_k_threshold must be in (0, 1]; got {threshold}."
        )
    return PriDeConfig(modal_k_threshold=threshold)


def model_supports_logprobs(model: ModelConfig) -> bool:
    """Single source of truth: does this model's backend expose score_options()?

    Both config validation (this module) and run_experiment's pre-run gate call
    this one function, so the two can never drift apart (FCD-2 / PF-2). Capability
    is read from the backends' own declared ``supports_logprobs`` rather than a
    hardcoded name list:
      - api backends are capable iff the provider is in _LOGPROB_API_PROVIDERS
        (currently empty — no API provider is accepted for config-driven runs).
      - dummy / huggingface report their class capability directly; constructing
        them here is cheap (no weights are loaded until .load()).
    """
    if model.backend == "api":
        return model.provider in _LOGPROB_API_PROVIDERS
    cls = _NONAPI_BACKEND_CLASSES.get(model.backend)
    if cls is None:
        return False
    if cls is HuggingFaceBackend:
        probe = cls(model.model_name_or_path, model.device)
    else:
        probe = cls()
    return bool(probe.supports_logprobs)


def _validate_cross_field(config: ExperimentConfig) -> None:
    """Rules that span more than one section — can't be checked per-field."""
    for m in config.methods:
        for model in config.models:
            if m.requires_logprobs and not model_supports_logprobs(model):
                raise ConfigError(
                    f"Method {m.name!r} requires score_options() / logprobs, "
                    f"but model.backend={model.backend!r} / provider={model.provider!r} "
                    f"does not support it. Logprob-capable options: "
                    f"{sorted(_NONAPI_BACKEND_CLASSES)} backends. No api provider "
                    f"is accepted for config-driven runs (vLLM's score_options is a "
                    f"top-20-logprob approximation, not a full-vocabulary logit). "
                    f"Either drop this method or switch backends."
                )
    seen_benchmark_keys: set[str] = set()
    for bench in config.benchmarks:
        # Key on the *resolved written label* (what actually lands in the CSV
        # filename / checkpoint key / benchmark_name column), not just
        # output_name — otherwise two `name: huggingface` entries that resolve
        # to the same stem would slip past and silently overwrite each other.
        key = benchmark_write_label(bench)
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
        pride=_build_pride(raw.get("pride")),
    )
    _validate_cross_field(config)
    return config
