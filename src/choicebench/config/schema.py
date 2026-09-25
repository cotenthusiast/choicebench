# src/choicebench/config/schema.py
#
# Validated schema for the unified experiment config (config/experiment_template.yaml).
# Describes a single (model, benchmark, methods, metrics) experiment, loaded
# and validated by load_config() before scripts/run_experiment.py uses it.

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import yaml

from choicebench.backends.dummy_backend import DummyBackend
from choicebench.backends.hf_backend import HuggingFaceBackend
from choicebench.benchmarks.registry import BENCHMARK_REGISTRY
from choicebench.identity import is_credential_key
from choicebench.metrics import BUILTIN_METRICS

DEFAULT_MAX_NEW_TOKENS = 512


def _credential_keys_in(value: Any) -> set[str]:
    """Collect credential-named keys at any nesting depth of a params tree."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if is_credential_key(str(key)):
                found.add(str(key))
            found |= _credential_keys_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= _credential_keys_in(item)
    return found

_VALID_BACKENDS = {"huggingface", "api", "dummy"}
_VALID_CACHE_SCOPES = {"per_run", "shared"}
# Non-api backend classes, consulted for their declared supports_logprobs so we
# never maintain a second hardcoded "logprob-capable" name list (FCD-2 / PF-2).
_NONAPI_BACKEND_CLASSES = {
    "dummy": DummyBackend,
    "huggingface": HuggingFaceBackend,
}
_VALID_DEVICES = {"cuda", "cpu", "auto"}
_VALID_TORCH_DTYPES = {"float32", "float16", "bfloat16", "auto"}

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
    # Generic upstream-routing controls, consumed by any client that
    # supports them (currently: openrouter); ignored by providers that
    # don't accept them.
    upstream_provider: str | None = None
    allow_fallbacks: bool = True
    # HuggingFaceBackend only: whether the tokenizer prepends a BOS token
    # (transformers' add_special_tokens, applied to both generate() and
    # score_options()). Default True preserves prior behavior (most
    # tokenizers' own default). Zheng et al., ICLR 2024 (arXiv:2309.03882)
    # do not prepend BOS for open-source models — set False to match.
    # Ignored by api/dummy backends.
    add_bos_token: bool = True
    revision: str | None = None
    # HuggingFaceBackend only: overrides the load-time torch dtype. None
    # preserves the existing default (float32 on cpu, float16 elsewhere).
    # "auto" defers to the checkpoint's own declared torch_dtype -- the
    # right choice for a quantized checkpoint (e.g. a compressed-tensors
    # FP8 model) whose non-quantized layers have their own native dtype
    # (often bfloat16) that forcing float16 would silently override.
    torch_dtype: str | None = None
    # API backends only. This is deliberately SEPARATE from run.seed (the
    # experiment seed, used for benchmark sampling/manifests, deterministic
    # tie-breaking, and bootstrap analysis) -- run.seed must never leak into
    # the provider request's own `seed` parameter, which is a genuinely
    # different concept (a provider best-effort sampling-determinism knob
    # the frozen ChoiceBench protocol does not specify). None (default)
    # means no `seed` is sent to the provider at all; set this explicitly
    # only for an experiment that specifically wants one.
    provider_seed: int | None = None


@dataclass(frozen=True)
class SamplingConfig:
    """Deterministic stratified/per-group benchmark sampling.

    Mutually exclusive with BenchmarkConfig.n_samples's flat global sample.
    Currently supports one strategy: "per_group", which draws exactly
    n_per_group rows from every distinct value of group_field (e.g. MMLU's
    "subject" column) — raising rather than silently under-sampling if any
    group has fewer than n_per_group rows.
    """
    strategy: str
    group_field: str
    n_per_group: int


@dataclass
class BenchmarkConfig:
    name: str
    split: str = "test"
    n_samples: int | None = None
    sampling: SamplingConfig | None = None
    # Mutually exclusive with n_samples and sampling: filters the loaded
    # benchmark down to EXACTLY the question IDs in this file (one column
    # "question_id", or one ID per line for a plain-text file), rather than
    # re-deriving a selection at run time. Use this to guarantee a frozen
    # evaluation manifest stays authoritative even if the underlying
    # prepared dataset artifact ever changes.
    question_id_manifest: str | None = None
    subject_filter: list[str] | None = None
    hf_path: str | None = None       # e.g. "cais/mmlu"
    hf_subset: str | None = None     # e.g. "all" or "ARC-Challenge"
    output_name: str | None = None   # override for normalized CSV filename stem
    source_revision: str | None = None
    transforms: list[str] = field(default_factory=list)


@dataclass
class PreflightConfig:
    source: str = "benchmark"   # "benchmark" or a file path
    split: str = "validation"   # split label; used for disjointness check
    n: int = 100                # number of preflight questions to load
    # When set, restricts the pool to rows with exactly this many options
    # BEFORE sampling n from it (eligibility-before-sampling) -- rather
    # than sampling n raw rows and filtering afterward, which offers no
    # guarantee of yielding n eligible rows if the raw pool contains any
    # ineligible ones (e.g. ARC-Challenge validation's few 3-/5-option
    # rows). None (default) samples from the raw pool unfiltered,
    # preserving prior behavior for any consumer that doesn't need this.
    n_choices: int | None = None


_VALID_EXECUTION_MODES = {"sync", "batch"}


@dataclass
class MethodConfig:
    name: str
    requires_logprobs: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    preflight: PreflightConfig | None = None
    # "sync" (default): concurrent per-request dispatch via generate_batch()
    # (APIBackend). "batch": the SAME generate_batch() calls are routed
    # through a real provider Batch API job instead (BatchAPIBackend) --
    # only safe for methods whose calls are all independently constructible
    # up front (no cross-call dependency), never for a method that must
    # construct call N+1 from call N's own result.
    execution_mode: str = "sync"


@dataclass
class RunConfig:
    seed: int = 42
    resume: bool = True
    dry_run: bool = False
    checkpoint_every_n: int = 50
    prompt_version: str = "v1"
    prompt_dir: str | None = None
    concurrency_limit: int = 10
    # "per_run" (default): cache lives under runs/<run_id>/cache/, so a new
    # run ID starts cold even for requests already cached under another run.
    # "shared": cache lives under a single run-independent directory keyed
    # only by model identity + request content, so identical requests reuse
    # a hit across run IDs. Safe either way — the cache key already excludes
    # anything run-specific (see infra/cache.py's _cache_key()).
    cache_scope: str = "per_run"


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


def _reject_unknown(d: Mapping, allowed: set[str], where: str) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise ConfigError(f"Unknown field(s) in {where}: {unknown}. Check spelling or migrate the config.")


def _strict_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field_name} must be an integer; got {value!r}.")
    if minimum is not None and value < minimum:
        qualifier = "a positive integer" if minimum == 1 else f">= {minimum}"
        raise ConfigError(f"{field_name} must be {qualifier}; got {value!r}.")
    return value


def _strict_float(
    value: Any, field_name: str, *, minimum: float, maximum: float,
    minimum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"{field_name} must be a finite number; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field_name} must be a finite number; got {value!r}.")
    lower_bad = result <= minimum if minimum_exclusive else result < minimum
    if lower_bad or result > maximum:
        left = "(" if minimum_exclusive else "["
        raise ConfigError(f"{field_name} must be in {left}{minimum}, {maximum}]; got {value!r}.")
    return result


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be true or false; got {value!r}.")
    return value


def _nonempty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string; got {value!r}.")
    return value.strip()


def _build_generation_kwargs(raw: dict | None) -> GenerationKwargsConfig:
    raw = raw or {}
    _reject_unknown(raw, {"max_new_tokens", "temperature", "do_sample"}, "generation_kwargs")
    return GenerationKwargsConfig(
        max_new_tokens=_strict_int(raw.get("max_new_tokens", 512), "generation_kwargs.max_new_tokens", minimum=1),
        temperature=_strict_float(raw.get("temperature", 0.0), "generation_kwargs.temperature", minimum=0.0, maximum=2.0),
        do_sample=_strict_bool(raw.get("do_sample", False), "generation_kwargs.do_sample"),
    )


def _build_models(raw: dict) -> list[ModelConfig]:
    models = []
    for i, entry in enumerate(raw.get("models", [])):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"models[{i}] must be a YAML mapping; got {entry!r}.")
        _reject_unknown(entry, {
            "backend", "model_name_or_path", "provider", "device", "generation_kwargs",
            "concurrency_limit", "base_url", "add_bos_token", "revision",
            "upstream_provider", "allow_fallbacks", "torch_dtype", "provider_seed",
        }, f"models[{i}]")
        backend = _nonempty(_require(entry, "backend", f"models[{i}]"), f"models[{i}].backend")
        if backend not in _VALID_BACKENDS:
            raise ConfigError(
                f"model.backend must be one of {sorted(_VALID_BACKENDS)}; got {backend!r}."
            )
        device = _nonempty(entry.get("device", "cuda"), f"models[{i}].device")
        if device not in _VALID_DEVICES:
            raise ConfigError(
                f"model.device must be one of {sorted(_VALID_DEVICES)}; got {device!r}."
            )
        if backend == "api" and "provider" not in entry:
            raise ConfigError("model.provider is required for API backends.")
        torch_dtype = entry.get("torch_dtype")
        if torch_dtype is not None:
            torch_dtype = _nonempty(torch_dtype, f"models[{i}].torch_dtype")
            if torch_dtype not in _VALID_TORCH_DTYPES:
                raise ConfigError(
                    f"models[{i}].torch_dtype must be one of {sorted(_VALID_TORCH_DTYPES)}; got {torch_dtype!r}."
                )
        models.append(
            ModelConfig(
                backend=backend,
                model_name_or_path=_nonempty(_require(entry, "model_name_or_path", f"models[{i}]"), f"models[{i}].model_name_or_path"),
                provider=(_nonempty(entry["provider"], f"models[{i}].provider") if entry.get("provider") is not None else None),
                device=device,
                generation_kwargs=_build_generation_kwargs(entry.get("generation_kwargs")),
                concurrency_limit=(
                    _strict_int(entry["concurrency_limit"], f"models[{i}].concurrency_limit", minimum=1)
                    if entry.get("concurrency_limit") is not None
                    else None
                ),
                base_url=entry.get("base_url"),
                upstream_provider=(
                    _nonempty(entry["upstream_provider"], f"models[{i}].upstream_provider")
                    if entry.get("upstream_provider") is not None else None
                ),
                allow_fallbacks=_strict_bool(entry.get("allow_fallbacks", True), f"models[{i}].allow_fallbacks"),
                add_bos_token=_strict_bool(entry.get("add_bos_token", True), f"models[{i}].add_bos_token"),
                torch_dtype=torch_dtype,
                revision=(
                    _nonempty(entry["revision"], f"models[{i}].revision")
                    if entry.get("revision") is not None else None
                ),
                provider_seed=(
                    _strict_int(entry["provider_seed"], f"models[{i}].provider_seed", minimum=0)
                    if entry.get("provider_seed") is not None else None
                ),
            )
        )
    return models


_SAMPLING_STRATEGIES = {"per_group"}


def _build_sampling_entry(raw: Any, where: str) -> SamplingConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where} must be a YAML mapping; got {raw!r}.")
    _reject_unknown(raw, {"strategy", "group_field", "n_per_group"}, where)
    strategy = _nonempty(_require(raw, "strategy", where), f"{where}.strategy")
    if strategy not in _SAMPLING_STRATEGIES:
        raise ConfigError(
            f"{where}.strategy must be one of {sorted(_SAMPLING_STRATEGIES)}; got {strategy!r}."
        )
    group_field = _nonempty(_require(raw, "group_field", where), f"{where}.group_field")
    n_per_group = _strict_int(_require(raw, "n_per_group", where), f"{where}.n_per_group", minimum=1)
    return SamplingConfig(strategy=strategy, group_field=group_field, n_per_group=n_per_group)


def _build_benchmark_entry(raw: dict, index: int) -> BenchmarkConfig:
    where = f"benchmarks[{index}]"
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where} must be a YAML mapping; got {raw!r}.")
    _reject_unknown(raw, {
        "name", "split", "n_samples", "sampling", "question_id_manifest", "subject_filter",
        "hf_path", "hf_subset", "output_name", "source_revision", "transforms",
    }, where)
    name = _nonempty(_require(raw, "name", where), f"{where}.name")
    valid = get_valid_benchmarks()
    if name not in valid:
        raise ConfigError(
            f"{where}.name must be one of {sorted(valid)}; got {name!r}."
        )
    n_samples = raw.get("n_samples")
    if n_samples is not None:
        n_samples = _strict_int(n_samples, f"{where}.n_samples", minimum=1)
    sampling_raw = raw.get("sampling")
    if sampling_raw is not None and n_samples is not None:
        raise ConfigError(
            f"{where}: n_samples and sampling are mutually exclusive; set only one."
        )
    sampling = _build_sampling_entry(sampling_raw, f"{where}.sampling") if sampling_raw is not None else None
    question_id_manifest = raw.get("question_id_manifest")
    if question_id_manifest is not None:
        question_id_manifest = _nonempty(question_id_manifest, f"{where}.question_id_manifest")
        if n_samples is not None or sampling is not None:
            raise ConfigError(
                f"{where}: question_id_manifest, n_samples, and sampling are mutually "
                "exclusive; set only one."
            )
    if name == BENCHMARK_HUGGINGFACE and not raw.get("hf_path"):
        raise ConfigError(f"{where}.hf_path is required when name is 'huggingface'.")
    transforms = raw.get("transforms", [])
    if not isinstance(transforms, list):
        raise ConfigError(f"{where}.transforms must be a list of strings.")
    transforms = [_nonempty(value, f"{where}.transforms") for value in transforms]
    subject_filter = raw.get("subject_filter")
    if subject_filter is not None:
        if not isinstance(subject_filter, list):
            raise ConfigError(f"{where}.subject_filter must be a list of strings or null.")
        subject_filter = [_nonempty(value, f"{where}.subject_filter") for value in subject_filter]
    source_revision = raw.get("source_revision")
    if source_revision is not None:
        source_revision = _nonempty(source_revision, f"{where}.source_revision")
    return BenchmarkConfig(
        name=name,
        split=_nonempty(raw.get("split", "test"), f"{where}.split"),
        n_samples=n_samples,
        sampling=sampling,
        question_id_manifest=question_id_manifest,
        subject_filter=subject_filter,
        hf_path=raw.get("hf_path"),
        hf_subset=raw.get("hf_subset"),
        output_name=raw.get("output_name"),
        source_revision=source_revision,
        transforms=transforms,
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


def _reject_reserved_params(params: dict, where: str) -> None:
    """Refuse orchestrator-owned runtime field(s) inside a method's params.

    Orchestrator-collision prevention: these keys are injected by the runner
    at execution time and must not be set by the user's config.
    """
    reserved = sorted(set(params) & {
        "calibration_identity", "calibration_questions", "calibration_runs_dir",
        "condition_id", "gate_summary", "modal_k", "preflight_questions",
    })
    if reserved:
        raise ConfigError(
            f"{where}.params contains orchestrator-owned runtime field(s) {reserved}; remove them."
        )


def _reject_credential_params(params: dict, where: str) -> None:
    """Refuse credential-named field(s) inside a method's params.

    Security refusal, distinct from _reject_reserved_params: ChoiceBench
    reads credentials from environment variables, never scientific config.
    """
    credentials = sorted(_credential_keys_in(params))
    if credentials:
        raise ConfigError(
            f"{where}.params contains credential-named field(s) {credentials}. "
            "ChoiceBench reads credentials from environment variables and refuses "
            "them in scientific configuration; rename the parameter if it is "
            "ordinary method configuration."
        )


def _validate_method_specific_params(name: str, params: dict, where: str) -> None:
    """Method-name-specific business rules for individual method plugins' params."""
    if name == "pride":
        if "calibration_n" in params:
            _strict_int(params["calibration_n"], f"{where}.params.calibration_n", minimum=0)
        if "calibration_seed" in params:
            _strict_int(params["calibration_seed"], f"{where}.params.calibration_seed", minimum=0)
        if "require_full_calibration" in params:
            _strict_bool(params["require_full_calibration"], f"{where}.params.require_full_calibration")
    if name == "two_stage" and "fallback_on_parse_failure" in params:
        _strict_bool(params["fallback_on_parse_failure"], f"{where}.params.fallback_on_parse_failure")


def _build_methods(raw: list | None) -> list[MethodConfig]:
    if not raw:
        raise ConfigError("methods must be a non-empty list of {name, ...} entries.")
    methods = []
    for i, entry in enumerate(raw):
        where = f"methods[{i}]"
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{where} must be a YAML mapping; got {entry!r}.")
        name = _nonempty(_require(entry, "name", where), f"{where}.name")
        params = entry.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ConfigError(f"{where}.params must be a YAML mapping; got {params!r}.")
        _reject_reserved_params(params, where)
        _reject_credential_params(params, where)
        preflight_raw = entry.get("preflight")
        preflight: PreflightConfig | None = None
        _reject_unknown(entry, {"name", "requires_logprobs", "params", "preflight", "execution_mode"}, where)
        execution_mode = entry.get("execution_mode", "sync")
        if execution_mode not in _VALID_EXECUTION_MODES:
            raise ConfigError(
                f"{where}.execution_mode must be one of {sorted(_VALID_EXECUTION_MODES)}; "
                f"got {execution_mode!r}."
            )
        if preflight_raw is not None:
            if not isinstance(preflight_raw, dict):
                raise ConfigError(
                    f"{where}.preflight must be a YAML mapping; got {preflight_raw!r}."
                )
            _reject_unknown(preflight_raw, {"source", "split", "n", "n_choices"}, f"{where}.preflight")
            preflight = PreflightConfig(
                source=_nonempty(preflight_raw.get("source", "benchmark"), f"{where}.preflight.source"),
                split=_nonempty(preflight_raw.get("split", "validation"), f"{where}.preflight.split"),
                n=_strict_int(preflight_raw.get("n", 100), f"{where}.preflight.n", minimum=1),
                n_choices=(
                    _strict_int(preflight_raw["n_choices"], f"{where}.preflight.n_choices", minimum=2)
                    if preflight_raw.get("n_choices") is not None else None
                ),
            )
        _validate_method_specific_params(name, params, where)
        methods.append(
            MethodConfig(
                name=name,
                requires_logprobs=_strict_bool(entry.get("requires_logprobs", False), f"{where}.requires_logprobs"),
                params=dict(params),
                preflight=preflight,
                execution_mode=execution_mode,
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
    _reject_unknown(raw, {
        "seed", "resume", "dry_run", "checkpoint_every_n", "prompt_version",
        "prompt_dir", "concurrency_limit", "cache_scope",
    }, "run")
    cache_scope = _nonempty(raw.get("cache_scope", "per_run"), "run.cache_scope")
    if cache_scope not in _VALID_CACHE_SCOPES:
        raise ConfigError(
            f"run.cache_scope must be one of {sorted(_VALID_CACHE_SCOPES)}; got {cache_scope!r}."
        )
    return RunConfig(
        seed=_strict_int(raw.get("seed", 42), "run.seed", minimum=0),
        resume=_strict_bool(raw.get("resume", True), "run.resume"),
        dry_run=_strict_bool(raw.get("dry_run", False), "run.dry_run"),
        checkpoint_every_n=_strict_int(raw.get("checkpoint_every_n", 50), "run.checkpoint_every_n", minimum=1),
        prompt_version=_nonempty(raw.get("prompt_version", "v1"), "run.prompt_version"),
        prompt_dir=(
            _nonempty(raw["prompt_dir"], "run.prompt_dir")
            if raw.get("prompt_dir") is not None else None
        ),
        concurrency_limit=_strict_int(raw.get("concurrency_limit", 10), "run.concurrency_limit", minimum=1),
        cache_scope=cache_scope,
    )


def _build_pride(raw: dict | None) -> PriDeConfig:
    raw = raw or {}
    _reject_unknown(raw, {"modal_k_threshold"}, "pride")
    threshold = _strict_float(raw.get("modal_k_threshold", 0.95), "pride.modal_k_threshold", minimum=0.0, maximum=1.0, minimum_exclusive=True)
    return PriDeConfig(modal_k_threshold=threshold)


def model_supports_logprobs(model: ModelConfig) -> bool:
    """Single source of truth: does this model's backend expose score_options()?

    Both config validation (this module) and run_experiment's pre-run gate call
    this one function, so the two can never drift apart (FCD-2 / PF-2). Capability
    is read from the backends' own declared ``supports_logprobs`` rather than a
    hardcoded name list:
      - api backends are never logprob-capable — no API provider client
        implements score_options() (all of OpenAI, Anthropic, Gemini, Groq,
        Together, vLLM are generate-only); pride/cyclic_logprob require
        backend=huggingface (or dummy, for tests).
      - dummy / huggingface report their class capability directly; constructing
        them here is cheap (no weights are loaded until .load()).
    """
    if model.backend == "api":
        return False
    cls = _NONAPI_BACKEND_CLASSES.get(model.backend)
    if cls is None:
        return False
    if cls is HuggingFaceBackend:
        probe = cls(model.model_name_or_path, model.device)
    else:
        probe = cls()
    return bool(probe.supports_logprobs)


def _validate_logprob_compatibility(config: ExperimentConfig) -> None:
    """Every (method, model) pair requiring logprobs must have a capable backend."""
    for m in config.methods:
        for model in config.models:
            if m.requires_logprobs and not model_supports_logprobs(model):
                raise ConfigError(
                    f"Method {m.name!r} requires score_options() / logprobs, "
                    f"but model.backend={model.backend!r} / provider={model.provider!r} "
                    f"does not support it. Logprob-capable options: "
                    f"{sorted(_NONAPI_BACKEND_CLASSES)} backends. No api provider "
                    f"is accepted for config-driven runs — all API provider clients "
                    f"(including vLLM) are generate-only. "
                    f"Either drop this method or switch backends."
                )


def _validate_no_duplicate_benchmarks(config: ExperimentConfig) -> None:
    """No two benchmark entries may resolve to the same output CSV key."""
    seen_benchmark_keys: set[str] = set()
    for bench in config.benchmarks:
        # Key on the *resolved written label* (what actually lands in the CSV
        # filename / checkpoint key / benchmark_name column), not just
        # output_name — otherwise two `name: huggingface` entries that resolve
        # to the same stem would slip past and silently overwrite each other.
        key = repr(canonical_benchmark_key(bench))
        if key in seen_benchmark_keys:
            raise ConfigError(
                f"Duplicate benchmark key {key!r} in benchmarks list — "
                f"two entries would write to the same output CSV. "
                f"Set output_name on one of them to disambiguate."
            )
        seen_benchmark_keys.add(key)


def _validate_cross_field(config: ExperimentConfig) -> None:
    """Rules that span more than one section — can't be checked per-field."""
    _validate_logprob_compatibility(config)
    _validate_no_duplicate_benchmarks(config)


def canonical_benchmark_key(bench: BenchmarkConfig) -> tuple:
    """Full prepared-source identity used only to reject exact duplicates."""
    return (
        benchmark_write_label(bench), bench.split, bench.hf_path, bench.hf_subset,
        bench.source_revision, tuple(bench.transforms), bench.output_name,
        bench.n_samples, tuple(bench.subject_filter or []),
    )


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
    _reject_unknown(raw, {"experiment", "models", "benchmarks", "methods", "metrics", "run", "pride"}, "top level")
    if not isinstance(experiment, Mapping):
        raise ConfigError("experiment must be a YAML mapping.")
    _reject_unknown(experiment, {"name"}, "experiment")
    models = _build_models(raw)
    if not models:
        raise ConfigError("models must be a non-empty list.")

    config = ExperimentConfig(
        name=_nonempty(_require(experiment, "name", "experiment"), "experiment.name"),
        models=models,
        benchmarks=_build_benchmarks(raw),
        methods=_build_methods(raw.get("methods")),
        metrics=_build_metrics(raw.get("metrics")),
        run=_build_run(raw.get("run")),
        pride=_build_pride(raw.get("pride")),
    )
    _validate_cross_field(config)
    return config
