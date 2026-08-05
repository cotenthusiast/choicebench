"""Experiment runner for MCQ evaluation.

Usage:
    python scripts/run_experiment.py --config config/my_experiment.yaml
    python scripts/run_experiment.py --config config/my_experiment.yaml --dry-run
    python scripts/run_experiment.py --config config/my_experiment.yaml --run-id 20260627_120000 --yes
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import logging
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from choicebench.backends.api_backend import APIBackend
from choicebench.backends.dummy_backend import DummyBackend
from choicebench.backends.hf_backend import HuggingFaceBackend
from choicebench.benchmarks.registry import BENCHMARK_REGISTRY, get_by_hf_path
from choicebench.config.paths import (
    PROCESSED_DIR,
    PROMPTS_DIR,
    RUNS_DIR,
    TOY_BENCHMARK_PATH,
    ensure_dirs,
    safe_reset_run_dir,
    validate_run_id,
)
from choicebench.config.schema import (
    BENCHMARK_HUGGINGFACE,
    BENCHMARK_TOY,
    BenchmarkConfig,
    ConfigError,
    ExperimentConfig,
    MethodConfig,
    ModelConfig,
    benchmark_normalized_stem,
    benchmark_write_label,
    load_config,
    model_supports_logprobs,
)
from choicebench.infra.checkpoint import CheckpointManager
from choicebench.io.writers import write_run_results
from choicebench.preflight import load_preflight
from choicebench.pride_gate import ModalKGateError, ModalKGateReport, apply_modal_k_gate
from choicebench.registry import CLIENT_REGISTRY, METHOD_REGISTRY
from choicebench.stats import compute_benchmark_stats
from choicebench.datasets import (
    PreparedDataset, dataset_content_digest,
    dataset_sample_identities, load_prepared_dataset, spec_for_benchmark,
)
from choicebench.identity import canonicalize, file_digest, integrity_digest, redact_text, short_id
from choicebench.infra.atomic_io import atomic_write_json, atomic_write_text
from choicebench.infra.file_lock import FileLock, LockHeldError
from choicebench.manifest import (
    ManifestCompatibilityError, build_manifest_payload, ensure_manifest,
    load_or_create_run_state, make_manifest, write_run_state,
)
from choicebench.pipeline.prompt_builder import prompt_bundle_identity
from choicebench.preflight import PreflightSelection
from choicebench.provenance import (
    ProvenanceResolutionError, directory_digest, implementation_identity,
    resolve_hf_model_identity,
)
from choicebench.metrics import BUILTIN_METRICS
from choicebench.io.writers import result_artifact_path, validate_result_artifact

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ConfigurationError(Exception):
    """Raised when the experiment configuration is incompatible before any run starts."""


class BenchmarkSelection:
    def __init__(self, config, artifact, questions, selection_id, sample_identities):
        self.config = config
        self.artifact = artifact
        self.questions = questions
        self.selection_id = selection_id
        self.sample_identities = sample_identities


class ExecutionPlan:
    def __init__(self, selections, preflights, conditions, manifest):
        self.selections = selections
        self.preflights = preflights
        self.conditions = conditions
        self.manifest = manifest


def _prompt_root(config: ExperimentConfig):
    return Path(config.run.prompt_dir).expanduser() if config.run.prompt_dir else PROMPTS_DIR


def _write_identity_artifact(path: Path, payload: dict) -> None:
    from choicebench.identity import integrity_digest
    record = canonicalize(payload, redact_secrets=False)
    record["artifact_digest"] = integrity_digest(record)
    atomic_write_json(path, record)


def _identity_safe_params(value, key: str = ""):
    if isinstance(value, dict):
        return {name: _identity_safe_params(item, name) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_identity_safe_params(item, key) for item in value]
    path_key = key.lower() in {"source", "path", "file", "directory", "dir"} or key.lower().endswith(
        ("_path", "_file", "_dir", "_directory")
    )
    if path_key and isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if path.exists():
            if path.is_dir():
                digest, files = directory_digest(path)
                return {"logical_name": path.name, "content_digest": digest, "file_count": len(files)}
            return {"logical_name": path.name, "content_digest": file_digest(path)}
        return Path(value).name
    return value


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def build_backend(
    model_config: ModelConfig,
    run_id: str,
    run_seed: int,
    default_concurrency_limit: int = 10,
    model_identity: str | None = None,
) -> APIBackend | HuggingFaceBackend | DummyBackend:
    """Construct the inference backend for a single model config.

    A model that does not set its own ``concurrency_limit`` inherits
    ``default_concurrency_limit`` (the run-level ``run.concurrency_limit``).
    """
    backend_type = model_config.backend

    if backend_type == "api":
        client_cls = CLIENT_REGISTRY.get(model_config.provider)
        if client_cls is None:
            raise ValueError(f"Unknown provider {model_config.provider!r}. Available: {sorted(CLIENT_REGISTRY)}")
        concurrency_limit = (
            model_config.concurrency_limit
            if model_config.concurrency_limit is not None
            else default_concurrency_limit
        )
        client_kwargs: dict = {
            "model_name": model_config.model_name_or_path,
            "concurrency_limit": concurrency_limit,
        }
        if model_config.base_url is not None:
            client_kwargs["base_url"] = model_config.base_url
        client = client_cls(**client_kwargs)
        cache_dir = RUNS_DIR / run_id / "cache" / (
            model_identity or short_id("model", canonicalize(asdict(model_config)))
        )
        return APIBackend(
            model_config.provider,
            model_config.model_name_or_path,
            client,
            cache_dir,
            model_config.generation_kwargs.temperature,
            model_config.generation_kwargs.max_new_tokens,
            run_seed,
            concurrency_limit,
            model_identity,
        )
    elif backend_type == "huggingface":
        hf_kwargs = {
            "add_bos_token": model_config.add_bos_token,
            "max_new_tokens": model_config.generation_kwargs.max_new_tokens,
            "temperature": model_config.generation_kwargs.temperature,
            "do_sample": model_config.generation_kwargs.do_sample,
        }
        if model_config.revision is not None:
            hf_kwargs["revision"] = model_config.revision
        backend = HuggingFaceBackend(
            model_config.model_name_or_path, model_config.device, **hf_kwargs,
        )
        backend.load()  # load tokenizer + weights before any generate()/score_options() call
        return backend
    elif backend_type == "dummy":
        return DummyBackend()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type!r}")


def load_benchmark_selection(benchmark: BenchmarkConfig, run_seed: int) -> BenchmarkSelection:
    """Load benchmark questions as a DataFrame, applying filters and sample cap."""
    if benchmark.name == BENCHMARK_HUGGINGFACE and get_by_hf_path(benchmark.hf_path, benchmark.hf_subset) is None:
        raise FileNotFoundError(
            f"No normalizer is registered for {benchmark.hf_path!r}. Add a benchmark "
            "normalizer decorated with @benchmark before preparing this dataset."
        )
    spec = spec_for_benchmark(benchmark)
    try:
        artifact = load_prepared_dataset(PROCESSED_DIR, spec)
    except FileNotFoundError as exc:
        if benchmark.name == BENCHMARK_TOY:
            command = "choicebench-prepare-toy"
        else:
            command = f"choicebench-prepare --hf-path {spec.hf_path} --split {spec.split}"
            if spec.hf_subset:
                command += f" --hf-subset {spec.hf_subset}"
            if spec.source_revision:
                command += f" --revision {spec.source_revision}"
            if spec.output_name:
                command += f" --output-name {spec.output_name}"
        raise FileNotFoundError(f"{exc}\nPrepare it with: {command}") from exc
    questions = artifact.dataframe.copy()

    if benchmark.subject_filter:
        if "subject" in questions.columns:
            questions = questions[questions["subject"].isin(benchmark.subject_filter)]
            if questions.empty:
                raise ValueError(
                    f"benchmark.subject_filter {benchmark.subject_filter!r} "
                    f"removed all rows for benchmark {benchmark.name!r}."
                )
        else:
            logger.warning(
                "Ignoring subject_filter for benchmark %s because loaded data "
                "has no 'subject' column.",
                benchmark.name,
            )

    if benchmark.n_samples is not None:
        questions = questions.sample(
            n=benchmark.n_samples,
            random_state=run_seed,
        )
    identities = tuple(dataset_sample_identities(questions))
    selection_id = short_id("sel", {
        "artifact_id": artifact.artifact_id,
        "content_digest": dataset_content_digest(questions),
        "sample_identities": identities,
        "seed": run_seed,
        "n_samples": benchmark.n_samples,
        "subject_filter": sorted(benchmark.subject_filter or []),
    })
    return BenchmarkSelection(benchmark, artifact, questions, selection_id, identities)


def _resolve_runner_cls(method_name: str):
    """Look up a runner class by method name (built-in or importable)."""
    runner_cls = METHOD_REGISTRY.get(method_name)
    if runner_cls is None and ":" in method_name:
        module_path, class_name = method_name.rsplit(":", 1)
        try:
            module = importlib.import_module(module_path)
            runner_cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            runner_cls = None
    return runner_cls


def instantiate_runner(
    config: ExperimentConfig,
    model_config: ModelConfig,
    method_config: MethodConfig,
    backend: APIBackend | HuggingFaceBackend | DummyBackend,
    run_id: str,
    benchmark_cfg: BenchmarkConfig,
    preflight_questions: list[dict] | None = None,
    extra_runtime_kwargs: dict | None = None,
    model_label: str | None = None,
    prompts_dir: Path | None = None,
):
    """Look up and instantiate the runner for a given method name.

    Built-in names (e.g. "direct_mcq") are resolved via METHOD_REGISTRY.
    External classes are loaded via importlib using "module.path:ClassName"
    syntax — e.g. "my_package.methods:MyRunner".
    """
    method_name = method_config.name
    method_params = method_config.params
    runner_cls = METHOD_REGISTRY.get(method_name)
    if runner_cls is None:
        if ":" not in method_name:
            raise ValueError(
                f"Unknown method {method_name!r}. "
                f"Built-ins: {sorted(METHOD_REGISTRY)}. "
                f"For external methods use 'module.path:ClassName'."
            )
        module_path, class_name = method_name.rsplit(":", 1)
        try:
            module = importlib.import_module(module_path)
            runner_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                f"Could not load method class {method_name!r}: {exc}"
            ) from exc

    extra_kwargs: dict = {}
    if preflight_questions is not None:
        extra_kwargs["preflight_questions"] = preflight_questions

    # Methods that fit a calibration sidecar (e.g. PriDe) write it under
    # calibration_runs_dir / run_id. Default that to RUNS_DIR so sidecars land in
    # the run's own directory instead of the current working directory — but only
    # if the method accepts the parameter and the user did not set it in params.
    import inspect
    runner_params = inspect.signature(runner_cls.__init__).parameters
    if "calibration_runs_dir" in runner_params and "calibration_runs_dir" not in method_params:
        extra_kwargs["calibration_runs_dir"] = RUNS_DIR

    # Runtime-resolved kwargs (e.g. modal_k / gate_summary from the modal-k
    # gate). Only forwarded to runners whose constructor accepts them and that
    # the user did not already set in YAML params, so non-PriDe runners are
    # unaffected.
    for key, value in (extra_runtime_kwargs or {}).items():
        if key in runner_params and key not in method_params:
            extra_kwargs[key] = value

    try:
        return runner_cls(
            backend=backend,
            method_name=method_name,
            split_name=benchmark_cfg.split,
            prompt_version=config.run.prompt_version,
            prompts_dir=prompts_dir or _prompt_root(config),
            run_id=run_id,
            temperature=model_config.generation_kwargs.temperature,
            max_tokens=model_config.generation_kwargs.max_new_tokens,
            seed=config.run.seed,
            model_label=model_label or model_config.model_name_or_path,
            **method_params,
            **extra_kwargs,
        )
    except TypeError as exc:
        raise TypeError(
            f"Could not instantiate method {method_name!r} with the configured params. "
            "The method constructor must accept them, or the YAML should remove them."
        ) from exc


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def validate_logprob_compatibility(config: ExperimentConfig) -> None:
    """Fail fast if any (method, model) pair requires logprobs but the backend cannot provide them.

    Checks runner class attribute requires_score_options against the configured
    provider. Raises ConfigurationError before any backend is built or network
    call is made.
    """
    for method_cfg in config.methods:
        runner_cls = METHOD_REGISTRY.get(method_cfg.name)
        if runner_cls is None and ":" in method_cfg.name:
            module_path, class_name = method_cfg.name.rsplit(":", 1)
            try:
                module = importlib.import_module(module_path)
                runner_cls = getattr(module, class_name)
            except (ImportError, AttributeError):
                runner_cls = None
        if runner_cls is None or not getattr(runner_cls, "requires_score_options", False):
            continue
        for model_cfg in config.models:
            if model_supports_logprobs(model_cfg):
                continue
            provider = model_cfg.provider or model_cfg.backend
            raise ConfigurationError(
                f"Method {method_cfg.name!r} requires logprob access (score_options) "
                f"but provider {provider!r} does not support it. "
                "Use a HuggingFace backend for logprob-dependent methods."
            )


# ---------------------------------------------------------------------------
# Per-method execution
# ---------------------------------------------------------------------------

def _validate_batch_results(batch: pd.DataFrame, batch_results: object, method_name: str) -> list[dict]:
    expected = [str(value) for value in batch["question_id"].tolist()]
    if not isinstance(batch_results, list) or len(batch_results) != len(expected):
        actual = len(batch_results) if isinstance(batch_results, list) else type(batch_results).__name__
        raise RuntimeError(
            f"Method {method_name!r} returned {actual} result rows for a batch of {len(expected)}; "
            "the checkpoint was not advanced."
        )
    if not all(isinstance(row, dict) for row in batch_results):
        raise RuntimeError(f"Method {method_name!r} returned a non-mapping result row.")
    actual_ids = [str(row.get("question_id")) for row in batch_results]
    if actual_ids != expected or len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError(
            f"Method {method_name!r} result question IDs do not exactly match the input batch order."
        )
    return batch_results

async def run_method(
    method_name: str,
    runner,
    questions: pd.DataFrame,
    checkpoint_mgr: CheckpointManager,
    output_dir: Path,
    checkpoint_every_n: int,
    resume: bool = True,
    model_name: str = "",
    benchmark: str = "",
    condition_metadata: dict | None = None,
) -> None:
    """Run a single evaluation method with checkpointing and write results to CSV.

    Resumes from a saved checkpoint if one exists, otherwise starts fresh.
    Progress is saved every checkpoint_every_n questions. For API backends the
    batch is processed concurrently; for HF/Dummy backends it runs serially.
    """
    # --- Resume or start fresh ---
    if resume:
        checkpoint = checkpoint_mgr.load()
    else:
        checkpoint_mgr.delete()
        checkpoint = None

    if checkpoint is not None:
        completed_ids: list[str] = [str(value) for value in checkpoint["completed_ids"]]
        accumulated_results: list[dict] = checkpoint["results"]
        started_at: str = checkpoint["started_at"]
        remaining = questions[~questions["question_id"].isin(completed_ids)]
        expected_ids = set(str(v) for v in questions["question_id"])
        if len(completed_ids) != len(set(completed_ids)) or not set(completed_ids) <= expected_ids:
            raise RuntimeError("Checkpoint completed IDs do not match the declared dataset selection.")
        result_ids = [str(row.get("question_id")) for row in accumulated_results]
        if result_ids != completed_ids:
            raise RuntimeError("Checkpoint result rows do not match its completed question IDs.")
        logger.info(
            "[%s] Resuming from checkpoint: %d done, %d remaining",
            method_name, len(completed_ids), len(remaining),
        )
    else:
        completed_ids = []
        accumulated_results = []
        started_at = datetime.now(timezone.utc).isoformat()
        remaining = questions
        logger.info("[%s] Starting fresh: %d questions", method_name, len(remaining))

    # Use async batch path for API backends; sync path for HF / Dummy.
    backend = getattr(runner, "backend", None)
    use_async = backend is not None and isinstance(backend, APIBackend)

    # --- Batched inference loop ---
    for batch_start in range(0, len(remaining), checkpoint_every_n):
        batch = remaining.iloc[batch_start : batch_start + checkpoint_every_n]

        if use_async:
            batch_results = await runner.run_many_async(batch)
        else:
            try:
                batch_results = runner.run_many(batch)
            except RuntimeError as exc:
                if "running event loop" in str(exc) or "generate_batch" in str(exc):
                    raise ConfigurationError(
                        f"Method {method_name!r} called backend.generate() directly on "
                        "an API backend. The API backend requires the async path "
                        "(run_many_async). This is a framework bug — ensure the method "
                        "is routed through run_many_async() for API backends."
                    ) from exc
                raise

        batch_results = _validate_batch_results(batch, batch_results, method_name)
        if condition_metadata:
            for result in batch_results:
                result.update(condition_metadata)
        accumulated_results.extend(batch_results)
        completed_ids.extend(str(value) for value in batch["question_id"].tolist())
        checkpoint_mgr.save(completed_ids, accumulated_results, started_at)

        logger.info(
            "[%s] Progress: %d / %d questions complete",
            method_name, len(completed_ids), len(questions),
        )

    # --- Write final CSV and clean up checkpoint ---
    output_path = write_run_results(
        results=accumulated_results,
        output_dir=output_dir,
        condition_id=(condition_metadata or {}).get("condition_id"),
        benchmark=benchmark,
    )
    checkpoint_mgr.delete()
    logger.info("[%s] Done → %s", method_name, output_path)


# ---------------------------------------------------------------------------
# Concurrent model execution
# ---------------------------------------------------------------------------

async def _run_model(
    method: MethodConfig,
    model_config: ModelConfig,
    benchmark_cfg: BenchmarkConfig,
    questions: pd.DataFrame,
    preflight_questions: list[dict] | None,
    config: ExperimentConfig,
    run_id: str,
    output_dir: Path,
    checkpoint_dir: Path,
    backend_cache: dict[int, APIBackend | HuggingFaceBackend | DummyBackend],
    condition: dict | None = None,
) -> None:
    """Set up and run one (method, model, benchmark) combination."""
    if condition is None:
        condition = _compat_condition(method, model_config, benchmark_cfg, config.run.seed)
    existing_result = output_dir / "results" / f"{condition['condition_id']}.csv"
    if existing_result.exists():
        if not config.run.resume:
            raise ConfigurationError(
                f"Result already exists for {condition['condition_id']}. "
                "Use --reset-run for an intentional rerun or enable run.resume."
            )
        try:
            _validate_completed_result(existing_result, condition, questions)
        except ConfigurationError:
            checkpoint_path = output_dir / condition["checkpoint_path"]
            if not checkpoint_path.exists():
                raise
            logger.warning(
                "Discarding an uncommitted/corrupt result for %s because a verified checkpoint "
                "is still present; resume will reconstruct it.", condition["condition_id"],
            )
            existing_result.unlink(missing_ok=True)
            result_artifact_path(existing_result).unlink(missing_ok=True)
        else:
            logger.info("Verified completed result; skipping condition %s", condition["condition_id"])
            return
    logger.info(
        "── Benchmark: %s  Method: %s  Model: %s (%s) ──────────────────",
        benchmark_cfg.name, method.name,
        model_config.model_name_or_path, model_config.backend,
    )
    # Keyed by object identity: config.models is the same list of ModelConfig
    # instances for the whole run, so a model is built/loaded once and reused
    # across every (benchmark, method) combination instead of reloading from
    # disk on each pass through the outer loop.
    cache_key = id(model_config)
    backend = backend_cache.get(cache_key)
    if backend is None:
        runtime_model = model_config
        resolved_model = condition.get("resolved_model")
        if model_config.backend == "huggingface" and resolved_model:
            if resolved_model.get("kind") == "local":
                actual_digest, _ = directory_digest(Path(model_config.model_name_or_path).expanduser())
                if actual_digest != resolved_model.get("content_digest"):
                    raise ConfigurationError(
                        f"Local model contents changed after manifest planning for {model_config.model_name_or_path!r}."
                    )
            else:
                runtime_model = replace(model_config, revision=resolved_model["resolved_commit"])
        backend = build_backend(
            runtime_model, run_id, config.run.seed, config.run.concurrency_limit,
            condition.get("model_id"),
        )
        backend_cache[cache_key] = backend
    logger.info("Backend: %s", backend.__class__.__name__)

    # The written identity for the generic `huggingface` path is the normalized
    # stem, not the literal name "huggingface" — otherwise two distinct HF
    # datasets collide on the CSV filename / checkpoint key (FCD-3 / MF-C).
    write_label = benchmark_write_label(benchmark_cfg)

    # Modal-k compatibility gate (PriDe only). Runs before calibration so a
    # heterogeneous-option benchmark fails fast, and so PriDe calibrates and
    # evaluates on a single option count. Per-question methods skip this.
    eval_questions = questions
    extra_runtime_kwargs: dict = {}
    runner_cls = _resolve_runner_cls(method.name)
    if getattr(runner_cls, "applies_modal_k_gate", False):
        stats = compute_benchmark_stats(questions)
        modal_k = stats.get("modal_k")
        if modal_k is None:
            raise ConfigurationError(
                f"Cannot resolve modal k from the manifest-bound rows for {write_label!r}."
            )
        modal_k = int(modal_k)
        gate_path = output_dir / "artifacts" / condition["condition_id"] / "modal_k_gate.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            eval_questions, gate_report = apply_modal_k_gate(
                questions, modal_k, config.pride.modal_k_threshold, write_label,
            )
        except ModalKGateError as exc:
            # Below-threshold coverage is an intentional design exclusion, not
            # a bug — write the report so the empty result cell is traceable,
            # then re-raise so _run_model_isolated can route it to the
            # run's "gated" tally instead of "failures" (no sys.exit(1),
            # evaluate_run.py still runs).
            if exc.report is not None:
                _write_identity_artifact(gate_path, {
                    **exc.report.as_dict(), "condition_id": condition["condition_id"],
                    "experiment_id": condition["experiment_id"],
                    "dataset_artifact_id": condition["artifact_id"],
                })
            logger.info(
                "[%s] modal-k gate: benchmark %s skipped by design — %s",
                method.name, write_label, exc,
            )
            raise
        logger.info(
            "[%s] modal-k gate: k=%d, %d/%d evaluated (%s).",
            method.name, gate_report.modal_k, gate_report.n_evaluated,
            gate_report.n_total, gate_report.reason,
        )
        _write_identity_artifact(gate_path, {
            **gate_report.as_dict(), "condition_id": condition["condition_id"],
            "experiment_id": condition["experiment_id"],
            "dataset_artifact_id": condition["artifact_id"],
        })
        extra_runtime_kwargs = {
            "modal_k": modal_k, "gate_summary": gate_report.as_dict(),
            "condition_id": condition["condition_id"],
            "calibration_identity": condition["identity"].get("preflight"),
        }

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        condition_id=condition["condition_id"],
        experiment_id=condition["experiment_id"],
        selection_id=condition["selection_id"],
    )
    runner = instantiate_runner(
        config, model_config, method, backend, run_id, benchmark_cfg,
        preflight_questions=preflight_questions,
        extra_runtime_kwargs=extra_runtime_kwargs,
        model_label=condition.get("model_display_name"),
        prompts_dir=(
            output_dir / condition["prompt_snapshot_path"]
            if condition.get("prompt_snapshot_path") else None
        ),
    )
    await run_method(
        method_name=method.name,
        runner=runner,
        questions=eval_questions,
        checkpoint_mgr=checkpoint_mgr,
        output_dir=output_dir,
        checkpoint_every_n=config.run.checkpoint_every_n,
        resume=config.run.resume,
        model_name=model_config.model_name_or_path,
        benchmark=write_label,
        condition_metadata={
            "experiment_id": condition["experiment_id"],
            "condition_id": condition["condition_id"],
            "dataset_artifact_id": condition["artifact_id"],
            "dataset_selection_id": condition["selection_id"],
            "model_id": condition["model_id"], "method_id": condition["method_id"],
            "prompt_id": condition["prompt_id"], "benchmark_split": condition["split"],
        },
    )


def _job_label(method: MethodConfig, model_config: ModelConfig, benchmark_cfg: BenchmarkConfig) -> str:
    return (
        f"benchmark={benchmark_cfg.name} method={method.name} "
        f"model={model_config.model_name_or_path} backend={model_config.backend}"
    )


def _compat_condition(method, model, benchmark, seed) -> dict:
    """Identity-safe fallback for direct programmatic helper calls in legacy code."""
    payload = canonicalize({
        "method": asdict(method), "model": asdict(model),
        "benchmark": asdict(benchmark), "seed": seed,
    })
    condition_id = short_id("cond", payload)
    return {
        "condition_id": condition_id, "experiment_id": "exp_programmatic",
        "selection_id": short_id("sel", {"benchmark": asdict(benchmark), "seed": seed}),
        "artifact_id": "ds_programmatic", "model_id": short_id("model", asdict(model)),
        "method_id": short_id("method", asdict(method)), "prompt_id": "prompt_programmatic",
        "split": benchmark.split, "identity": {"preflight": None},
    }


def _validate_completed_result(path: Path, condition: dict, questions: pd.DataFrame) -> None:
    try:
        metadata = validate_result_artifact(path)
        frame = pd.read_csv(path)
    except (OSError, RuntimeError, pd.errors.EmptyDataError) as exc:
        raise ConfigurationError(f"Completed result is unreadable: {path}: {exc}") from exc
    if metadata.get("condition_id") != condition["condition_id"] or metadata.get("row_count") != len(frame):
        raise ConfigurationError(f"Completed result metadata conflicts with {condition['condition_id']}.")
    for column, expected in (
        ("condition_id", condition["condition_id"]),
        ("experiment_id", condition["experiment_id"]),
        ("dataset_selection_id", condition["selection_id"]),
        ("model_id", condition["model_id"]),
        ("method_id", condition["method_id"]),
        ("prompt_id", condition["prompt_id"]),
        ("dataset_artifact_id", condition["artifact_id"]),
        ("benchmark_split", condition["split"]),
    ):
        # Fail closed on missing/null identity values (never drop corrupt rows).
        if column not in frame or frame[column].isna().any() or set(frame[column].astype(str)) != {str(expected)}:
            raise ConfigurationError(f"Existing result {path} has invalid {column}; use --reset-run.")
    expected_ids = set(str(value) for value in questions["question_id"])
    actual_ids = [str(value) for value in frame.get("question_id", [])]
    if len(actual_ids) != len(set(actual_ids)) or not set(actual_ids) <= expected_ids:
        raise ConfigurationError(f"Existing result {path} conflicts with the dataset selection.")
    if "gate_n_evaluated" in frame and frame["gate_n_evaluated"].notna().any():
        expected_count = int(frame["gate_n_evaluated"].dropna().iloc[0])
        complete = len(actual_ids) == expected_count
    else:
        complete = set(actual_ids) == expected_ids
    if not complete:
        raise ConfigurationError(f"Existing result {path} is incomplete; use --reset-run.")


async def _run_model_isolated(
    method: MethodConfig,
    model_config: ModelConfig,
    benchmark_cfg: BenchmarkConfig,
    questions: pd.DataFrame,
    preflight_questions: list[dict] | None,
    config: ExperimentConfig,
    run_id: str,
    output_dir: Path,
    checkpoint_dir: Path,
    backend_cache: dict[int, APIBackend | HuggingFaceBackend | DummyBackend],
    condition: dict,
) -> tuple[str, Exception | None, ModalKGateReport | None]:
    """Run one model job, catching any exception so sibling jobs are unaffected.

    Returns (job_label, exception_or_None, gate_report_or_None). A populated
    exception means an unexpected failure; a populated gate_report means the
    job was skipped by design (PriDe modal-k gate below threshold) — neither
    a bug nor a completed evaluation, and not counted as a run failure.
    """
    label = f"{condition['condition_id']} {_job_label(method, model_config, benchmark_cfg)}"
    try:
        await _run_model(method, model_config, benchmark_cfg, questions, preflight_questions,
                         config, run_id, output_dir, checkpoint_dir, backend_cache, condition)
        return label, None, None
    except ModalKGateError as exc:
        return label, None, exc.report
    except Exception as exc:  # noqa: BLE001 - intentionally broad: isolate any job failure
        logger.error("Job failed [%s]: %s", label, redact_text(exc))
        return label, exc, None


async def run_models_concurrently(
    method: MethodConfig,
    benchmark_cfg: BenchmarkConfig,
    model_configs: list[ModelConfig],
    preflight_questions: list[dict] | None,
    questions: pd.DataFrame,
    config: ExperimentConfig,
    run_id: str,
    output_dir: Path,
    checkpoint_dir: Path,
    backend_cache: dict[int, APIBackend | HuggingFaceBackend | DummyBackend] | None = None,
    conditions: list[dict] | None = None,
) -> tuple[list[tuple[str, Exception]], list[tuple[str, ModalKGateReport]]]:
    """Run all models for one (benchmark, method) combination.

    API models run concurrently via asyncio.gather(); HF and Dummy models
    run sequentially after, since their generate() is blocking and they have
    no async machinery to parallelise across. Each model's job is isolated:
    one model's exception is logged and does not prevent its siblings from
    completing. Returns (failures, gated): failures are (job_label, exception)
    pairs for unexpected errors; gated are (job_label, gate_report) pairs for
    jobs skipped by design (PriDe modal-k gate) — not failures.

    backend_cache persists loaded backends across calls (keyed by model config
    identity) so a model already built for an earlier benchmark/method is
    reused instead of rebuilt. Defaults to a fresh, call-scoped dict when not
    supplied by the caller.
    """
    if backend_cache is None:
        backend_cache = {}

    if conditions is None:
        conditions = [_compat_condition(method, model, benchmark_cfg, config.run.seed) for model in model_configs]
    if len(conditions) != len(model_configs):
        raise ConfigurationError("Canonical condition records are required for every model.")
    api_models = [(i, m) for i, m in enumerate(model_configs) if m.backend == "api"]
    sync_models = [(i, m) for i, m in enumerate(model_configs) if m.backend != "api"]

    failures: list[tuple[str, Exception]] = []
    gated: list[tuple[str, ModalKGateReport]] = []

    if api_models:
        results = await asyncio.gather(*[
            _run_model_isolated(method, m, benchmark_cfg, questions, preflight_questions,
                                 config, run_id, output_dir, checkpoint_dir, backend_cache,
                                 conditions[i])
            for i, m in api_models
        ])
        for label, exc, gate_report in results:
            if exc is not None:
                failures.append((label, exc))
            elif gate_report is not None:
                gated.append((label, gate_report))

    for i, m in sync_models:
        label, exc, gate_report = await _run_model_isolated(
            method, m, benchmark_cfg, questions, preflight_questions,
            config, run_id, output_dir, checkpoint_dir, backend_cache,
            conditions[i],
        )
        if exc is not None:
            failures.append((label, exc))
        elif gate_report is not None:
            gated.append((label, gate_report))

    return failures, gated


def build_execution_plan(config: ExperimentConfig) -> ExecutionPlan:
    """Resolve all scientific inputs and expand the immutable condition grid."""
    prompt_record = prompt_bundle_identity(config.run.prompt_version, _prompt_root(config))
    prompt_record["run_snapshot_path"] = f"artifacts/prompts/{prompt_record['prompt_id']}"
    model_records: list[dict] = []
    for model in config.models:
        raw = asdict(model)
        if model.backend == "huggingface":
            try:
                resolved = resolve_hf_model_identity(model.model_name_or_path, model.revision)
            except ProvenanceResolutionError as exc:
                raise ConfigurationError(str(exc)) from exc
            display_name = (
                Path(model.model_name_or_path).expanduser().name
                if resolved["kind"] == "local" else model.model_name_or_path
            )
            scientific = {
                "backend": model.backend, "model": resolved, "device": model.device,
                "add_bos_token": model.add_bos_token,
                "generation_kwargs": raw["generation_kwargs"],
                "loader": {
                    "trust_remote_code": True,
                    "torch_dtype": "float32" if model.device == "cpu" else "float16",
                },
            }
        elif model.backend == "api":
            display_name = model.model_name_or_path
            # API requests carry only temperature/max tokens: do_sample is a
            # HuggingFace-only knob the API backend never consumes, so it must
            # not split scientifically identical API conditions.
            api_generation = {
                key: value for key, value in raw["generation_kwargs"].items() if key != "do_sample"
            }
            scientific = {
                "backend": model.backend, "provider": model.provider,
                "model_name_or_path": model.model_name_or_path,
                "base_url": model.base_url, "generation_kwargs": api_generation,
            }
            resolved = None
        else:
            display_name = model.model_name_or_path
            scientific = {"backend": model.backend, "model_name_or_path": display_name}
            resolved = None
        payload = canonicalize(scientific)
        model_records.append({
            "model_id": short_id("model", payload), "config": {
                **payload, "model_name_or_path": display_name,
            }, "resolved_model": resolved,
        })
    method_records: list[dict] = []
    runtime_parameters = {
        "self", "args", "kwargs", "backend", "method_name", "split_name", "prompt_version",
        "prompts_dir", "run_id", "temperature", "max_tokens", "seed", "perturbation_name",
        "model_label", "preflight_questions", "calibration_questions", "calibration_runs_dir",
        "modal_k", "gate_summary", "condition_id", "calibration_identity",
    }
    for method in config.methods:
        runner_cls = _resolve_runner_cls(method.name)
        if runner_cls is None:
            raise ConfigurationError(f"Cannot resolve method implementation {method.name!r}.")
        effective_params: dict[str, object] = {}
        for name, parameter in inspect.signature(runner_cls.__init__).parameters.items():
            if name not in runtime_parameters and parameter.default is not inspect.Parameter.empty:
                effective_params[name] = parameter.default
        effective_params.update(method.params)
        effective_params = _identity_safe_params(effective_params)
        preflight_config = None
        if method.preflight is not None:
            preflight_config = asdict(method.preflight)
            if preflight_config["source"] != "benchmark":
                source_path = Path(preflight_config["source"]).expanduser()
                preflight_config["source"] = source_path.name
        payload = canonicalize({
            "name": method.name, "effective_params": effective_params,
            "preflight": preflight_config, "implementation": implementation_identity(runner_cls),
        })
        method_records.append({
            "method_id": short_id("method", payload),
            "config": {"name": method.name, "params": effective_params, "preflight": preflight_config},
            "implementation": payload["implementation"],
        })

    selections: list[BenchmarkSelection] = []
    dataset_records: list[dict] = []
    preflights: dict[tuple[int, int], PreflightSelection | None] = {}
    calibration_records: list[dict] = []
    seen_calibrations: set[str] = set()
    for bi, benchmark in enumerate(config.benchmarks):
        selection = load_benchmark_selection(benchmark, config.run.seed)
        selections.append(selection)
        dataset_records.append({
            "selection_id": selection.selection_id,
            "artifact_id": selection.artifact.artifact_id,
            "benchmark": benchmark.name,
            "split": benchmark.split,
            "prepared_path": str(selection.artifact.path.relative_to(PROCESSED_DIR)),
            "prepared_content_digest": selection.artifact.content_digest,
            "prepared_metadata": selection.artifact.metadata,
            "selected_content_digest": dataset_content_digest(selection.questions),
            "selected_question_ids": [str(v) for v in selection.questions["question_id"].tolist()],
            "selected_sample_identities": list(selection.sample_identities),
            "run_snapshot_path": f"artifacts/datasets/{selection.selection_id}.csv",
            "row_count": len(selection.questions),
        })
        for mi, method in enumerate(config.methods):
            value = load_preflight(
                method, benchmark, config.run.seed,
                eval_question_ids=set(selection.questions["question_id"]),
                eval_artifact_id=selection.artifact.artifact_id,
                eval_sample_identities=set(selection.sample_identities),
                return_selection=True,
            )
            preflights[(bi, mi)] = value
            if value is not None and value.selection_id not in seen_calibrations:
                seen_calibrations.add(value.selection_id)
                calibration_records.append({
                    "selection_id": value.selection_id, "artifact_id": value.artifact_id,
                    "source": value.source, "split": value.split,
                    "prepared_path": value.artifact_path,
                    "prepared_content_digest": value.content_digest,
                    "selected_content_digest": dataset_content_digest(pd.DataFrame(value.records)),
                    "prepared_metadata": value.artifact_metadata,
                    "selected_question_ids": list(value.question_ids),
                    "selected_sample_identities": list(value.sample_identities),
                    "row_count": len(value.records),
                    "run_snapshot_path": f"artifacts/calibrations/{value.selection_id}.csv",
                })

    conditions: dict[tuple[int, int, int], dict] = {}
    seen: set[str] = set()
    condition_list: list[dict] = []
    for bi, selection in enumerate(selections):
        benchmark = config.benchmarks[bi]
        for mi, method in enumerate(config.methods):
            preflight = preflights[(bi, mi)]
            for model_i, model in enumerate(config.models):
                payload = {
                    "benchmark": {
                        "name": benchmark.name, "split": benchmark.split,
                        "selection_id": selection.selection_id,
                        "artifact_id": selection.artifact.artifact_id,
                    },
                    "preflight": (
                        {"artifact_id": preflight.artifact_id, "selection_id": preflight.selection_id,
                         "split": preflight.split, "content_digest": preflight.content_digest}
                        if preflight else None
                    ),
                    "model_id": model_records[model_i]["model_id"],
                    "method_id": method_records[mi]["method_id"],
                    "prompt_id": prompt_record["prompt_id"],
                    "prompt_snapshot_path": prompt_record["run_snapshot_path"],
                    "seed": config.run.seed,
                }
                runner_cls = _resolve_runner_cls(method.name)
                if getattr(runner_cls, "applies_modal_k_gate", False):
                    payload["protocol_settings"] = {
                        "pride_modal_k_threshold": config.pride.modal_k_threshold,
                    }
                condition_id = short_id("cond", payload)
                if condition_id in seen:
                    raise ConfigurationError(
                        f"Duplicate scientific condition {condition_id}. Remove identical duplicate entries; "
                        "parameterized methods/models receive distinct IDs automatically."
                    )
                seen.add(condition_id)
                record = {
                    "condition_id": condition_id,
                    "benchmark_name": benchmark_write_label(benchmark), "split": benchmark.split,
                    "selection_id": selection.selection_id,
                    "artifact_id": selection.artifact.artifact_id,
                    "model_id": model_records[model_i]["model_id"],
                    "method_id": method_records[mi]["method_id"],
                    "prompt_id": prompt_record["prompt_id"],
                    "prompt_snapshot_path": prompt_record["run_snapshot_path"],
                    "result_path": f"results/{condition_id}.csv",
                    "checkpoint_path": f"checkpoints/{condition_id}.json",
                    "result_metadata_path": f"results/{condition_id}.artifact.json",
                    "model_display_name": model_records[model_i]["config"]["model_name_or_path"],
                    "resolved_model": model_records[model_i]["resolved_model"],
                    "identity": canonicalize(payload),
                }
                if getattr(runner_cls, "applies_modal_k_gate", False):
                    record["gate_path"] = f"artifacts/{condition_id}/modal_k_gate.json"
                conditions[(bi, mi, model_i)] = record
                condition_list.append(record)

    metric_records = []
    for name in config.metrics:
        metric_cls = BUILTIN_METRICS.get(name)
        if metric_cls is None and ":" in name:
            module_name, class_name = name.rsplit(":", 1)
            metric_cls = getattr(importlib.import_module(module_name), class_name)
        metric_records.append({"name": name, "implementation": implementation_identity(metric_cls)})
    payload = build_manifest_payload(
        config=config, datasets=dataset_records, prompts=prompt_record,
        models=model_records, methods=method_records, conditions=condition_list,
        calibrations=calibration_records,
    )
    payload["evaluation"] = canonicalize(sorted(metric_records, key=lambda item: item["name"]))
    return ExecutionPlan(selections, preflights, conditions, make_manifest(payload))


def _ensure_run_dataset_snapshots(run_dir: Path, plan: ExecutionPlan, manifest: dict) -> None:
    """Archive the exact selected evaluation/calibration rows named by the manifest."""
    eval_frames = {selection.selection_id: selection.questions for selection in plan.selections}
    calibration_frames = {
        value.selection_id: pd.DataFrame(value.records)
        for value in plan.preflights.values() if value is not None
    }
    for section, frames in (("datasets", eval_frames), ("calibrations", calibration_frames)):
        for record in manifest["payload"].get(section, []):
            frame = frames[record["selection_id"]]
            expected = record.get("selected_content_digest")
            if dataset_content_digest(frame) != expected:
                raise ConfigurationError(f"Resolved {section[:-1]} rows no longer match the manifest.")
            path = run_dir / record["run_snapshot_path"]
            if path.exists():
                existing = pd.read_csv(path, dtype={"question_id": "string"})
                if dataset_content_digest(existing) != expected:
                    raise ConfigurationError(f"Archived run dataset snapshot is corrupt: {path}")
            else:
                atomic_write_text(path, frame.to_csv(index=False))


def _ensure_run_prompt_snapshot(run_dir: Path, manifest: dict) -> None:
    """Archive and verify the exact prompt bytes consumed by every runner."""
    record = manifest["payload"]["prompts"]
    root = run_dir / record["run_snapshot_path"]
    version_root = root / record["version"]
    declared = {f"{name}.txt" for name in record["files"]}
    if version_root.exists():
        unexpected = {path.name for path in version_root.glob("*.txt")} - declared
        if unexpected:
            raise ConfigurationError(f"Archived prompt snapshot has unexpected files: {sorted(unexpected)}")
    for name, item in record["files"].items():
        path = version_root / f"{name}.txt"
        content = item["content"]
        if path.exists():
            if path.read_text(encoding="utf-8") != content or integrity_digest(content) != item["sha256"]:
                raise ConfigurationError(f"Archived run prompt snapshot is corrupt: {path}")
        else:
            atomic_write_text(path, content)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an MCQ evaluation experiment with the specified configuration."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the experiment YAML config file.",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Explicit run ID; auto-generates a timestamp if omitted.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config and exit without running anything.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the confirmation prompt.",
    )
    parser.add_argument(
        "--reset-run", action="store_true",
        help=(
            "Clear the run directory before running. This is refused while another "
            "process holds the run lock. Without this flag, reuse is allowed only when "
            "the newly resolved immutable manifest is identical."
        ),
    )
    return parser.parse_args()


async def _async_main(
    config: ExperimentConfig,
    run_id: str,
    output_dir: Path,
    checkpoint_dir: Path,
    plan: ExecutionPlan | None = None,
) -> tuple[list[tuple[str, Exception]], list[tuple[str, ModalKGateReport]]]:
    """Async body of the experiment: benchmark → method → concurrent models.

    Returns (failures, gated). failures are (job_label, exception) pairs for
    unexpected errors — a non-empty list means the run did not fully succeed.
    gated are (job_label, gate_report) pairs for jobs skipped by design (PriDe
    modal-k gate below threshold) — expected, and not a run failure.
    """
    failures: list[tuple[str, Exception]] = []
    gated: list[tuple[str, ModalKGateReport]] = []
    # Shared across every benchmark/method iteration so each model is built
    # and loaded (weights onto GPU, for HF) exactly once per run, not once
    # per (benchmark, method) combination.
    backend_cache: dict[int, APIBackend | HuggingFaceBackend | DummyBackend] = {}
    if plan is None:
        plan = build_execution_plan(config)
        for condition in plan.conditions.values():
            condition["experiment_id"] = plan.manifest["experiment_id"]
    for bi, benchmark_cfg in enumerate(config.benchmarks):
        questions = plan.selections[bi].questions
        logger.info("Loaded %d questions from %s", len(questions), benchmark_cfg.name)

        for mi, method in enumerate(config.methods):
            preflight = plan.preflights[(bi, mi)]
            preflight_questions = preflight.records if preflight is not None else None

            combo_failures, combo_gated = await run_models_concurrently(
                method=method,
                benchmark_cfg=benchmark_cfg,
                model_configs=config.models,
                preflight_questions=preflight_questions,
                questions=questions,
                config=config,
                run_id=run_id,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                backend_cache=backend_cache,
                conditions=[plan.conditions[(bi, mi, model_i)] for model_i in range(len(config.models))],
            )
            failures.extend(combo_failures)
            gated.extend(combo_gated)
    return failures, gated


def main() -> None:
    ensure_dirs()
    args = parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info("Loaded config: %s", args.config)

    # --- Dry run: print summary and exit ---
    if args.dry_run or config.run.dry_run:
        logger.info("Experiment : %s", config.name)
        logger.info("Models     : %s", [m.model_name_or_path for m in config.models])
        logger.info("Benchmarks : %s", [b.name for b in config.benchmarks])
        logger.info("Methods    : %s", [m.name for m in config.methods])
        logger.info("Dry run complete — exiting.")
        return

    # --- Early validation: fail before building any backends ---
    validate_logprob_compatibility(config)

    # --- Confirmation prompt ---
    if not args.yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            logger.info("Aborted.")
            return

    # --- Resolve scientific inputs and immutable identity before backend calls ---
    run_id = validate_run_id(args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    output_dir = RUNS_DIR / run_id
    checkpoint_dir = output_dir / "checkpoints"

    lock_path = RUNS_DIR / ".locks" / f"{run_id}.run.lock"
    try:
        with FileLock(lock_path, f"execute or reset run {run_id}"):
            _execute_locked_run(args, config, run_id, output_dir, checkpoint_dir)
    except (ManifestCompatibilityError, LockHeldError) as exc:
        # Expected refusals (incompatible resume, rerun without resume, another
        # live owner). Surface the actionable message, not a traceback.
        logger.error("%s", exc)
        sys.exit(1)


def _execute_locked_run(args, config, run_id: str, output_dir: Path, checkpoint_dir: Path) -> None:

    if args.reset_run:
        safe_reset_run_dir(output_dir)
        logger.info("Cleared run directory (--reset-run): %s", output_dir)

    plan = build_execution_plan(config)
    manifest, reused = ensure_manifest(output_dir, plan.manifest)
    # An identical experiment already lives under this run ID. Without
    # run.resume every condition job would refuse to touch its existing result
    # and be recorded as failed — silently downgrading a valid completed run
    # (e.g. after an accidental Slurm requeue). Refuse up front instead, before
    # any state is written, so the existing run stays byte-identical.
    if reused and not config.run.resume:
        raise ManifestCompatibilityError(
            f"Run '{run_id}' already exists for this exact experiment and run.resume is "
            "disabled. Enable run.resume to verify and skip completed conditions, pass "
            "--reset-run to intentionally discard the existing run, or choose a new "
            "--run-id. The existing run directory was left unchanged."
        )
    plan.manifest = manifest
    for condition in plan.conditions.values():
        condition["experiment_id"] = manifest["experiment_id"]
    state = load_or_create_run_state(output_dir, manifest)
    _ensure_run_dataset_snapshots(output_dir, plan, manifest)
    _ensure_run_prompt_snapshot(output_dir, manifest)
    if not reused:
        atomic_write_text(
            output_dir / "config.yaml",
            yaml.safe_dump(manifest["payload"]["config"], sort_keys=False),
        )
    logger.info("Run ID: %s  |  Output: %s", run_id, output_dir)
    logger.info("Experiment identity: %s", manifest["experiment_id"])

    failures, gated = asyncio.run(_async_main(config, run_id, output_dir, checkpoint_dir, plan))
    failed_errors = {label.split()[0]: redact_text(exc) for label, exc in failures}
    gated_ids = {label.split()[0] for label, _ in gated}
    for condition_id, item in state["conditions"].items():
        result_path = output_dir / "results" / f"{condition_id}.csv"
        if condition_id in failed_errors:
            item["status"] = "failed"
            item["error"] = failed_errors[condition_id]
        elif condition_id in gated_ids:
            item["status"] = "gated"
            item.pop("error", None)
        elif result_path.exists():
            result_metadata = validate_result_artifact(result_path)
            item["status"] = "completed"
            item["result_path"] = str(result_path.relative_to(output_dir))
            item["result_sha256"] = result_metadata["file_sha256"]
            item.pop("error", None)
    write_run_state(output_dir, state)

    # --- Run summary ---
    logger.info("── Run complete ─────────────────────────────────")
    logger.info("  Run ID:     %s", run_id)
    logger.info("  Results in: %s", output_dir)
    if failures:
        logger.error("  Failed jobs: %d", len(failures))
        for label, exc in failures:
            logger.error("    - %s: %s", label, redact_text(exc))
    else:
        logger.info("  Failed jobs: 0")
    if gated:
        logger.info("  Gated by design (PriDe modal-k threshold): %d", len(gated))
        for label, report in gated:
            logger.info("    - %s: %s", label, report.reason)
    logger.info("─────────────────────────────────────────────────")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
