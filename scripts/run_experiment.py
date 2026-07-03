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
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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
    get_benchmark_path,
    safe_reset_run_dir,
)
from choicebench.config.schema import (
    BENCHMARK_HUGGINGFACE,
    BENCHMARK_TOY,
    BenchmarkConfig,
    ExperimentConfig,
    MethodConfig,
    ModelConfig,
    benchmark_normalized_stem,
    benchmark_write_label,
    load_config,
    model_supports_logprobs,
)
from choicebench.infra.checkpoint import CheckpointManager
from choicebench.io.readers import read_benchmark
from choicebench.io.writers import write_run_results
from choicebench.preflight import load_preflight
from choicebench.pride_gate import apply_modal_k_gate
from choicebench.registry import CLIENT_REGISTRY, METHOD_REGISTRY
from choicebench.stats import compute_benchmark_stats, read_stats, stats_path_for

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ConfigurationError(Exception):
    """Raised when the experiment configuration is incompatible before any run starts."""


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
        cache_dir = RUNS_DIR / run_id / "cache"
        return APIBackend(
            model_config.provider,
            model_config.model_name_or_path,
            client,
            cache_dir,
            model_config.generation_kwargs.temperature,
            model_config.generation_kwargs.max_new_tokens,
            run_seed,
            concurrency_limit,
        )
    elif backend_type == "huggingface":
        backend = HuggingFaceBackend(
            model_config.model_name_or_path,
            model_config.device,
            max_new_tokens=model_config.generation_kwargs.max_new_tokens,
            temperature=model_config.generation_kwargs.temperature,
            do_sample=model_config.generation_kwargs.do_sample,
        )
        backend.load()  # load tokenizer + weights before any generate()/score_options() call
        return backend
    elif backend_type == "dummy":
        return DummyBackend()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type!r}")


def load_benchmark(benchmark: BenchmarkConfig, run_seed: int) -> pd.DataFrame:
    """Load benchmark questions as a DataFrame, applying filters and sample cap."""
    name = benchmark.name
    if name == BENCHMARK_TOY:
        questions = read_benchmark(TOY_BENCHMARK_PATH)
    elif name == BENCHMARK_HUGGINGFACE:
        stem = benchmark_normalized_stem(benchmark)
        csv_path = PROCESSED_DIR / f"{stem}_normalized.csv"
        if csv_path.exists():
            questions = read_benchmark(csv_path)
        else:
            hf_path = benchmark.hf_path
            hf_subset = benchmark.hf_subset
            if get_by_hf_path(hf_path, hf_subset) is None:
                # No registered normalizer for this hf_path: prepare_data.py
                # would raise NotImplementedError, so do NOT send the user back
                # to it (that was a circular dead end). Every HuggingFace
                # dataset needs a registered @benchmark normalizer first.
                raise FileNotFoundError(
                    f"Normalized benchmark not found: {csv_path}\n"
                    f"No normalizer is registered for hf_path {hf_path!r} "
                    f"(subset={hf_subset!r}), so prepare_data.py cannot "
                    f"normalize it. Every HuggingFace dataset needs a "
                    f"registered @benchmark normalizer before it can be "
                    f"prepared. See the 'Add a benchmark' section of README.md: "
                    f"create src/choicebench/benchmarks/<name>.py, decorate "
                    f"build_normalized_dataframe with @benchmark(name=..., "
                    f"hf_path={hf_path!r}, hf_subset={hf_subset!r}), import it in "
                    f"src/choicebench/benchmarks/__init__.py, then run "
                    f"prepare_data.py."
                )
            cmd = f"python scripts/prepare_data.py --hf-path {hf_path}"
            if hf_subset:
                cmd += f" --hf-subset {hf_subset}"
            cmd += f" --output-name {stem}"
            raise FileNotFoundError(
                f"Normalized benchmark not found: {csv_path}\n"
                f"Run: {cmd}"
            )
    elif name in BENCHMARK_REGISTRY:
        path = get_benchmark_path(name)
        if not path.exists():
            entry = BENCHMARK_REGISTRY[name]
            cmd = f"python scripts/prepare_data.py --hf-path {entry.hf_path}"
            if entry.hf_subset:
                cmd += f" --hf-subset {entry.hf_subset}"
            if entry.default_split != "test":
                cmd += f" --split {entry.default_split}"
            # Pin the stem to the registry name so the produced file lands at
            # exactly `path`. prepare_data.py also auto-resolves this for a
            # registered hf_path, but stating it makes the hint correct
            # regardless and avoids the old ai2_arc → arc_challenge dead end.
            cmd += f" --output-name {name}"
            raise FileNotFoundError(
                f"Normalized benchmark not found: {path}\n"
                f"Run: {cmd}"
            )
        questions = read_benchmark(path)
    else:
        raise ValueError(f"Unknown benchmark: {name!r}")

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
    return questions


def _normalized_csv_path(benchmark_cfg: BenchmarkConfig) -> Path:
    """Return the expected normalized CSV path for a benchmark config.

    Mirrors load_benchmark()'s path resolution (toy / huggingface / registry)
    without the existence checks, so the modal-k gate can locate the stats
    sidecar next to the same CSV.
    """
    name = benchmark_cfg.name
    if name == BENCHMARK_TOY:
        return TOY_BENCHMARK_PATH
    if name == BENCHMARK_HUGGINGFACE:
        return PROCESSED_DIR / f"{benchmark_normalized_stem(benchmark_cfg)}_normalized.csv"
    if name in BENCHMARK_REGISTRY:
        return get_benchmark_path(name)
    raise ValueError(f"Unknown benchmark: {name!r}")


def resolve_modal_k(benchmark_cfg: BenchmarkConfig) -> int:
    """Resolve a benchmark's precomputed modal choice count.

    Prefers the persisted stats sidecar (written by prepare_data.py). Falls back
    to computing modal k from the normalized CSV if the sidecar is absent.
    """
    csv_path = _normalized_csv_path(benchmark_cfg)
    sidecar = stats_path_for(csv_path)
    if sidecar.exists():
        k = read_stats(sidecar).get("modal_k")
        if k is not None:
            return int(k)
    if csv_path.exists():
        logger.warning(
            "No stats sidecar for %s — computing modal k from the CSV.", csv_path
        )
        stats = compute_benchmark_stats(read_benchmark(csv_path))
        if stats["modal_k"] is None:
            raise ValueError(f"Cannot compute modal k for empty benchmark {csv_path}.")
        return int(stats["modal_k"])
    raise FileNotFoundError(
        f"Cannot resolve modal k: neither a stats sidecar nor the normalized "
        f"CSV exists at {csv_path}. Run scripts/prepare_data.py first."
    )


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
            prompts_dir=PROMPTS_DIR,
            run_id=run_id,
            temperature=model_config.generation_kwargs.temperature,
            max_tokens=model_config.generation_kwargs.max_new_tokens,
            seed=config.run.seed,
            model_label=model_config.model_name_or_path,
            **method_params,
            **extra_kwargs,
        )
    except TypeError as exc:
        raise TypeError(
            f"Could not instantiate method {method_name!r} with params "
            f"{method_params!r}. The method constructor must accept these "
            f"params, or the YAML should remove them."
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
        completed_ids: list[str] = checkpoint["completed_ids"]
        accumulated_results: list[dict] = checkpoint["results"]
        started_at: str = checkpoint["started_at"]
        remaining = questions[~questions["question_id"].isin(completed_ids)]
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

        accumulated_results.extend(batch_results)
        completed_ids.extend(batch["question_id"].tolist())
        checkpoint_mgr.save(completed_ids, accumulated_results, started_at)

        logger.info(
            "[%s] Progress: %d / %d questions complete",
            method_name, len(completed_ids), len(questions),
        )

    # --- Write final CSV and clean up checkpoint ---
    output_path = write_run_results(
        results=accumulated_results,
        output_dir=output_dir,
        run_id=runner.run_id,
        method_name=method_name,
        model_name=model_name,
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
) -> None:
    """Set up and run one (method, model, benchmark) combination."""
    logger.info(
        "── Benchmark: %s  Method: %s  Model: %s (%s) ──────────────────",
        benchmark_cfg.name, method.name,
        model_config.model_name_or_path, model_config.backend,
    )
    backend = build_backend(
        model_config, run_id, config.run.seed, config.run.concurrency_limit
    )
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
        modal_k = resolve_modal_k(benchmark_cfg)
        eval_questions, gate_report = apply_modal_k_gate(
            questions, modal_k, config.pride.modal_k_threshold, write_label,
        )
        logger.info(
            "[%s] modal-k gate: k=%d, %d/%d evaluated (%s).",
            method.name, gate_report.modal_k, gate_report.n_evaluated,
            gate_report.n_total, gate_report.reason,
        )
        safe_model = model_config.model_name_or_path.replace("/", "_")
        gate_path = output_dir / f"pride_modal_k_gate__{safe_model}__{write_label}.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(gate_report.as_dict(), indent=2))
        extra_runtime_kwargs = {"modal_k": modal_k, "gate_summary": gate_report.as_dict()}

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        run_id=run_id,
        condition=method.name,
        model=model_config.model_name_or_path,
        benchmark=write_label,
    )
    runner = instantiate_runner(
        config, model_config, method, backend, run_id, benchmark_cfg,
        preflight_questions=preflight_questions,
        extra_runtime_kwargs=extra_runtime_kwargs,
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
    )


def _job_label(method: MethodConfig, model_config: ModelConfig, benchmark_cfg: BenchmarkConfig) -> str:
    return (
        f"benchmark={benchmark_cfg.name} method={method.name} "
        f"model={model_config.model_name_or_path} backend={model_config.backend}"
    )


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
) -> tuple[str, Exception | None]:
    """Run one model job, catching any exception so sibling jobs are unaffected.

    Returns (job_label, exception_or_None) instead of raising, so a bug in one
    model/method/benchmark combination can't abort jobs running concurrently
    alongside it.
    """
    label = _job_label(method, model_config, benchmark_cfg)
    try:
        await _run_model(method, model_config, benchmark_cfg, questions, preflight_questions,
                         config, run_id, output_dir, checkpoint_dir)
        return label, None
    except Exception as exc:  # noqa: BLE001 - intentionally broad: isolate any job failure
        logger.error("Job failed [%s]: %s", label, exc, exc_info=True)
        return label, exc


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
) -> list[tuple[str, Exception]]:
    """Run all models for one (benchmark, method) combination.

    API models run concurrently via asyncio.gather(); HF and Dummy models
    run sequentially after, since their generate() is blocking and they have
    no async machinery to parallelise across. Each model's job is isolated:
    one model's exception is logged and does not prevent its siblings from
    completing. Returns the list of (job_label, exception) for jobs that failed.
    """
    api_models = [m for m in model_configs if m.backend == "api"]
    sync_models = [m for m in model_configs if m.backend != "api"]

    failures: list[tuple[str, Exception]] = []

    if api_models:
        results = await asyncio.gather(*[
            _run_model_isolated(method, m, benchmark_cfg, questions, preflight_questions,
                                 config, run_id, output_dir, checkpoint_dir)
            for m in api_models
        ])
        failures.extend((label, exc) for label, exc in results if exc is not None)

    for m in sync_models:
        label, exc = await _run_model_isolated(
            method, m, benchmark_cfg, questions, preflight_questions,
            config, run_id, output_dir, checkpoint_dir,
        )
        if exc is not None:
            failures.append((label, exc))

    return failures


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
            "Clear the run directory before running (fresh run, no resume). "
            "Without this flag, an existing run-id is reused: results are "
            "merged in and checkpoints resumed."
        ),
    )
    return parser.parse_args()


async def _async_main(
    config: ExperimentConfig,
    run_id: str,
    output_dir: Path,
    checkpoint_dir: Path,
) -> list[tuple[str, Exception]]:
    """Async body of the experiment: benchmark → method → concurrent models.

    Returns the list of (job_label, exception) for any model job that failed;
    an empty list means every job across every benchmark/method completed.
    """
    failures: list[tuple[str, Exception]] = []
    for benchmark_cfg in config.benchmarks:
        questions = load_benchmark(benchmark_cfg, config.run.seed)
        logger.info("Loaded %d questions from %s", len(questions), benchmark_cfg.name)

        for method in config.methods:
            preflight_questions = load_preflight(
                method, benchmark_cfg, config.run.seed,
                eval_question_ids=set(questions["question_id"]),
            )

            failures.extend(await run_models_concurrently(
                method=method,
                benchmark_cfg=benchmark_cfg,
                model_configs=config.models,
                preflight_questions=preflight_questions,
                questions=questions,
                config=config,
                run_id=run_id,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            ))
    return failures


def main() -> None:
    ensure_dirs()
    args = parse_args()
    config = load_config(args.config)
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

    # --- One-time setup ---
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = RUNS_DIR / run_id
    checkpoint_dir = output_dir / "checkpoints"

    # Default: reuse an existing run-id in place (config.yaml overwritten,
    # matching CSVs overwritten, checkpoints resumed when run.resume is set).
    # --reset-run opts into clearing the directory first for a clean run.
    if args.reset_run:
        safe_reset_run_dir(output_dir)
        logger.info("Cleared run directory (--reset-run): %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")
    logger.info("Run ID: %s  |  Output: %s", run_id, output_dir)

    failures = asyncio.run(_async_main(config, run_id, output_dir, checkpoint_dir))

    # --- Run summary ---
    logger.info("── Run complete ─────────────────────────────────")
    logger.info("  Run ID:     %s", run_id)
    logger.info("  Results in: %s", output_dir)
    if failures:
        logger.error("  Failed jobs: %d", len(failures))
        for label, exc in failures:
            logger.error("    - %s: %s", label, exc)
    else:
        logger.info("  Failed jobs: 0")
    logger.info("─────────────────────────────────────────────────")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
