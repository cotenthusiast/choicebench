"""Experiment runner for MCQ evaluation.

Usage:
    python scripts/run_experiment.py --config config/my_experiment.yaml
    python scripts/run_experiment.py --config config/my_experiment.yaml --dry-run
    python scripts/run_experiment.py --config config/my_experiment.yaml --run-id 20260627_120000 --yes
"""

from __future__ import annotations

import argparse
import importlib
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from choicebench.backends.api_backend import APIBackend
from choicebench.backends.dummy_backend import DummyBackend
from choicebench.backends.hf_backend import HuggingFaceBackend
from choicebench.config.paths import (
    ARC_NORMALIZED_PATH,
    MMLU_NORMALIZED_PATH,
    PROCESSED_DIR,
    PROMPTS_DIR,
    RUNS_DIR,
    TOY_BENCHMARK_PATH,
    ensure_dirs,
)
from choicebench.config.schema import (
    BENCHMARK_ARC_CHALLENGE,
    BENCHMARK_HUGGINGFACE,
    BENCHMARK_MMLU,
    BENCHMARK_TOY,
    BenchmarkConfig,
    ExperimentConfig,
    MethodConfig,
    ModelConfig,
    benchmark_normalized_stem,
    load_config,
)
from choicebench.infra.checkpoint import CheckpointManager
from choicebench.io.readers import read_benchmark
from choicebench.io.writers import write_run_results
from choicebench.registry import CLIENT_REGISTRY, METHOD_REGISTRY

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
    concurrency_limit: int = 10,
) -> APIBackend | HuggingFaceBackend | DummyBackend:
    """Construct the inference backend for a single model config."""
    backend_type = model_config.backend

    if backend_type == "api":
        client_cls = CLIENT_REGISTRY.get(model_config.provider)
        if client_cls is None:
            raise ValueError(f"Unknown provider {model_config.provider!r}. Available: {sorted(CLIENT_REGISTRY)}")
        client = client_cls(model_name=model_config.model_name_or_path)
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
    if name == BENCHMARK_MMLU:
        questions = read_benchmark(MMLU_NORMALIZED_PATH)
    elif name == BENCHMARK_ARC_CHALLENGE:
        questions = read_benchmark(ARC_NORMALIZED_PATH)
    elif name == BENCHMARK_TOY:
        questions = read_benchmark(TOY_BENCHMARK_PATH)
    elif name == BENCHMARK_HUGGINGFACE:
        stem = benchmark_normalized_stem(benchmark)
        csv_path = PROCESSED_DIR / f"{stem}_normalized.csv"
        if csv_path.exists():
            questions = read_benchmark(csv_path)
        else:
            hf_path = benchmark.hf_path
            hf_subset = benchmark.hf_subset
            cmd = f"python scripts/prepare_data.py --hf-path {hf_path}"
            if hf_subset:
                cmd += f" --hf-subset {hf_subset}"
            cmd += f" --output-name {stem}"
            raise FileNotFoundError(
                f"Normalized benchmark not found: {csv_path}\n"
                f"Run: {cmd}"
            )
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


def instantiate_runner(
    config: ExperimentConfig,
    model_config: ModelConfig,
    method_config: MethodConfig,
    backend: APIBackend | HuggingFaceBackend | DummyBackend,
    run_id: str,
    benchmark_cfg: BenchmarkConfig,
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
        )
    except TypeError as exc:
        raise TypeError(
            f"Could not instantiate method {method_name!r} with params "
            f"{method_params!r}. The method constructor must accept these "
            f"params, or the YAML should remove them."
        ) from exc


# ---------------------------------------------------------------------------
# Per-method execution
# ---------------------------------------------------------------------------

def run_method(
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
    Progress is saved to disk every checkpoint_every_n questions so a crash
    can be recovered without restarting from scratch.
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

    # --- Batched inference loop ---
    for batch_start in range(0, len(remaining), checkpoint_every_n):
        batch = remaining.iloc[batch_start : batch_start + checkpoint_every_n]
        batch_results = runner.run_many(batch)

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
    return parser.parse_args()


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

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")
    logger.info("Run ID: %s  |  Output: %s", run_id, output_dir)

    # --- Run each benchmark sequentially ---
    for benchmark_cfg in config.benchmarks:
        questions = load_benchmark(benchmark_cfg, config.run.seed)
        logger.info("Loaded %d questions from %s", len(questions), benchmark_cfg.name)

        for model_config in config.models:
            logger.info(
                "── Benchmark: %s  Model: %s (%s) ───────────────────────────────",
                benchmark_cfg.name, model_config.model_name_or_path, model_config.backend,
            )
            backend = build_backend(
                model_config, run_id, config.run.seed, config.run.concurrency_limit
            )
            logger.info("Backend: %s", backend.__class__.__name__)

            for method in config.methods:
                checkpoint_mgr = CheckpointManager(
                    checkpoint_dir=checkpoint_dir,
                    run_id=run_id,
                    condition=method.name,
                    model=model_config.model_name_or_path,
                    benchmark=benchmark_cfg.name,
                )
                runner = instantiate_runner(
                    config, model_config, method, backend, run_id, benchmark_cfg
                )
                run_method(
                    method_name=method.name,
                    runner=runner,
                    questions=questions,
                    checkpoint_mgr=checkpoint_mgr,
                    output_dir=output_dir,
                    checkpoint_every_n=config.run.checkpoint_every_n,
                    resume=config.run.resume,
                    model_name=model_config.model_name_or_path,
                    benchmark=benchmark_cfg.name,
                )

    # --- Run summary ---
    logger.info("── Run complete ─────────────────────────────────")
    logger.info("  Run ID:     %s", run_id)
    logger.info("  Results in: %s", output_dir)
    logger.info("─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
