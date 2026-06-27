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

from mcq_eval.backends.api_backend import APIBackend
from mcq_eval.backends.dummy_backend import DummyBackend
from mcq_eval.backends.hf_backend import HuggingFaceBackend
from mcq_eval.config.paths import ARC_NORMALIZED_PATH, MMLU_NORMALIZED_PATH, PROMPTS_DIR, RUNS_DIR, TOY_BENCHMARK_PATH, ensure_dirs
from mcq_eval.config.schema import (
    BENCHMARK_ARC_CHALLENGE,
    BENCHMARK_MMLU,
    BENCHMARK_TOY,
    ExperimentConfig,
    load_config,
)
from mcq_eval.infra.checkpoint import CheckpointManager
from mcq_eval.io.readers import read_benchmark
from mcq_eval.io.writers import write_run_results
from mcq_eval.methods import DirectMCQRunner, PermutationRunner, PriDeRunner, TwoStageRunner
from mcq_eval.clients import OpenAIClient, GeminiClient, GroqClient, TogetherAIClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps method name strings (used in YAML configs) to their runner classes.
# To add a built-in method: import its runner class and add an entry here.
# External methods can be registered without touching this file — use
# "module.path:ClassName" syntax in the YAML config's methods[].name field.
METHOD_REGISTRY = {
    "direct_mcq": DirectMCQRunner,
    "cyclic_permutation": PermutationRunner,
    "two_stage": TwoStageRunner,
    "pride": PriDeRunner,
}

# TODO: add Anthropic and other provider clients
CLIENT_REGISTRY = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "groq": GroqClient,
    "together": TogetherAIClient,
}

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

def build_backend(config: ExperimentConfig, run_id: str) -> APIBackend | HuggingFaceBackend | DummyBackend:
    """Construct the inference backend specified in the experiment config."""
    backend_type = config.model.backend

    if backend_type == "api":
        client_cls = CLIENT_REGISTRY.get(config.model.provider)
        client = client_cls(model_name=config.model.model_name_or_path)
        cache_dir = RUNS_DIR / run_id / "cache"
        return APIBackend(config.model.provider, config.model.model_name_or_path, client, cache_dir, config.model.generation_kwargs.temperature, config.model.generation_kwargs.max_new_tokens, config.run.seed)
    elif backend_type == "huggingface":
        return HuggingFaceBackend(config.model.model_name_or_path, config.model.device)
    elif backend_type == "dummy":
        return DummyBackend()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type!r}")


def load_benchmark(config: ExperimentConfig) -> pd.DataFrame:
    """Load benchmark questions as a DataFrame, applying n_samples cap if set."""
    name = config.benchmark.name
    if name == BENCHMARK_MMLU:
        questions = read_benchmark(MMLU_NORMALIZED_PATH)
    elif name == BENCHMARK_ARC_CHALLENGE:
        questions = read_benchmark(ARC_NORMALIZED_PATH)
    elif name == BENCHMARK_TOY:
        questions = read_benchmark(TOY_BENCHMARK_PATH)
    else:
        raise ValueError(f"Unknown benchmark: {name!r}")

    if config.benchmark.n_samples is not None:
        questions = questions.sample(
            n=config.benchmark.n_samples,
            random_state=config.run.seed,
        )
    return questions


def instantiate_runner(
    config: ExperimentConfig,
    method_name: str,
    backend: APIBackend | HuggingFaceBackend | DummyBackend,
    run_id: str,
):
    """Look up and instantiate the runner for a given method name.

    Built-in names (e.g. "direct_mcq") are resolved via METHOD_REGISTRY.
    External classes are loaded via importlib using "module.path:ClassName"
    syntax — e.g. "my_package.methods:MyRunner".
    """
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
    return runner_cls(
        backend=backend,
        method_name=method_name,
        split_name=config.benchmark.split,
        prompt_version=config.run.prompt_version,
        prompts_dir=PROMPTS_DIR,
        run_id=run_id,
        temperature=config.model.generation_kwargs.temperature,
        max_tokens=config.model.generation_kwargs.max_new_tokens,
        seed=config.run.seed,
    )


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
    benchmark: str = "",
) -> None:
    """Run a single evaluation method with checkpointing and write results to CSV.

    Resumes from a saved checkpoint if one exists, otherwise starts fresh.
    Progress is saved to disk every checkpoint_every_n questions so a crash
    can be recovered without restarting from scratch.
    """
    # --- Resume or start fresh ---
    checkpoint = checkpoint_mgr.load()
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
        model_name=runner.backend.model_name,
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
    if args.dry_run:
        logger.info("Experiment : %s", config.name)
        logger.info("Model      : %s (%s)", config.model.model_name_or_path, config.model.backend)
        logger.info("Benchmark  : %s", config.benchmark.name)
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

    backend = build_backend(config, run_id)
    logger.info("Backend: %s", backend.__class__.__name__)

    questions = load_benchmark(config)
    logger.info("Loaded %d questions from %s", len(questions), config.benchmark.name)

    # --- Run each method sequentially ---
    for method in config.methods:
        checkpoint_mgr = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            run_id=run_id,
            condition=method.name,
            model=config.model.model_name_or_path,
            benchmark=config.benchmark.name,
        )
        runner = instantiate_runner(config, method.name, backend, run_id)
        run_method(
            method_name=method.name,
            runner=runner,
            questions=questions,
            checkpoint_mgr=checkpoint_mgr,
            output_dir=output_dir,
            checkpoint_every_n=config.run.checkpoint_every_n,
            benchmark=config.benchmark.name,
        )

    # --- Run summary ---
    logger.info("── Run complete ─────────────────────────────────")
    logger.info("  Run ID:     %s", run_id)
    logger.info("  Results in: %s", output_dir)
    logger.info("─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()