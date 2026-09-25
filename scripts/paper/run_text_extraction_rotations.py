# scripts/paper/run_text_extraction_rotations.py
#
# Paper-specific (eacl-2026-revision): computes text_extraction's flip-rate
# by fully rerunning it under every cyclic rotation of the options -- one
# new call per rotation per question. Unlike two_stage_v1/semantic_matching_v1,
# text_extraction's options are visible in the prompt, so a rotation
# genuinely changes what the model sees; there is nothing to reuse from a
# prior run.
#
# Takes a plain questions CSV (the same shape load_benchmark_selection(...)
# .questions would produce for the frozen evaluation manifest) rather than
# reading benchmark config directly, keeping this script simple and
# testable without a prepared-dataset dependency. In production, export
# that DataFrame once with:
#   load_benchmark_selection(benchmark_cfg, run_seed).questions.to_csv(...)
#
# Resumable: writes results incrementally (one row at a time) and, on
# restart, skips any question_id already present in the output file.
#
# Run with:
#   python scripts/paper/run_text_extraction_rotations.py \
#       --questions-csv <exported questions CSV> \
#       --model-config config/paper/mmlu_core_methods.yaml --model-index 0 \
#       --run-id <id> --output runs/<id>/text_extraction_rotations_<model>_<benchmark>.csv

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

import pandas as pd

from choicebench.cli.run_experiment import build_backend
from choicebench.config.schema import load_config
from choicebench.methods.library.text_extraction import TextExtractionRunner

METHOD_NAME = "text_extraction"


def _load_completed_question_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    if "question_id" not in existing.columns:
        return set()
    return set(existing["question_id"].astype(str))


def _write_results(results: list[dict], output_path: Path, file_exists: bool) -> int:
    n_written = 0
    with open(output_path, "a", newline="") as f:
        writer = None
        for result in results:
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                if not file_exists:
                    writer.writeheader()
            writer.writerow(result)
            f.flush()  # one row at a time: a crash loses at most this one row
            n_written += 1
    return n_written


def run(
        questions_csv: Path,
        model_config,
        run_id: str,
        output_path: Path,
        run_seed: int = 42,
        resume: bool = True,
        execution_mode: str = "batch",
) -> int:
    """Run the rotation rerun for every row in questions_csv.

    ``execution_mode`` ("batch" default | "sync") is forwarded to
    build_backend() -- text_extraction rotations is batch-safe per the
    frozen Batch execution policy (every rotation call across every
    question is constructible up front), so an async-capable backend
    (BatchAPIBackend under "batch", or a plain APIBackend under "sync")
    always goes through TextExtractionRunner.run_rotations_many_async(),
    ONE call across every still-pending question's rotations -- never the
    sequential run_rotations() path, which would submit one single-request
    "batch" per rotation call and defeat the whole point. Dummy/HuggingFace
    backends (not async-capable) keep using the sequential path.

    Returns:
        Number of new rows written (excludes rows already present on resume).
    """
    source_df = pd.read_csv(questions_csv)
    if "question_text" not in source_df.columns:
        raise ValueError(
            f"{questions_csv} has no 'question_text' column -- expected an "
            "exported benchmark questions CSV."
        )

    completed_ids = _load_completed_question_ids(output_path) if resume else set()
    completed_mask = source_df["question_id"].astype(str).isin(completed_ids)
    pending_df = source_df[~completed_mask].reset_index(drop=True)

    backend = build_backend(model_config, run_id, run_seed=run_seed, execution_mode=execution_mode)
    runner = TextExtractionRunner(
        backend=backend, method_name=METHOD_NAME, split_name="test",
        prompt_version="v1", prompts_dir=Path("prompts"), run_id=run_id,
        seed=run_seed, benchmark_name=source_df["benchmark_name"].iloc[0] if len(source_df) else "",
        temperature=model_config.generation_kwargs.temperature,
        max_tokens=model_config.generation_kwargs.max_new_tokens,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    if len(pending_df) == 0:
        return 0

    if backend.is_async_capable():
        # The expensive step (batch submission) is already atomically
        # resumable on its own (BatchAPIBackend's content-addressed
        # batch-state file) -- generate_batch() returns everything at
        # once regardless, so there's no finer-grained crash window to
        # protect against here the way the sequential path below has.
        results = asyncio.run(runner.run_rotations_many_async(pending_df))
        return _write_results(results, output_path, file_exists)

    # Sync path (Dummy/HuggingFace, not async-capable): one call at a
    # time, written and flushed immediately -- a crash loses at most the
    # one in-flight row.
    n_written = 0
    with open(output_path, "a", newline="") as f:
        writer = None
        for sample_index, source_row in enumerate(pending_df.to_dict(orient="records")):
            result = runner.run_rotations(source_row, sample_index)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                if not file_exists:
                    writer.writeheader()
            writer.writerow(result)
            f.flush()
            n_written += 1
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-csv", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--execution-mode", choices=["batch", "sync"], default="batch")
    args = parser.parse_args()

    config = load_config(str(args.model_config))
    model_config = config.models[args.model_index]

    n_written = run(
        args.questions_csv, model_config, args.run_id, args.output,
        run_seed=args.seed, resume=not args.no_resume, execution_mode=args.execution_mode,
    )
    print(f"Wrote {n_written} new rows -> {args.output}")


if __name__ == "__main__":
    main()
