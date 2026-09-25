# scripts/paper/run_stochasticity_repeats.py
#
# Paper-specific (eacl-2026-revision): the frozen stochasticity protocol --
# for the 100-MMLU / 100-ARC frozen question subsets
# (data/manifests/{mmlu,arc_challenge}_stochasticity_v2.csv), API models
# only, runs N independent repetitions (observation 0 + 3 repeats, i.e.
# n_repetitions=4) per question, in canonical (unrotated) option order.
# Methods: baseline (direct_mcq), two_stage_v1 (two_stage), reasoning_mcq,
# reasoning_two_stage. Two-stage methods do a COMPLETE independent
# Stage1->Stage2 repeat every repetition -- stage 1 is never reused across
# repetitions (unlike the flip-rate rotation scripts, which deliberately
# reuse stage 1 across rotations of the SAME call).
#
# Batch policy: direct_mcq/reasoning_mcq repetitions are batch-safe (every
# repetition's call is independently constructible up front -- it's the
# same canonical prompt repeated, not a dependency chain) and default to
# execution_mode="batch". two_stage/reasoning_two_stage repetitions must
# stay synchronous (each repetition's own stage-1/stage-2 dependency) --
# call with execution_mode="sync" for those. Either way this always goes
# through run_many_async(): for a plain "sync" APIBackend that's ordinary
# concurrent dispatch (never a real provider Batch API job -- no
# multi-wave batching is built or used here), for a "batch" BatchAPIBackend
# it's a real batch job. TwoStageRunner.run_many_async's own two
# generate_batch() calls per repetition (stage 1 wave, then stage 2 wave)
# are therefore ordinary concurrent HTTP dispatch under "sync", not two
# provider batch jobs.
#
# Each repetition gets a DISTINCT model_identity (-> distinct cache_dir /
# batch_state_dir), so repetitions never share a cache bucket. This is
# not an optimization -- without it, repetition 2/3/4 would silently
# return repetition 1's cached response for the byte-identical canonical
# prompt, measuring nothing. Resumable across (question_id,
# repetition_index) pairs, one repetition's batch of pending questions at
# a time.
#
# Run with:
#   python scripts/paper/run_stochasticity_repeats.py \
#       --questions-csv data/manifests/mmlu_stochasticity_v2.csv \
#       --model-config config/paper/mmlu_stochasticity.yaml --model-index 0 \
#       --method-name direct_mcq --prompt-version v1 \
#       --run-id <id> --output runs/<id>/stochasticity_direct_mcq_<model>_mmlu.csv \
#       --execution-mode batch

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

import pandas as pd

from choicebench.cli.run_experiment import build_backend
from choicebench.config.schema import load_config
from choicebench.registry import METHOD_REGISTRY


def _load_completed_pairs(output_path: Path) -> set[tuple[str, int]]:
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    if "question_id" not in existing.columns or "repetition_index" not in existing.columns:
        return set()
    return set(zip(
        existing["question_id"].astype(str), existing["repetition_index"].astype(int),
    ))


def _check_schema_matches(output_path: Path, result_fieldnames: list[str]) -> None:
    """Refuse to silently corrupt an existing output file whose columns
    don't match what this run is about to write (e.g. produced by an
    older version of this script) -- appending rows with a different
    field count than the file's own header produces a CSV pandas can no
    longer parse back, discovered only much later."""
    if not output_path.exists():
        return
    existing_columns = list(pd.read_csv(output_path, nrows=0).columns)
    if existing_columns != result_fieldnames:
        raise ValueError(
            f"{output_path} has a different schema than this run would write -- "
            f"existing columns {existing_columns} != {result_fieldnames}. "
            "Move or delete the stale file (it was likely produced by a "
            "different script version) before resuming."
        )


def _write_results(results: list[dict], output_path: Path, file_exists: bool) -> int:
    if results and file_exists:
        _check_schema_matches(output_path, list(results[0].keys()))
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
        method_name: str,
        prompt_version: str,
        n_repetitions: int = 4,
        run_seed: int = 42,
        resume: bool = True,
        execution_mode: str = "batch",
) -> int:
    """Run n_repetitions independent repeats of method_name over every
    row in questions_csv.

    Returns:
        Number of new (question_id, repetition_index) rows written
        (excludes pairs already present on resume).
    """
    source_df = pd.read_csv(questions_csv)
    if "question_text" not in source_df.columns:
        raise ValueError(
            f"{questions_csv} has no 'question_text' column -- expected an "
            "exported benchmark questions CSV."
        )

    runner_cls = METHOD_REGISTRY[method_name]
    completed_pairs = _load_completed_pairs(output_path) if resume else set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    n_written = 0

    for repetition_index in range(n_repetitions):
        pending_mask = ~source_df["question_id"].astype(str).apply(
            lambda qid: (qid, repetition_index) in completed_pairs
        )
        pending_df = source_df[pending_mask].reset_index(drop=True)
        if len(pending_df) == 0:
            continue

        backend = build_backend(
            model_config, run_id, run_seed=run_seed, execution_mode=execution_mode,
            model_identity=f"stochasticity_{method_name}_rep{repetition_index}",
        )
        runner = runner_cls(
            backend=backend, method_name=method_name, split_name="test",
            prompt_version=prompt_version, prompts_dir=Path("prompts"), run_id=run_id,
            seed=run_seed, benchmark_name=source_df["benchmark_name"].iloc[0] if len(source_df) else "",
            temperature=model_config.generation_kwargs.temperature,
            max_tokens=model_config.generation_kwargs.max_new_tokens,
        )

        results = asyncio.run(runner.run_many_async(pending_df))
        for result in results:
            result["repetition_index"] = repetition_index

        written_this_rep = _write_results(results, output_path, file_exists)
        if written_this_rep > 0:
            file_exists = True
        n_written += written_this_rep

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-csv", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--method-name", required=True,
                         choices=["direct_mcq", "two_stage", "reasoning_mcq", "reasoning_two_stage"])
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--n-repetitions", type=int, default=4)
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
        method_name=args.method_name, prompt_version=args.prompt_version,
        n_repetitions=args.n_repetitions, run_seed=args.seed,
        resume=not args.no_resume, execution_mode=args.execution_mode,
    )
    print(f"Wrote {n_written} new rows -> {args.output}")


if __name__ == "__main__":
    main()
