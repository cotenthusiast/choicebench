# scripts/paper/run_two_stage_rotations.py
#
# Paper-specific (eacl-2026-revision): computes two_stage_v1's flip-rate by
# rerunning ONLY stage 2 (option matching) under every cyclic rotation of
# the options, reusing each question's already-saved stage-1 free-text
# answer from a prior two_stage run -- no new stage-1 call is made. Like
# visible_llm_matcher, this has a per-model data dependency (the saved
# free_text_response column comes from THAT SAME model's own prior
# two_stage run) that doesn't fit run_experiment.py's shared-benchmark
# grid, so it runs as a small standalone script instead.
#
# Resumable: writes results incrementally (one row at a time) and, on
# restart, skips any question_id already present in the output file.
# Questions whose source row has no free_text_response (a stage-1 failure
# in the original run) are skipped -- there is no free-text answer to
# reuse, so they cannot be rotation-rerun.
#
# Run with:
#   python scripts/paper/run_two_stage_rotations.py \
#       --two-stage-csv runs/<run_id>/<..._two_stage_..._<model>_<benchmark>.csv \
#       --model-config config/paper/mmlu_core_methods.yaml --model-index 0 \
#       --run-id <id> --output runs/<id>/two_stage_rotations_<model>_<benchmark>.csv

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from choicebench.cli.run_experiment import build_backend
from choicebench.config.schema import load_config
from choicebench.infra.resumable_csv import check_resume_compatible
from choicebench.methods.library.two_stage import TwoStageRunner

METHOD_NAME = "two_stage"


def _load_completed_question_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    if "question_id" not in existing.columns:
        return set()
    return set(existing["question_id"].astype(str))


def run(
        two_stage_csv: Path,
        model_config,
        run_id: str,
        output_path: Path,
        run_seed: int = 42,
        resume: bool = True,
        method_name: str = METHOD_NAME,
        prompt_version: str = "v1",
) -> int:
    """Run the stage-2 rotation rerun for every row in two_stage_csv.

    ``method_name``/``prompt_version`` default to two_stage_v1's own
    ("two_stage", "v1"); pass "reasoning_two_stage"/"v1_reasoning" to reuse
    this same script for reasoning_two_stage's flip-rate -- the only
    difference is the stamped method_name and which prompt bundle supplies
    stage 2's option_matching template, same as reasoning_two_stage's own
    accuracy run relates to two_stage's.

    Returns:
        Number of new rows written (excludes rows already present on
        resume, and rows skipped for having no free-text answer to reuse).
    """
    source_df = pd.read_csv(two_stage_csv)
    if "free_text_response" not in source_df.columns:
        raise ValueError(
            f"{two_stage_csv} has no 'free_text_response' column -- expected a "
            "saved two_stage result CSV."
        )

    completed_ids = _load_completed_question_ids(output_path) if resume else set()

    backend = build_backend(model_config, run_id, run_seed=run_seed)
    runner = TwoStageRunner(
        backend=backend, method_name=method_name, split_name="test",
        prompt_version=prompt_version, prompts_dir=Path("prompts"), run_id=run_id,
        seed=run_seed, benchmark_name=source_df["benchmark_name"].iloc[0] if len(source_df) else "",
        temperature=model_config.generation_kwargs.temperature,
        max_tokens=model_config.generation_kwargs.max_new_tokens,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    n_written = 0

    with open(output_path, "a", newline="") as f:
        writer = None
        for sample_index, source_row in enumerate(source_df.to_dict(orient="records")):
            qid = str(source_row["question_id"])
            if qid in completed_ids:
                continue
            free_text = source_row["free_text_response"]
            if pd.isna(free_text):
                continue  # stage-1 failure in the source run -- nothing to reuse

            result = runner.run_stage2_rotations(source_row, free_text, sample_index)

            if writer is None:
                # file_exists reflects state BEFORE this open(path, "a")
                # call -- which itself creates an empty file the instant
                # it runs, so checking output_path.exists() from here on
                # would always see "exists" even for a genuinely fresh run.
                if file_exists:
                    check_resume_compatible(
                        output_path, exact_columns=list(result.keys()), expected_method_name=method_name,
                    )
                writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                if not file_exists:
                    writer.writeheader()
            writer.writerow(result)
            f.flush()  # one row at a time: a crash loses at most this one row
            n_written += 1

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-stage-csv", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path,
                         help="A config file containing the model to use for stage 2.")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--method-name", default=METHOD_NAME,
                         help="Stamped on output rows; pass 'reasoning_two_stage' "
                              "together with --prompt-version v1_reasoning to reuse "
                              "this script for reasoning_two_stage's flip-rate.")
    parser.add_argument("--prompt-version", default="v1")
    args = parser.parse_args()

    config = load_config(str(args.model_config))
    model_config = config.models[args.model_index]

    n_written = run(
        args.two_stage_csv, model_config, args.run_id, args.output,
        run_seed=args.seed, resume=not args.no_resume,
        method_name=args.method_name, prompt_version=args.prompt_version,
    )
    print(f"Wrote {n_written} new rows -> {args.output}")


if __name__ == "__main__":
    main()
