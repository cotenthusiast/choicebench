# scripts/paper/run_visible_llm_matcher.py
#
# Paper-specific (eacl-2026-revision): visible_llm_matcher (the 2x2 grid's
# fourth cell) has a per-model data dependency -- Stage 1 (extracted_text)
# comes from THAT SAME model's own prior text_extraction run, which doesn't
# fit run_experiment.py's "one shared benchmark dataset across every model
# in the run" assumption. Run as a small standalone script instead of
# forcing it through the full grid machinery: one model at a time, reading
# that model's saved text_extraction result CSV, making exactly one new
# live call per question (Stage 2's LLM match).
#
# Resumable: writes results incrementally (one row at a time) and, on
# restart, skips any question_id already present in the output file --
# a crash partway through loses at most the one in-flight row, never
# silently duplicates a completed one.
#
# Run with:
#   python scripts/paper/run_visible_llm_matcher.py \
#       --text-extraction-csv runs/<run_id>/<..._text_extraction_..._<model>_<benchmark>.csv \
#       --model-config config/paper/mmlu_core_methods.yaml --model-index 0 \
#       --run-id <id> --output runs/<id>/visible_llm_matcher_<model>_<benchmark>.csv

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from choicebench.cli.run_experiment import build_backend
from choicebench.config.schema import load_config
from choicebench.infra.resumable_csv import check_resume_compatible
from choicebench.methods.library.visible_llm_matcher import VisibleLLMMatcherRunner

METHOD_NAME = "visible_llm_matcher"


def _load_completed_question_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    if "question_id" not in existing.columns:
        return set()
    return set(existing["question_id"].astype(str))


def run(
        text_extraction_csv: Path,
        model_config,
        run_id: str,
        output_path: Path,
        run_seed: int = 42,
        resume: bool = True,
) -> int:
    """Run visible_llm_matcher for every row in text_extraction_csv.

    Returns:
        Number of new rows written (excludes rows already present on resume).
    """
    source_df = pd.read_csv(text_extraction_csv)
    if "raw_text" not in source_df.columns:
        raise ValueError(
            f"{text_extraction_csv} has no 'raw_text' column -- expected a saved "
            "text_extraction result CSV."
        )

    completed_ids = _load_completed_question_ids(output_path) if resume else set()

    backend = build_backend(model_config, run_id, run_seed=run_seed)
    runner = VisibleLLMMatcherRunner(
        backend=backend, method_name=METHOD_NAME, split_name="test",
        prompt_version="v1", prompts_dir=Path("prompts"), run_id=run_id,
        seed=run_seed, benchmark_name=source_df["benchmark_name"].iloc[0] if len(source_df) else "",
        temperature=model_config.generation_kwargs.temperature,
        max_tokens=model_config.generation_kwargs.max_new_tokens,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    n_written = 0

    with open(output_path, "a", newline="") as f:
        writer = None
        for _, source_row in source_df.iterrows():
            qid = str(source_row["question_id"])
            if qid in completed_ids:
                continue
            row = source_row.to_dict()
            row["extracted_text"] = row["raw_text"]
            result = runner.run_one(row, sample_index=0)

            if writer is None:
                # file_exists reflects state BEFORE this open(path, "a")
                # call -- which itself creates an empty file the instant
                # it runs, so checking output_path.exists() from here on
                # would always see "exists" even for a genuinely fresh run.
                if file_exists:
                    check_resume_compatible(
                        output_path, exact_columns=list(result.keys()), expected_method_name=METHOD_NAME,
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
    parser.add_argument("--text-extraction-csv", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path,
                         help="A config file containing the model to use for Stage 2.")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.model_config))
    model_config = config.models[args.model_index]

    n_written = run(
        args.text_extraction_csv, model_config, args.run_id, args.output,
        run_seed=args.seed, resume=not args.no_resume,
    )
    print(f"Wrote {n_written} new rows -> {args.output}")


if __name__ == "__main__":
    main()
