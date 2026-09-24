# scripts/paper/run_visible_llm_matcher_rotations.py
#
# Paper-specific (eacl-2026-revision): computes visible_llm_matcher's
# flip-rate by rerunning ONLY its Stage-2 LLM match under every cyclic
# rotation, reusing text_extraction's OWN per-rotation extracted text
# (from a prior run_text_extraction_rotations.py run) -- Stage 1 is never
# re-elicited, since visible_llm_matcher's Stage 1 IS text_extraction's.
#
# Resumable: writes results incrementally (one row at a time) and, on
# restart, skips any question_id already present in the output file. A
# question whose every rotation has a null extracted text (every upstream
# Stage-1 call failed) still produces one unscorable row -- no calls are
# made for it, but it is not silently dropped from the output.
#
# Run with:
#   python scripts/paper/run_visible_llm_matcher_rotations.py \
#       --text-extraction-rotations-csv runs/<run_id>/text_extraction_rotations_<model>_<benchmark>.csv \
#       --model-config config/paper/mmlu_core_methods.yaml --model-index 0 \
#       --run-id <id> --output runs/<id>/visible_llm_matcher_rotations_<model>_<benchmark>.csv

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from choicebench.cli.run_experiment import build_backend
from choicebench.config.schema import load_config
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
        text_extraction_rotations_csv: Path,
        model_config,
        run_id: str,
        output_path: Path,
        run_seed: int = 42,
        resume: bool = True,
) -> int:
    """Run the Stage-2 rotation rerun for every row in
    text_extraction_rotations_csv.

    Returns:
        Number of new rows written (excludes rows already present on resume).
    """
    source_df = pd.read_csv(text_extraction_rotations_csv)
    if "per_rotation_raw_text_json" not in source_df.columns:
        raise ValueError(
            f"{text_extraction_rotations_csv} has no 'per_rotation_raw_text_json' "
            "column -- expected a saved run_text_extraction_rotations.py result CSV."
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
        for sample_index, source_row in enumerate(source_df.to_dict(orient="records")):
            qid = str(source_row["question_id"])
            if qid in completed_ids:
                continue

            per_rotation_text = json.loads(source_row["per_rotation_raw_text_json"])
            result = runner.run_matching_rotations(source_row, per_rotation_text, sample_index)

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                if not file_exists:
                    writer.writeheader()
            writer.writerow(result)
            f.flush()  # one row at a time: a crash loses at most this one row
            n_written += 1

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-extraction-rotations-csv", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path,
                         help="A config file containing the model to use for stage 2.")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.model_config))
    model_config = config.models[args.model_index]

    n_written = run(
        args.text_extraction_rotations_csv, model_config, args.run_id, args.output,
        run_seed=args.seed, resume=not args.no_resume,
    )
    print(f"Wrote {n_written} new rows -> {args.output}")


if __name__ == "__main__":
    main()
