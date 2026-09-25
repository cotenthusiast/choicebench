# scripts/paper/export_benchmark_questions.py
#
# Paper-specific (eacl-2026-revision): exports a config's loaded/filtered
# benchmark questions (after question_id_manifest / n_samples / sampling
# is applied, exactly as run_experiment.py's own grid would see them) to
# a plain CSV -- the "exported questions CSV" input format
# run_text_extraction_rotations.py and run_stochasticity_repeats.py both
# expect (they deliberately decouple from reading benchmark config
# directly, to stay simple and testable without a prepared-dataset
# dependency).
#
# Injects a benchmark_name column from the benchmark config's own `name`
# -- load_benchmark_selection()'s raw DataFrame has no such column at all
# (confirmed against real prepared MMLU/ARC data), but both consuming
# scripts read source_df["benchmark_name"].iloc[0]. Without this, either
# script crashes the first time it's pointed at real exported data
# instead of a synthetic test fixture that happened to already include
# the column.
#
# Point --config at any config whose benchmarks[--benchmark-index] has
# the question_id_manifest you want (the frozen eval manifest, or the
# frozen stochasticity subset) -- the rest of that config (models,
# methods) is irrelevant to this script.
#
# Run with:
#   python scripts/paper/export_benchmark_questions.py \
#       --config config/paper/mmlu_core_methods.yaml --benchmark-index 0 \
#       --output data/manifests/mmlu_eval_v2_full.csv

from __future__ import annotations

import argparse
from pathlib import Path

from choicebench.cli.run_experiment import load_benchmark_selection
from choicebench.config.schema import load_config


def export(
        config_path: Path,
        benchmark_index: int,
        output_path: Path,
        run_seed: int | None = None,
) -> int:
    """Export one benchmark's loaded question selection to CSV.

    Returns:
        Number of rows written.
    """
    config = load_config(str(config_path))
    benchmark_cfg = config.benchmarks[benchmark_index]
    seed = run_seed if run_seed is not None else config.run.seed

    selection = load_benchmark_selection(benchmark_cfg, seed)
    questions = selection.questions.copy()
    questions["benchmark_name"] = benchmark_cfg.name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    questions.to_csv(output_path, index=False)
    return len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark-index", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    n_rows = export(args.config, args.benchmark_index, args.output, run_seed=args.seed)
    print(f"Wrote {n_rows} rows -> {args.output}")


if __name__ == "__main__":
    main()
