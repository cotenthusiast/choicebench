"""Compute evaluation metrics for one experiment run.

Usage:
    python scripts/evaluate_run.py --run-id toy_experiment
    python scripts/evaluate_run.py --run-id toy_experiment --reparse
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging

import pandas as pd
import yaml

from choicebench.config.paths import REPORTS_DIR, RUNS_DIR
from choicebench.io.readers import read_all_run_results
from choicebench.metrics import BUILTIN_METRICS
from choicebench.parsing.parser import parse_model_answer
from choicebench.pipeline.options import build_option_map
from choicebench.scoring.scorer import score_prediction

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
# Data loading
# ---------------------------------------------------------------------------

def load_run(run_id: str):
    """Concatenate all result CSVs from a run directory into one DataFrame."""
    run_dir = RUNS_DIR / run_id
    df = read_all_run_results(run_dir)
    if df.empty:
        raise FileNotFoundError(f"No result CSVs found in: {run_dir}")
    return df


def load_run_config(run_id: str) -> dict:
    """Load the config snapshot saved alongside the run results."""
    config_path = RUNS_DIR / run_id / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Reparsing
# ---------------------------------------------------------------------------

# parse_reason values that mark an answer NOT derived from a single raw_text
# generation. Reparsing raw_text for these would silently overwrite a correct,
# aggregated answer:
#   - majority_vote (permutation): the row stores only the first rotation's
#     raw_text; the answer is a vote across all rotations.
#   - pride_eq8 (pride) / eq1_averaging (cyclic_logprob): logprob methods with
#     raw_text=None; the answer comes from score_options, not text.
_NON_DIRECT_PARSE_REASONS = frozenset({"majority_vote", "pride_eq8", "eq1_averaging"})


def _is_reparseable_row(row) -> bool:
    """True only for direct_mcq-shaped rows whose answer came from one raw_text call.

    Multi-call (two_stage — its raw_text is the unparseable stage-2 text the
    fallback recovered from) and aggregated/logprob methods are excluded, since
    reparsing their stored raw_text does not reproduce their reported answer.
    """
    if row.get("parse_reason") in _NON_DIRECT_PARSE_REASONS:
        return False
    return row.get("method_name") == "direct_mcq"


def reparse_run(run_df: pd.DataFrame) -> pd.DataFrame:
    """Re-run the current parser and scorer on direct_mcq rows only.

    Useful when the parser has been updated and you want to recompute
    parsed_choice / is_correct without re-running expensive model inference.
    Only direct_mcq rows are touched: every other method's answer is either
    aggregated across calls or derived from logprobs, so reparsing the single
    stored raw_text would silently corrupt the reported answer (see FSF-2).
    """
    run_df = run_df.copy()
    skipped: dict[str, int] = {}
    for idx, row in run_df.iterrows():
        if row.get("model_status") == "failure":
            continue
        if not _is_reparseable_row(row):
            method = row.get("method_name") or "<unknown>"
            skipped[method] = skipped.get(method, 0) + 1
            continue
        options = build_option_map(row)
        parsed = parse_model_answer(row["raw_text"], options)
        scored = score_prediction(parsed, row["correct_option"])
        run_df.loc[idx, "parsed_choice"]  = parsed.final_choice
        run_df.loc[idx, "parse_status"]   = parsed.status
        run_df.loc[idx, "normalized_text"] = parsed.normalized_text
        run_df.loc[idx, "parse_reason"]   = parsed.reason
        run_df.loc[idx, "score_status"]   = scored.status
        run_df.loc[idx, "is_correct"]     = scored.is_correct

    if skipped:
        logger.warning(
            "--reparse only recomputes direct_mcq rows (answers from a single "
            "raw_text generation). Left untouched: %s. These methods' answers are "
            "aggregated (majority_vote), logprob-derived (pride_eq8, eq1_averaging), "
            "or multi-call (two_stage), so reparsing their raw_text would silently "
            "corrupt the reported answer.",
            ", ".join(f"{m} ({n} rows)" for m, n in sorted(skipped.items())),
        )
    return run_df


# ---------------------------------------------------------------------------
# Metric loading
# ---------------------------------------------------------------------------

def _load_metric(metric_name: str):
    """Return an instantiated metric for the given name or importlib path.

    Built-in names (e.g. "accuracy") are resolved via BUILTIN_METRICS.
    External classes are loaded via importlib using "module.path:ClassName"
    syntax — e.g. "my_package.metrics:MyMetric".
    """
    if metric_name in BUILTIN_METRICS:
        return BUILTIN_METRICS[metric_name]()
    if ":" in metric_name:
        module_path, class_name = metric_name.rsplit(":", 1)
        try:
            module = importlib.import_module(module_path)
            metric_cls = getattr(module, class_name)
            return metric_cls()
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                f"Could not load metric class {metric_name!r}: {exc}"
            ) from exc
    raise ValueError(
        f"Unknown metric {metric_name!r}. "
        f"Built-ins: {sorted(BUILTIN_METRICS)}. "
        f"For external metrics use 'module.path:ClassName'."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an MCQ experiment run using the specified run ID."
    )
    parser.add_argument(
        "--run-id", required=True, type=str,
        help="Run ID (folder name under runs/).",
    )
    parser.add_argument(
        "--reparse", action="store_true",
        help="Re-run the parser on raw_text before computing metrics. Only "
             "direct_mcq rows are affected; aggregated/logprob/multi-call "
             "methods are skipped (a warning lists them).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load results ---
    logger.info("Loading run: %s", args.run_id)
    run_df = load_run(args.run_id)
    logger.info("Loaded %d rows", len(run_df))

    # --- Optionally reparse ---
    if args.reparse:
        logger.info("Reparsing model outputs...")
        run_df = reparse_run(run_df)

    # --- Load config to get metrics list ---
    config = load_run_config(args.run_id)
    metric_names = config.get("metrics", [])
    logger.info("Metrics to compute: %s", metric_names)

    # --- Compute metrics per (method, model, benchmark) ---
    results: dict = {}
    for (method, model, bench), group in run_df.groupby(
        ["method_name", "model_name", "benchmark_name"]
    ):
        results.setdefault(method, {}).setdefault(model, {}).setdefault(bench, {})
        for metric_name in metric_names:
            metric = _load_metric(metric_name)
            results[method][model][bench].update(metric.compute(group))

    # --- Write summary JSON ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / f"{args.run_id}_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Metrics saved to %s", output_path)

    # --- Print summary ---
    logger.info("── Eval complete ────────────────────────────────")
    for method, models in results.items():
        for model, benchmarks in models.items():
            for bench, metrics in benchmarks.items():
                logger.info("  [%s | %s | %s]  %s", method, model, bench, metrics)
    logger.info("─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
