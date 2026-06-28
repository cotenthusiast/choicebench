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

def reparse_run(run_df: pd.DataFrame) -> pd.DataFrame:
    """Re-run the current parser and scorer on all non-failed rows.

    Useful when the parser has been updated and you want to recompute
    parsed_choice / is_correct without re-running expensive model inference.
    """
    run_df = run_df.copy()
    for idx, row in run_df.iterrows():
        if row.get("model_status") == "failure":
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
        help="Re-run the parser on raw_text before computing metrics.",
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

    # --- Compute metrics per (method, model) ---
    results: dict = {}
    for (method, model), group in run_df.groupby(["method_name", "model_name"]):
        results.setdefault(method, {}).setdefault(model, {})
        for metric_name in metric_names:
            metric = _load_metric(metric_name)
            results[method][model].update(metric.compute(group))

    # --- Write summary JSON ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / f"{args.run_id}_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Metrics saved to %s", output_path)

    # --- Print summary ---
    logger.info("── Eval complete ────────────────────────────────")
    for method, models in results.items():
        for model, metrics in models.items():
            logger.info("  [%s | %s]  %s", method, model, metrics)
    logger.info("─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
