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
import math
import sys

import pandas as pd

from choicebench.clients.types import FAILURE_STATUS
from choicebench.config.paths import REPORTS_DIR, RUNS_DIR, ensure_dirs, validate_run_id
from choicebench.infra.artifacts import atomic_write_json
from choicebench.identity import canonicalize, integrity_digest, short_id
from choicebench.io.readers import ResultSetError, read_manifest_results
from choicebench.manifest import ManifestCompatibilityError, load_run_state
from choicebench.metrics import BUILTIN_METRICS
from choicebench.parsing.parser import parse_model_answer
from choicebench.pipeline.options import build_option_map
from choicebench.provenance import implementation_identity
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

# A "completed" condition (its result CSV exists) can still be entirely
# transport failures — the backend call itself never succeeded for most or
# all rows. Left unflagged, this reads identically to a genuinely bad model
# (e.g. "accuracy: 0.0, status: completed"). Once transport failures reach
# this fraction of a condition's rows, report it distinctly as
# "infra_failure" instead of silently folding it into "completed".
INFRA_FAILURE_FRACTION_THRESHOLD = 0.5


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(run_id: str):
    """Concatenate all result CSVs from a run directory into one DataFrame."""
    run_dir = RUNS_DIR / validate_run_id(run_id)
    df, _ = read_manifest_results(run_dir)
    return df


def load_run_config(run_id: str) -> dict:
    """Load the config snapshot saved alongside the run results."""
    manifest_path = RUNS_DIR / validate_run_id(run_id) / "manifest.json"
    return json.loads(manifest_path.read_text())["payload"]["config"]


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
        if row.get("transport_status") == "failure":
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


def _verified_metric_records(manifest: dict) -> tuple[list[dict], dict[str, object]]:
    declared = manifest["payload"].get("evaluation", [])
    records: list[dict] = []
    metrics: dict[str, object] = {}
    for item in declared:
        name = item["name"]
        metric = _load_metric(name)
        record = {"name": name, "implementation": implementation_identity(type(metric))}
        if canonicalize(record) != canonicalize(item):
            raise RuntimeError(
                f"Metric implementation for {name!r} no longer matches the run manifest. "
                "Evaluate with the original environment/code or start a new experiment."
            )
        records.append(record)
        metrics[name] = metric
    return sorted(records, key=lambda item: item["name"]), metrics


def build_evaluation_report(run_id: str, run_df: pd.DataFrame, manifest: dict, reparse: bool) -> dict:
    """Compute a self-identifying report bound to run and post-processing code."""
    metric_records, metrics = _verified_metric_records(manifest)
    postprocessing = (
        {
            "parser": implementation_identity(parse_model_answer),
            "option_builder": implementation_identity(build_option_map),
            "scorer": implementation_identity(score_prediction),
        }
        if reparse else None
    )
    state = load_run_state(RUNS_DIR / run_id, manifest)
    declared_conditions = {item["condition_id"]: item for item in manifest["payload"]["conditions"]}
    # Bind the exact consumed result set into the evaluation identity: the
    # ordered condition statuses plus the verified content digest of each
    # completed result file. Byte-identical result sets share an evaluation ID;
    # materially different results (e.g. after --reset-run and a rerun) receive
    # a different ID, so one report can never silently replace another.
    result_records = []
    for condition_id in sorted(declared_conditions):
        entry = state["conditions"][condition_id]
        result_records.append({
            "condition_id": condition_id,
            "status": entry.get("status"),
            "result_sha256": (
                entry.get("result_sha256") if entry.get("status") == "completed" else None
            ),
        })
    identity = canonicalize({
        "schema_version": "choicebench.evaluation.v1",
        "experiment_digest": manifest["experiment_digest"],
        "metrics": metric_records,
        "reparse": reparse,
        "postprocessing": postprocessing,
        "results": result_records,
    })
    evaluation_id = short_id("eval", identity)
    results: dict[str, dict] = {}
    for condition_id in sorted(declared_conditions):
        status = state["conditions"][condition_id]["status"]
        condition = declared_conditions[condition_id]
        if status in ("gated", "failed"):
            results[condition_id] = {
                "status": status, "method_name": _method_name(manifest, condition),
                "model_name": condition["model_display_name"],
                "benchmark_name": condition["benchmark_name"], "benchmark_split": condition["split"],
                "metrics": {},
            }
            if status == "failed" and state["conditions"][condition_id].get("error"):
                results[condition_id]["error"] = state["conditions"][condition_id]["error"]
            continue
        if status != "completed":
            raise RuntimeError(
                f"Condition {condition_id} has status {status!r}; the run has not finished. "
                "Complete, resume, or reset the run before evaluating."
            )
        group = run_df[run_df["condition_id"] == condition_id]
        if group.empty:
            raise RuntimeError(f"Completed condition {condition_id} has no validated result rows.")
        transport_failures = int((group["transport_status"] == FAILURE_STATUS).sum())
        transport_failure_fraction = transport_failures / len(group)
        condition_status = (
            "infra_failure"
            if transport_failure_fraction >= INFRA_FAILURE_FRACTION_THRESHOLD
            else "completed"
        )
        first = group.iloc[0]
        results[condition_id] = {
            "status": condition_status, "method_name": first["method_name"],
            "model_name": first["model_name"], "benchmark_name": first["benchmark_name"],
            "benchmark_split": first["benchmark_split"], "metrics": {},
            "transport_failure_count": transport_failures,
            "transport_failure_fraction": transport_failure_fraction,
        }
        for name in sorted(metrics):
            results[condition_id]["metrics"].update(metrics[name].compute(group))
    condition_counts = {
        "completed": sum(1 for item in results.values() if item["status"] == "completed"),
        "gated": sum(1 for item in results.values() if item["status"] == "gated"),
        "failed": sum(1 for item in results.values() if item["status"] == "failed"),
        "infra_failure": sum(1 for item in results.values() if item["status"] == "infra_failure"),
    }
    report = _json_safe({
        "schema_version": "choicebench.evaluation.v1", "evaluation_id": evaluation_id,
        "source": {
            "run_id": run_id, "experiment_id": manifest["experiment_id"],
            "experiment_digest": manifest["experiment_digest"],
            "manifest_payload_digest": manifest["payload_digest"],
            "protocol_version": manifest["payload"]["protocol_version"],
        },
        # "complete" means every declared condition either completed cleanly
        # or was gated by design; any failed or infra_failure condition marks
        # the report as a partial accounting of an unfinished/untrustworthy
        # experiment.
        "run_status": (
            "complete"
            if condition_counts["failed"] == 0 and condition_counts["infra_failure"] == 0
            else "partial"
        ),
        "condition_counts": condition_counts,
        "reparse": reparse, "metrics": metric_records,
        "postprocessing": postprocessing, "conditions": results,
    })
    report["report_digest"] = integrity_digest(report)
    return report


def _method_name(manifest: dict, condition: dict) -> str:
    methods = {item["method_id"]: item for item in manifest["payload"].get("methods", [])}
    return methods.get(condition["method_id"], {}).get("config", {}).get("name", condition["method_id"])


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


def write_evaluation_report(output_path, report: dict) -> None:
    """Write a report without ever silently replacing a different one.

    The evaluation ID already binds the consumed result contents, so an
    existing file at the same path must be byte-equivalent (idempotent
    re-evaluation). Anything else — a hash collision, manual tampering, or a
    stale file from another source — is refused rather than overwritten.
    """
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = None
        # Full-content comparison, not just the recorded digest: a tampered
        # file that kept its old report_digest line must still be refused.
        if existing != report:
            raise ResultSetError(
                f"Refusing to overwrite {output_path}: an existing report under this "
                "evaluation ID has different contents. Move the existing file aside "
                "before re-evaluating."
            )
        logger.info("An identical report already exists at %s; rewriting in place.", output_path)
    atomic_write_json(output_path, report)


def main() -> None:
    ensure_dirs()
    args = parse_args()
    args.run_id = validate_run_id(args.run_id)

    try:
        # --- Load results ---
        logger.info("Loading run: %s", args.run_id)
        run_dir = RUNS_DIR / args.run_id
        run_df, manifest = read_manifest_results(run_dir)
        logger.info("Loaded %d rows", len(run_df))

        # --- Optionally reparse ---
        if args.reparse:
            logger.info("Reparsing model outputs...")
            run_df = reparse_run(run_df)

        report = build_evaluation_report(args.run_id, run_df, manifest, args.reparse)

        # --- Write summary JSON ---
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / f"{args.run_id}_{report['evaluation_id']}_metrics.json"
        write_evaluation_report(output_path, report)
    except (ResultSetError, ManifestCompatibilityError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info("Metrics saved to %s", output_path)

    # --- Print summary ---
    logger.info("── Eval complete ────────────────────────────────")
    for condition_id, item in report["conditions"].items():
        logger.info(
            "  [%s | %s | %s | %s]  %s%s",
            condition_id, item["method_name"], item["model_name"], item["benchmark_name"],
            item["metrics"], "" if item["status"] == "completed" else f"  (status: {item['status']})",
        )
    if report["run_status"] != "complete":
        counts = report["condition_counts"]
        logger.warning(
            "Run %s is a PARTIAL accounting: %d condition(s) failed, %d condition(s) hit "
            "INFRA_FAILURE (transport failures on >= %d%% of rows — not a model-quality "
            "result) (%d completed, %d gated). Metrics for infra_failure conditions "
            "reflect the transport outage, not model behavior.",
            args.run_id, counts["failed"], counts["infra_failure"],
            int(INFRA_FAILURE_FRACTION_THRESHOLD * 100), counts["completed"], counts["gated"],
        )
    logger.info("─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
