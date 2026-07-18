# examples/pride_from_artifacts.py
#
# Offline PriDe (Zheng et al., ICLR 2024, arXiv:2309.03882) recompute for the
# reproduction wired in examples/pride_reproduction.yaml. Reads the completed
# direct_logprob + cyclic_logprob result CSVs for --run-id and reconstructs the
# full PriDe grid (every alpha, every calibration seed) purely from the
# per-question option_distributions_json those two methods already persisted —
# zero additional inference. See examples/pride_reproduction_targets.json for
# the official Table 3 numbers this grid is compared against.
#
# Usage:
#   python examples/pride_from_artifacts.py --run-id pride_reproduction_123 \
#       --alphas 0.05 0.40 0.80 --seeds 0 1 2 3 4 \
#       --targets examples/pride_reproduction_targets.json \
#       --out reports/pride_reproduction_123_pride_grid.json

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from choicebench.clients.types import SUCCESS_STATUS
from choicebench.config.paths import RUNS_DIR
from choicebench.config.paths import validate_run_id
from choicebench.constants import letters_for
from choicebench.io.readers import read_all_run_results, read_manifest_results
from choicebench.identity import file_digest, short_id
from choicebench.infra.artifacts import atomic_write_json
from choicebench.manifest import validate_manifest
from choicebench.methods.library.pride_math import (
    average_prior_probability_vectors,
    equation1_cyclic_debiased_content_probs,
    equation7_prior_from_rollouts,
    equation8_debiased_content_probs,
)
from choicebench.metrics import BUILTIN_METRICS

logger = logging.getLogger(__name__)

# examples/pride_reproduction_targets.json keys, by alpha value.
_ALPHA_TARGET_KEYS: dict[float, str] = {0.05: "pride_5", 0.40: "pride_40", 0.80: "pride_80"}


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    """One question's precomputed PriDe ingredients, derived once from its
    direct_logprob/cyclic_logprob CSV rows so every (alpha, seed) cell reuses
    the same arrays instead of re-parsing JSON or re-averaging permutations.
    """

    question_id: str
    correct_option: str
    letters: tuple[str, ...]
    default_probs: np.ndarray  # direct_logprob's observed distribution, Eq. 8's numerator
    eq1_probs: np.ndarray  # Eq. 1 debiased content probs — cyclic_logprob's own answer
    eq7_prior: np.ndarray  # Eq. 7 per-question prior — averaged into the global prior


def _parse_matrix(raw: object) -> np.ndarray | None:
    """Parse an option_distributions_json cell into a 2-D float array, or None.

    Returns None (rather than raising) for missing/NaN/malformed cells so the
    caller can treat them as an exclusion from the scored subset, matching
    direct_logprob/cyclic_logprob's own answer_status semantics.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and raw != raw:  # NaN
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    arr = np.array(parsed, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    return arr


def _load_method_frame(run_dir: Path, method_name: str) -> pd.DataFrame:
    """Load one method's result CSV(s) for a run, indexed by question_id.

    Raises:
        FileNotFoundError: no result CSVs found for this method under run_dir.
        ValueError: the CSV(s) contain more than one row for the same
            question_id (e.g. a stale/duplicated checkpoint) — recomputing
            PriDe from an ambiguous set of rows would silently pick one at
            random via pandas indexing, so this fails loudly instead.
    """
    if (run_dir / "manifest.json").exists():
        df, _ = read_manifest_results(run_dir)
        df = df[df["method_name"] == method_name].copy()
    else:
        df = read_all_run_results(run_dir, method_name=method_name)
    if df.empty:
        raise FileNotFoundError(
            f"No {method_name!r} result CSVs found under {run_dir}. This script "
            "recomputes PriDe entirely from completed direct_logprob/"
            "cyclic_logprob runs — run examples/pride_reproduction.yaml (or "
            "the matching HPC job) first."
        )
    dupes = df["question_id"][df["question_id"].duplicated()].unique()
    if len(dupes):
        raise ValueError(
            f"{method_name} results under {run_dir} contain {len(dupes)} "
            f"duplicate question_id value(s) (e.g. {dupes[0]!r}); expected "
            "exactly one row per question. Investigate before recomputing "
            "PriDe — a stale or doubly-checkpointed run would silently "
            "corrupt the scored subset."
        )
    return df.set_index("question_id", drop=False)


def build_question_records(
        direct_df: pd.DataFrame,
        cyclic_df: pd.DataFrame,
) -> dict[str, QuestionRecord]:
    """Intersect direct_logprob/cyclic_logprob rows into per-question records.

    A question_id is included only if BOTH methods produced a successful,
    shape-consistent option_distributions_json for it — PriDe's Eq. 1/7/8 all
    need both the default distribution (direct_logprob) and the cyclic
    rollout matrix (cyclic_logprob) for the same question. Rows that failed
    or are unparseable in either CSV, or are present in only one CSV, are
    dropped from the scored subset entirely (not imputed), so a partially
    failed run shrinks N rather than silently contaminating an alpha/seed
    cell.
    """
    direct_ids = set(direct_df.index)
    cyclic_ids = set(cyclic_df.index)
    common_ids = sorted(direct_ids & cyclic_ids)
    n_direct_only = len(direct_ids - cyclic_ids)
    n_cyclic_only = len(cyclic_ids - direct_ids)
    if n_direct_only or n_cyclic_only:
        logger.info(
            "Intersection policy: dropping %d question(s) present only in "
            "direct_logprob and %d present only in cyclic_logprob.",
            n_direct_only, n_cyclic_only,
        )

    records: dict[str, QuestionRecord] = {}
    n_excluded = 0
    for qid in common_ids:
        d_row = direct_df.loc[qid]
        c_row = cyclic_df.loc[qid]
        if d_row.get("answer_status") != SUCCESS_STATUS or c_row.get("answer_status") != SUCCESS_STATUS:
            n_excluded += 1
            continue

        default_mat = _parse_matrix(d_row.get("option_distributions_json"))
        cyclic_mat = _parse_matrix(c_row.get("option_distributions_json"))
        if default_mat is None or cyclic_mat is None:
            n_excluded += 1
            continue

        n = cyclic_mat.shape[0]
        if default_mat.shape != (1, n) or cyclic_mat.shape != (n, n):
            n_excluded += 1
            continue

        correct_option = str(d_row["correct_option"])
        if correct_option != str(c_row["correct_option"]):
            raise ValueError(
                f"Question {qid!r} has mismatched correct_option between "
                f"direct_logprob ({correct_option!r}) and cyclic_logprob "
                f"({c_row['correct_option']!r}) — the two CSVs do not agree "
                "on the same benchmark; refusing to recompute PriDe."
            )

        letters = tuple(letters_for(n))
        if correct_option not in letters:
            n_excluded += 1
            continue

        records[qid] = QuestionRecord(
            question_id=qid,
            correct_option=correct_option,
            letters=letters,
            default_probs=default_mat[0],
            eq1_probs=equation1_cyclic_debiased_content_probs(cyclic_mat),
            eq7_prior=equation7_prior_from_rollouts(cyclic_mat),
        )

    if n_excluded:
        logger.info(
            "Excluded %d question(s) present in both CSVs but failed or "
            "unparseable in at least one.", n_excluded,
        )

    n_choices_seen = {len(rec.letters) for rec in records.values()}
    if len(n_choices_seen) > 1:
        raise ValueError(
            f"Scored questions have mixed option counts {sorted(n_choices_seen)}. "
            "This script assumes a single option count across the scored set "
            "so calibration priors can be averaged with "
            "average_prior_probability_vectors() (mirrors PriDeRunner's own "
            "modal-k assumption in choicebench.methods.library.pride). "
            "Regenerate the CSVs from a modal-k-filtered benchmark."
        )

    return records


def sample_calibration_ids(full_ids: list[str], alpha: float, seed: int) -> list[str]:
    """K = floor(alpha * N) question_ids, uniform without replacement, seeded.

    ``full_ids`` must already be in a fixed (sorted) order so that a given
    seed always maps to the same question_ids regardless of CSV row order —
    the determinism guarantee this script relies on.

    Raises:
        ValueError: floor(alpha * N) rounds down to zero — PriDe needs at
            least one calibration question to estimate a prior.
    """
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be a finite fraction in (0, 1]; got {alpha!r}.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}.")
    n = len(full_ids)
    k = int(np.floor(alpha * n))
    if k == 0:
        raise ValueError(
            f"alpha={alpha} over N={n} scored questions yields K=0 calibration "
            "questions (floor(alpha * N) == 0). PriDe requires at least one "
            "calibration question to estimate a prior; use a larger alpha or "
            "a larger scored set."
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return sorted(full_ids[i] for i in idx)


def _predict_default(rec: QuestionRecord) -> str:
    return rec.letters[int(np.argmax(rec.default_probs))]


def _predict_eq1(rec: QuestionRecord) -> str:
    return rec.letters[int(np.argmax(rec.eq1_probs))]


def _predict_eq8(rec: QuestionRecord, global_prior: np.ndarray) -> str:
    deb = equation8_debiased_content_probs(rec.default_probs, global_prior)
    return rec.letters[int(np.argmax(deb))]


def _results_df(
        records: dict[str, QuestionRecord],
        ids: list[str],
        predictions: dict[str, str],
) -> pd.DataFrame:
    return pd.DataFrame({
        "question_id": ids,
        "correct_option": [records[qid].correct_option for qid in ids],
        "parsed_choice": [predictions[qid] for qid in ids],
    })


def _compute_metrics(
        records: dict[str, QuestionRecord],
        ids: list[str],
        predictions: dict[str, str],
) -> dict[str, float]:
    """Accuracy/recall_rstd/mad on (ids, predictions), via the built-in metrics."""
    df = _results_df(records, ids, predictions)
    out: dict[str, float] = {}
    out.update(BUILTIN_METRICS["accuracy"]().compute(df))
    out.update(BUILTIN_METRICS["recall_rstd"]().compute(df))
    out.update(BUILTIN_METRICS["mad"]().compute(df))
    return out


def compute_pride_cell(
        records: dict[str, QuestionRecord],
        full_ids: list[str],
        alpha: float,
        seed: int,
) -> tuple[dict[str, str], dict]:
    """One (alpha, seed) grid cell: calibrate, debias, score.

    Calibration questions (D_e) are answered via Eq. 1 (cyclic debiasing);
    every other question is answered via Eq. 8 (prior-debiased default
    distribution). Returns (predictions, summary) — predictions is exposed
    for tests; summary (metrics + N/K accounting) is what callers persist.
    """
    n = len(full_ids)
    calibration_ids = sample_calibration_ids(full_ids, alpha, seed)
    k = len(calibration_ids)
    calibration_set = set(calibration_ids)

    prior_vectors = [records[qid].eq7_prior for qid in calibration_ids]
    global_prior = average_prior_probability_vectors(prior_vectors)

    predictions: dict[str, str] = {}
    for qid in calibration_ids:
        predictions[qid] = _predict_eq1(records[qid])
    for qid in full_ids:
        if qid in calibration_set:
            continue
        predictions[qid] = _predict_eq8(records[qid], global_prior)

    metrics = _compute_metrics(records, full_ids, predictions)
    summary = {
        "alpha": alpha, "seed": seed, "n": n, "k": k,
        "calibration_question_ids": calibration_ids, **metrics,
    }
    return predictions, summary


def compute_default_row(records: dict[str, QuestionRecord], full_ids: list[str]) -> dict:
    """Default baseline over the full scored set: argmax(direct_logprob distribution)."""
    predictions = {qid: _predict_default(records[qid]) for qid in full_ids}
    return _compute_metrics(records, full_ids, predictions)


def compute_cyclic_row(records: dict[str, QuestionRecord], full_ids: list[str]) -> dict:
    """Cyclic-permutation baseline over the full scored set: Eq. 1 for every question."""
    predictions = {qid: _predict_eq1(records[qid]) for qid in full_ids}
    return _compute_metrics(records, full_ids, predictions)


def aggregate_alpha(cells: list[dict]) -> dict[str, float]:
    """Mean/std over seeds for one alpha. acc is rescaled to 0-100 to match rstd/mad."""
    accs = np.array([c["accuracy"] * 100 for c in cells])
    rstds = np.array([c["rstd"] for c in cells])
    mads = np.array([c["mad"] for c in cells])
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "rstd_mean": float(np.mean(rstds)),
        "rstd_std": float(np.std(rstds)),
        "mad_mean": float(np.mean(mads)),
        "mad_std": float(np.std(mads)),
    }


def build_comparison(
        default_row: dict,
        cyclic_row: dict,
        per_alpha: dict[float, dict],
        targets: dict,
) -> dict:
    """Our default/cyclic/pride_* numbers beside the fixture's, with deltas.

    The fixture (examples/pride_reproduction_targets.json) reports acc/rstd
    only (no mad), so deltas are only computed for those two.
    """
    fixture_targets = targets.get("targets", {})
    rows: dict[str, dict[str, float]] = {
        "default": {"acc": default_row["accuracy"] * 100, "rstd": default_row["rstd"], "mad": default_row["mad"]},
        "cyclic_perm": {"acc": cyclic_row["accuracy"] * 100, "rstd": cyclic_row["rstd"], "mad": cyclic_row["mad"]},
    }
    for alpha, key in _ALPHA_TARGET_KEYS.items():
        if alpha in per_alpha:
            agg = per_alpha[alpha]
            rows[key] = {"acc": agg["acc_mean"], "rstd": agg["rstd_mean"], "mad": agg["mad_mean"]}

    comparison: dict[str, dict] = {}
    for name, ours in rows.items():
        entry: dict = {"ours": ours}
        target = fixture_targets.get(name)
        if target:
            entry["target"] = target
            entry["delta"] = {
                "acc": ours["acc"] - target["acc"],
                "rstd": ours["rstd"] - target["rstd"],
            }
        comparison[name] = entry
    return comparison


def print_comparison_table(comparison: dict) -> None:
    """Compact aligned stdout summary, in the style of evaluate_run.py's log lines."""
    logger.info(
        "%-14s%8s%10s%8s%8s%10s%8s%8s",
        "row", "acc", "tgt_acc", "d_acc", "rstd", "tgt_rstd", "d_rstd", "mad",
    )
    for name in ("default", "cyclic_perm", "pride_5", "pride_40", "pride_80"):
        entry = comparison.get(name)
        if entry is None:
            continue
        ours = entry["ours"]
        target = entry.get("target")
        delta = entry.get("delta", {})
        tgt_acc = f"{target['acc']:.1f}" if target else "-"
        tgt_rstd = f"{target['rstd']:.1f}" if target else "-"
        d_acc = f"{delta['acc']:+.1f}" if target else "-"
        d_rstd = f"{delta['rstd']:+.1f}" if target else "-"
        logger.info(
            "%-14s%8.1f%10s%8s%8.1f%10s%8s%8.1f",
            name, ours["acc"], tgt_acc, d_acc, ours["rstd"], tgt_rstd, d_rstd, ours["mad"],
        )


def run(
        run_id: str,
        alphas: list[float],
        seeds: list[int],
        targets_path: Path,
        run_dir: Path | None = None,
) -> dict:
    """Recompute the full PriDe grid for one run. Returns the output JSON dict."""
    run_id = validate_run_id(run_id)
    if len(alphas) != len(set(alphas)):
        raise ValueError("alphas contains duplicate coordinates; every grid cell must be unique.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds contains duplicate coordinates; every grid cell must be unique.")
    run_dir = run_dir if run_dir is not None else (RUNS_DIR / run_id)
    direct_df = _load_method_frame(run_dir, "direct_logprob")
    cyclic_df = _load_method_frame(run_dir, "cyclic_logprob")
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest)
        source_identity = {
            "provenance_status": "verified-v2",
            "experiment_id": manifest["experiment_id"],
            "experiment_digest": manifest["experiment_digest"],
            "protocol_version": manifest["payload"]["protocol_version"],
            "direct_condition_ids": sorted(direct_df["condition_id"].astype(str).unique()),
            "cyclic_condition_ids": sorted(cyclic_df["condition_id"].astype(str).unique()),
        }
    else:
        source_identity = {"provenance_status": "legacy-unverified", "run_id": run_id}

    records = build_question_records(direct_df, cyclic_df)
    full_ids = sorted(records)
    n = len(full_ids)
    logger.info(
        "Scored subset: N=%d (intersection of direct_logprob/cyclic_logprob successes)", n,
    )
    if n == 0:
        raise ValueError(
            f"No question_id is valid in both direct_logprob and cyclic_logprob "
            f"results for run {run_id!r}; nothing to recompute."
        )

    default_row = compute_default_row(records, full_ids)
    cyclic_row = compute_cyclic_row(records, full_ids)

    per_cell: dict[str, dict] = {}
    per_alpha: dict[float, dict] = {}
    for alpha in alphas:
        cells = []
        for seed in seeds:
            _predictions, cell = compute_pride_cell(records, full_ids, alpha, seed)
            cell_identity = {
                "source": source_identity,
                "alpha": alpha, "seed": seed,
                "calibration_question_ids": cell["calibration_question_ids"],
            }
            cell["condition_id"] = short_id("pridecell", cell_identity)
            logger.info(
                "alpha=%.2f seed=%d  N=%d K=%d  acc=%.1f rstd=%.1f mad=%.1f",
                alpha, seed, cell["n"], cell["k"],
                cell["accuracy"] * 100, cell["rstd"], cell["mad"],
            )
            per_cell[cell["condition_id"]] = cell
            cells.append(cell)
        per_alpha[alpha] = aggregate_alpha(cells)

    targets = json.loads(targets_path.read_text())
    comparison = build_comparison(default_row, cyclic_row, per_alpha, targets)

    output = {
        "schema_version": "choicebench.pride-grid.v2",
        "run_id": run_id,
        "source": source_identity,
        "implementation_sha256": file_digest(Path(__file__)),
        "targets": {"path": targets_path.name, "sha256": file_digest(targets_path)},
        "n_scored": n,
        "alphas": alphas,
        "seeds": seeds,
        "cells": per_cell,
        "per_alpha": {str(alpha): agg for alpha, agg in per_alpha.items()},
        "comparison": comparison,
    }
    print_comparison_table(comparison)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline PriDe recompute from completed direct_logprob/"
                     "cyclic_logprob result CSVs — zero additional inference."
    )
    parser.add_argument("--run-id", required=True, type=str, help="Run ID (folder name under runs/).")
    parser.add_argument("--alphas", required=True, type=float, nargs="+", help="Calibration fractions, e.g. 0.05 0.40 0.80.")
    parser.add_argument("--seeds", required=True, type=int, nargs="+", help="Calibration-subset seeds, e.g. 0 1 2 3 4.")
    parser.add_argument("--targets", required=True, type=Path, help="Path to the official-numbers targets JSON fixture.")
    parser.add_argument("--out", required=True, type=Path, help="Path to write the PriDe grid JSON.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    output = run(args.run_id, args.alphas, args.seeds, args.targets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, output)
    logger.info("PriDe grid saved to %s", args.out)


if __name__ == "__main__":
    main()
