# src/twoprompt/metrics/mad.py
# Ported from scripts/evaluate_run.py's compute_positional_bias() and
# _bootstrap_ci_mean_abs_deviation() — same algorithm, repackaged as a
# BaseMetric. See base.py's REVIEW NEEDED note: this operates on the actual
# correct_option/parsed_choice columns, not an option_position column (which
# doesn't exist in this codebase's result rows).

import numpy as np
import pandas as pd

from twoprompt.metrics.base import BaseMetric

_OPTIONS = ["A", "B", "C", "D"]
_N_BOOTSTRAP = 10_000
_BOOTSTRAP_SEED = 42
_OPT_INDEX = {opt: i for i, opt in enumerate(_OPTIONS)}


class MAD(BaseMetric):
    """Mean Absolute Deviation — positional bias from the ground-truth answer distribution.

    For each option letter, compares the percentage of questions where the
    model's prediction was that letter against the percentage of questions
    where that letter was actually correct. MAD is the average absolute gap
    across all four letters. Higher MAD = more positional bias (the model's
    answer distribution deviates more from the ground-truth distribution).

    A 0.0 MAD would mean the model picks each letter exactly as often as
    that letter is actually correct — not the same as high accuracy, but
    the absence of a systematic positional preference.
    """

    @property
    def name(self) -> str:
        return "mad"

    def compute(self, results_df: pd.DataFrame) -> dict[str, float]:
        scored = results_df[results_df["parsed_choice"].notna()]
        total_scored = len(scored)
        if total_scored == 0:
            return {"mad": float("nan"), "mad_std": float("nan")}

        gt_total = len(results_df)
        gt_counts = results_df["correct_option"].value_counts()
        pred_counts = scored["parsed_choice"].value_counts()

        deviations = []
        for opt in _OPTIONS:
            gt_pct = gt_counts.get(opt, 0) / gt_total * 100
            pred_pct = pred_counts.get(opt, 0) / total_scored * 100
            deviations.append(abs(pred_pct - gt_pct))

        mad = sum(deviations) / len(deviations)
        mad_std = self._bootstrap_std(results_df)

        return {"mad": mad, "mad_std": mad_std}

    @staticmethod
    def _bootstrap_std(group: pd.DataFrame) -> float:
        """Bootstrap standard deviation of MAD, resampling questions (rows).

        Vectorized over all resamples — see evaluate_run.py's
        _bootstrap_ci_mean_abs_deviation for the same approach reporting CI
        bounds instead of std.
        """
        n = len(group)
        rng = np.random.default_rng(_BOOTSTRAP_SEED)

        def _enc(series: pd.Series) -> np.ndarray:
            return np.array(
                [_OPT_INDEX.get(v, -1) if isinstance(v, str) else -1 for v in series.values],
                dtype=np.int8,
            )

        gt_enc = _enc(group["correct_option"])
        pred_enc = _enc(group["parsed_choice"])

        idx = rng.integers(0, n, size=(_N_BOOTSTRAP, n))
        gt_boot = gt_enc[idx]
        pred_boot = pred_enc[idx]

        scored_boot = pred_boot >= 0
        total_scored = scored_boot.sum(axis=1).astype(float)

        total_abs_dev = np.zeros(_N_BOOTSTRAP)
        for k in range(len(_OPTIONS)):
            gt_pct = (gt_boot == k).sum(axis=1) / n * 100.0
            pred_count = ((pred_boot == k) & scored_boot).sum(axis=1)
            pred_pct = np.where(total_scored > 0, pred_count / total_scored * 100.0, 0.0)
            total_abs_dev += np.abs(pred_pct - gt_pct)

        stats = np.where(total_scored > 0, total_abs_dev / len(_OPTIONS), np.nan)
        valid = stats[~np.isnan(stats)]
        if len(valid) == 0:
            return float("nan")
        return float(np.std(valid))
