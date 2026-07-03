# src/choicebench/metrics/mad.py

import numpy as np
import pandas as pd

from choicebench.metrics.base import BaseMetric

_N_BOOTSTRAP = 10_000
_BOOTSTRAP_SEED = 42


def _label_set(results_df: pd.DataFrame) -> list[str]:
    """Option labels present in a results frame (union of gold and predicted).

    The label space is derived from the data rather than a fixed A-D set, so a
    benchmark with any number of options (e.g. MMLU-Pro's up to 10, A-J) is
    scored over its full label space instead of silently dropping E+.
    """
    labels: set[str] = set()
    for col in ("correct_option", "parsed_choice"):
        if col in results_df.columns:
            for v in results_df[col].dropna():
                if isinstance(v, str) and v:
                    labels.add(v)
    return sorted(labels)


class MAD(BaseMetric):
    """Mean Absolute Deviation — marginal answer-letter skew vs. the gold distribution.

    For each option letter, compares the percentage of (scored) questions
    where the model's prediction was that letter against the percentage of
    those same questions where that letter was actually correct. MAD is the
    average absolute gap across every letter in the benchmark's label space.
    Higher MAD = the model's answer-letter distribution deviates more from
    the gold-answer distribution.

    Both percentages are computed over the *same* subset (rows with a
    non-null parsed_choice) and the *same* denominator, so MAD is not
    confounded by the parse-failure rate — a benchmark/method with more
    unparseable rows does not mechanically inflate or deflate MAD.

    The label space is derived from the data (union of gold and predicted
    labels), so benchmarks with more than four options are handled correctly.

    A 0.0 MAD would mean the model picks each letter exactly as often as
    that letter is actually correct among scored questions — not the same
    as high accuracy, but the absence of a systematic marginal preference
    for any one letter.

    Note: MAD is a marginal-distribution indicator, not a measure of causal
    answer-order bias (i.e. it does not tell you whether swapping an option's
    position changes the model's answer for a given question). For that,
    see the `order_sensitivity` metric, which uses per-permutation data.
    """

    @property
    def name(self) -> str:
        return "mad"

    def compute(self, results_df: pd.DataFrame) -> dict[str, float]:
        scored = results_df[results_df["parsed_choice"].notna()]
        total_scored = len(scored)
        if total_scored == 0:
            return {"mad": float("nan"), "mad_std": float("nan")}

        options = _label_set(results_df)
        if not options:
            return {"mad": float("nan"), "mad_std": float("nan")}

        gt_counts = scored["correct_option"].value_counts()
        pred_counts = scored["parsed_choice"].value_counts()

        deviations = []
        for opt in options:
            gt_pct = gt_counts.get(opt, 0) / total_scored * 100
            pred_pct = pred_counts.get(opt, 0) / total_scored * 100
            deviations.append(abs(pred_pct - gt_pct))

        mad = sum(deviations) / len(deviations)
        mad_std = self._bootstrap_std(results_df, options)

        return {"mad": mad, "mad_std": mad_std}

    @staticmethod
    def _bootstrap_std(group: pd.DataFrame, options: list[str]) -> float:
        """Bootstrap standard deviation of MAD, resampling questions (rows).

        Vectorized over all resamples: each of _N_BOOTSTRAP resamples draws n
        rows with replacement, recomputes the per-letter deviations over the
        supplied label space, and the std across resamples is returned as the
        MAD uncertainty estimate. Both gt_pct and pred_pct are restricted to
        the scored subset of each resample (mirroring compute()), so a
        resample's parse-failure rate doesn't confound the deviation.
        """
        n = len(group)
        rng = np.random.default_rng(_BOOTSTRAP_SEED)
        opt_index = {opt: i for i, opt in enumerate(options)}

        def _enc(series: pd.Series) -> np.ndarray:
            return np.array(
                [opt_index.get(v, -1) if isinstance(v, str) else -1 for v in series.values],
                dtype=np.int16,
            )

        gt_enc = _enc(group["correct_option"])
        pred_enc = _enc(group["parsed_choice"])

        idx = rng.integers(0, n, size=(_N_BOOTSTRAP, n))
        gt_boot = gt_enc[idx]
        pred_boot = pred_enc[idx]

        scored_boot = pred_boot >= 0
        total_scored = scored_boot.sum(axis=1).astype(float)

        total_abs_dev = np.zeros(_N_BOOTSTRAP)
        for k in range(len(options)):
            gt_count = ((gt_boot == k) & scored_boot).sum(axis=1)
            gt_pct = np.where(total_scored > 0, gt_count / total_scored * 100.0, 0.0)
            pred_count = ((pred_boot == k) & scored_boot).sum(axis=1)
            pred_pct = np.where(total_scored > 0, pred_count / total_scored * 100.0, 0.0)
            total_abs_dev += np.abs(pred_pct - gt_pct)

        stats = np.where(total_scored > 0, total_abs_dev / len(options), np.nan)
        valid = stats[~np.isnan(stats)]
        if len(valid) == 0:
            return float("nan")
        return float(np.std(valid))
