# src/choicebench/metrics/recall_rstd.py

import numpy as np
import pandas as pd

from choicebench.metrics.base import BaseMetric, _label_set

_N_BOOTSTRAP = 10_000
_BOOTSTRAP_SEED = 42


class RecallRStd(BaseMetric):
    """Selection-bias RStd — dispersion of per-letter recall (Zheng et al., ICLR
    2024, "Large Language Models Are Not Robust Multiple Choice Selectors",
    arXiv:2309.03882, §2.2).

    For each option letter, recall is accuracy restricted to the (scored)
    questions whose *gold* answer sits at that letter — i.e. for letter L,
    recall_L = P(predicted == L | gold == L, scored). RStd is the standard
    deviation of these per-letter recalls across the benchmark's label space,
    reported on a 0-100 scale (paper Table 3 values, e.g. 17.4, are
    percentages) — so both the recall_* values and rstd here are percentages,
    not fractions.

    This is a *marginal* selection-bias indicator: it asks whether the model
    is systematically worse at recognizing the correct answer when it happens
    to sit behind one particular letter, aggregated over the whole label
    distribution. It is distinct from `order_sensitivity`'s `order_rstd`,
    which measures the same "dispersion of recall by position" idea but from
    per-permutation logprob data — i.e. it is causal (the same question's
    content is moved and its position tracked), where this metric is
    correlational (it only sees each question's one naturally-occurring gold
    letter, not what would happen if that content were printed elsewhere).
    Mirrors the MAD-vs-order_sensitivity split in mad.py.

    Same scored-subset semantics as MAD: only rows with a non-null
    parsed_choice contribute to any recall_L (an unparseable row is excluded
    from both the numerator and denominator, not counted as incorrect).

    A letter with zero scored questions whose gold answer sits there has an
    undefined recall (NaN) and is excluded from the RStd calculation itself;
    its n_per_letter_* count is still reported so a thin/zero subset for that
    letter is visible downstream rather than silently dropped.

    Std convention: population standard deviation (ddof=0, no Bessel
    correction) — the paper does not specify ddof, but the reference
    implementation (github.com/chujiezheng/LLM-MCQ-Bias,
    code/debias_pride.py, ``recall_stds.append(np.std(recalls))``) calls
    ``np.std`` with no ddof argument, i.e. population std. This also matches
    this repo's own convention in order_sensitivity.py and mad.py's bootstrap.
    """

    @property
    def name(self) -> str:
        return "recall_rstd"

    def compute(self, results_df: pd.DataFrame) -> dict[str, float]:
        options = _label_set(results_df)
        if not options:
            return {"rstd": float("nan"), "rstd_std": float("nan")}

        scored = results_df[results_df["parsed_choice"].notna()]

        out: dict[str, float] = {}
        recalls: list[float] = []
        for opt in options:
            gold_rows = scored[scored["correct_option"] == opt]
            n = len(gold_rows)
            out[f"n_per_letter_{opt}"] = float(n)
            if n == 0:
                out[f"recall_{opt}"] = float("nan")
                continue
            recall_pct = float((gold_rows["parsed_choice"] == opt).sum()) / n * 100
            out[f"recall_{opt}"] = recall_pct
            recalls.append(recall_pct)

        out["rstd"] = float(np.std(recalls)) if recalls else float("nan")
        out["rstd_std"] = self._bootstrap_std(scored, options)
        return out

    @staticmethod
    def _bootstrap_std(scored: pd.DataFrame, options: list[str]) -> float:
        """Bootstrap standard deviation of RStd, resampling questions
        (rows) with replacement -- mirrors mad.py's MAD._bootstrap_std
        exactly (same _N_BOOTSTRAP/_BOOTSTRAP_SEED convention, same
        resample-rows/recompute-per-letter-statistic/std-across-resamples
        shape), applied to per-letter recall instead of per-letter
        prediction-vs-gold deviation.

        Args:
            scored: already restricted to rows with a non-null
                parsed_choice (same scored-subset convention as compute()).
        """
        n = len(scored)
        if n == 0 or not options:
            return float("nan")
        rng = np.random.default_rng(_BOOTSTRAP_SEED)
        opt_index = {opt: i for i, opt in enumerate(options)}

        def _enc(series: pd.Series) -> np.ndarray:
            return np.array(
                [opt_index.get(v, -1) if isinstance(v, str) else -1 for v in series.values],
                dtype=np.int16,
            )

        gold_enc = _enc(scored["correct_option"])
        pred_enc = _enc(scored["parsed_choice"])

        idx = rng.integers(0, n, size=(_N_BOOTSTRAP, n))
        gold_boot = gold_enc[idx]
        pred_boot = pred_enc[idx]

        recalls_per_resample = np.full((_N_BOOTSTRAP, len(options)), np.nan)
        for k in range(len(options)):
            is_gold_k = gold_boot == k
            n_gold_k = is_gold_k.sum(axis=1).astype(float)
            n_correct_k = (is_gold_k & (pred_boot == k)).sum(axis=1).astype(float)
            recall_k = np.divide(
                n_correct_k * 100.0, n_gold_k,
                out=np.full(_N_BOOTSTRAP, np.nan), where=n_gold_k > 0,
            )
            recalls_per_resample[:, k] = recall_k

        # A resample with every letter's recall undefined (e.g. n=1 row
        # resampled such that no gold letter has >0 occurrences for any
        # letter it could be -- practically only possible for a
        # pathologically tiny input) yields all-NaN for that resample row;
        # np.nanstd on an all-NaN slice already returns NaN with a
        # (suppressed) warning, consistent with compute()'s own NaN
        # handling for an empty per-letter recall list.
        with np.errstate(invalid="ignore"):
            rstds = np.nanstd(recalls_per_resample, axis=1)
        valid_rstds = rstds[~np.isnan(rstds)]
        if len(valid_rstds) == 0:
            return float("nan")
        return float(np.std(valid_rstds))
