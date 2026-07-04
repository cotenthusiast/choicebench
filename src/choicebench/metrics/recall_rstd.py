# src/choicebench/metrics/recall_rstd.py

import numpy as np
import pandas as pd

from choicebench.metrics.base import BaseMetric


def _label_set(results_df: pd.DataFrame) -> list[str]:
    """Option labels present in a results frame (union of gold and predicted).

    Mirrors mad.py's helper of the same name: the label space is derived from
    the data rather than a fixed A-D set, so 5+ option benchmarks are scored
    over their full label space.
    """
    labels: set[str] = set()
    for col in ("correct_option", "parsed_choice"):
        if col in results_df.columns:
            for v in results_df[col].dropna():
                if isinstance(v, str) and v:
                    labels.add(v)
    return sorted(labels)


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
            return {"rstd": float("nan")}

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
        return out
