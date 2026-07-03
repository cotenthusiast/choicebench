# src/choicebench/metrics/accuracy.py

import pandas as pd
from scipy.stats import beta

from choicebench.metrics.base import BaseMetric

_CI_ALPHA = 0.05  # 95% confidence


def _clopper_pearson(k: int, n: int, alpha: float = _CI_ALPHA) -> tuple[float, float]:
    """Exact (Clopper-Pearson) confidence interval for k successes in n trials.

    Standard beta-distribution formulation: lower/upper bounds are the
    alpha/2 and 1-alpha/2 quantiles of Beta(k, n-k+1) and Beta(k+1, n-k)
    respectively, with the k=0 / k=n edge cases pinned to 0.0 / 1.0 since
    Beta(0, .) and Beta(., 0) are undefined.
    """
    if n == 0:
        return float("nan"), float("nan")
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


class Accuracy(BaseMetric):
    """Fraction of rows where parsed_choice == correct_option.

    Reports two sub-metrics, each with a 95% Clopper-Pearson (exact binomial)
    confidence interval:
      - accuracy: end-to-end — correct / total rows (an unparseable or
        failed row counts as incorrect, not excluded).
      - accuracy_conditional: correct / scored rows only (rows with no
        parsed_choice at all are excluded from both numerator and
        denominator).
    """

    @property
    def name(self) -> str:
        return "accuracy"

    def compute(self, results_df: pd.DataFrame) -> dict[str, float]:
        total = len(results_df)
        if total == 0:
            return {
                "accuracy": 0.0,
                "accuracy_ci_low": float("nan"),
                "accuracy_ci_high": float("nan"),
                "accuracy_conditional": 0.0,
                "accuracy_conditional_ci_low": float("nan"),
                "accuracy_conditional_ci_high": float("nan"),
            }

        is_correct = results_df["parsed_choice"] == results_df["correct_option"]
        correct = int(is_correct.sum())

        scored = int(results_df["parsed_choice"].notna().sum())

        accuracy_ci_low, accuracy_ci_high = _clopper_pearson(correct, total)
        if scored > 0:
            accuracy_conditional_ci_low, accuracy_conditional_ci_high = _clopper_pearson(
                correct, scored
            )
        else:
            accuracy_conditional_ci_low = float("nan")
            accuracy_conditional_ci_high = float("nan")

        return {
            "accuracy": correct / total,
            "accuracy_ci_low": accuracy_ci_low,
            "accuracy_ci_high": accuracy_ci_high,
            "accuracy_conditional": correct / scored if scored > 0 else 0.0,
            "accuracy_conditional_ci_low": accuracy_conditional_ci_low,
            "accuracy_conditional_ci_high": accuracy_conditional_ci_high,
        }
