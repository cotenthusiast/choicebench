# src/choicebench/metrics/accuracy.py

import pandas as pd

from choicebench.metrics.base import BaseMetric


class Accuracy(BaseMetric):
    """Fraction of rows where parsed_choice == correct_option.

    Reports two sub-metrics, matching the two accuracy definitions already
    used by scripts/evaluate_run.py's compute_accuracy():
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
            return {"accuracy": 0.0, "accuracy_conditional": 0.0}

        is_correct = results_df["parsed_choice"] == results_df["correct_option"]
        correct = int(is_correct.sum())

        scored = int(results_df["parsed_choice"].notna().sum())

        return {
            "accuracy": correct / total,
            "accuracy_conditional": correct / scored if scored > 0 else 0.0,
        }
