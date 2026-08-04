# src/choicebench/metrics/base.py

from abc import ABC, abstractmethod

import pandas as pd


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


class BaseMetric(ABC):
    """
    Interface for all evaluation metrics.

    To add a custom metric:
    1. Subclass BaseMetric
    2. Implement name (property) and compute(results_df) -> dict
    3. Register it in your config.yaml under metrics:
    4. The eval runner will call it automatically

    compute() receives a DataFrame of results for ONE (method, model) slice
    (one row per question) and returns a dict of {metric_name: value} — can
    return multiple sub-metrics if needed.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in config.yaml and output tables."""
        ...

    @abstractmethod
    def compute(self, results_df: pd.DataFrame) -> dict[str, float]:
        """
        Compute metric(s) from a results DataFrame.

        Args:
            results_df: DataFrame with at least columns:
                - question_id: str
                - correct_option: str   (ground truth label, e.g. "A")
                - parsed_choice: str | NaN (model prediction, e.g. "B";
                                          NaN if unparseable/unscored)
                - method_name: str      (method name)
                - model_name: str       (model name)
                Already filtered to one (method_name, model_name) pair —
                the caller is responsible for grouping; compute() does not
                group internally.

        Returns:
            dict of {sub_metric_name: float}
            e.g. {"accuracy": 0.73} or {"mad": 0.12, "mad_std": 0.03}
        """
        ...
