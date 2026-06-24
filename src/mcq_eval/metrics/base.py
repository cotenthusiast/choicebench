# src/mcq_eval/metrics/base.py
#
# REVIEW NEEDED: the originally-specified docstring for compute() described
# columns correct_answer/predicted_answer/method/model/option_position. That
# doesn't match this codebase's actual result-row schema (see
# _build_result_row in runners/base.py / methods/*.py and
# scripts/evaluate_run.py's load_run()/compute_accuracy()): the real columns
# are correct_option, parsed_choice, method_name, model_name — and there is
# no option_position column at all (per-permutation position data is not
# persisted past majority voting). I adapted the docstring to the columns
# that actually exist rather than document a contract no real data satisfies.

from abc import ABC, abstractmethod

import pandas as pd


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
