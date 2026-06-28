# src/choicebench/metrics/templates/base_metric_template.py
#
# Template for implementing a new evaluation metric.
#
# NAMING CONVENTION
#   File name:  my_metric_name.py  (snake_case)
#   Class name: MyMetricNameMetric  (CamelCase + "Metric" suffix)
#   YAML key:   "my_metric_name"  (must match the name property below)
#
# REGISTRATION
#   Built-in (lives inside this repo):
#     Add to BUILTIN_METRICS in src/choicebench/metrics/__init__.py.
#
#   External (lives in your own package):
#     Use "module.path:ClassName" in the YAML config — no framework files needed.
#     Example:  metrics:
#                 - my_package.metrics.my_metric:MyMetricMetric
#
# WHAT A METRIC RECEIVES
#   compute() receives a results DataFrame already filtered to ONE
#   (method_name, model_name) pair — one row per question.
#   Guaranteed columns (from _build_result_row in methods/base.py):
#     question_id     — unique question hash
#     correct_option  — ground-truth letter ("A"/"B"/"C"/"D")
#     parsed_choice   — model's parsed answer letter, or missing/NaN
#     method_name     — name of the evaluation method
#     model_name      — name of the model that produced these results
#     is_correct      — True/False/None (None when unscored, e.g. parse/backend failure)
#     score_status    — "score_correct" / "score_incorrect" / "score_unscorable"
#                       (constants in scoring/types.py)
#     subject         — MMLU subject string (or equivalent)
#     provider        — provider name string
#
# WHAT A METRIC MUST NOT DO
#   - Import runners, backends, or clients
#   - Call any model or API
#   - Read files or environment variables
#   - Mutate the input DataFrame

from __future__ import annotations

import pandas as pd

from choicebench.metrics.base import BaseMetric


class YourMetricMetric(BaseMetric):
    # Rename this class: e.g. PositionalBiasMetric.

    @property
    def name(self) -> str:
        # Must match the key used in the YAML config's metrics: list.
        # Returned metric keys appear in the JSON report and logged summary.
        return "your_metric"

    def compute(self, results_df: pd.DataFrame) -> dict[str, float]:
        """Compute the metric over one (method, model) slice of results.

        Args:
            results_df: One row per question, pre-filtered to a single
                (method_name, model_name) pair.

        Returns:
            dict of {metric_key: float}. Return a single key for simple
            metrics, or multiple keys if your metric naturally produces
            several related numbers. All keys appear in the JSON report and
            logged summary.

        Examples of multi-key returns:
            # Simple:
            {"accuracy": 0.72}

            # With uncertainty:
            {"accuracy": 0.72, "accuracy_std": 0.03}

            # Subject-level breakdown (one key per subject):
            {f"accuracy_{s}": v for s, v in by_subject.items()}
        """
        total = len(results_df)
        if total == 0:
            return {self.name: 0.0}

        # Example: count correct answers (is_correct may be None on failure)
        n_correct = results_df["is_correct"].sum()
        value = n_correct / total

        return {self.name: float(value)}
