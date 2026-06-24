# src/twoprompt/metrics/templates/base_metric_template.py
#
# Template for implementing a new metric. Copy this file into
# src/twoprompt/metrics/<your_metric_name>.py and fill in the TODOs.
#
# NOTE: metrics must not import runners, methods, clients, or backends.
# A metric only ever sees a results DataFrame — it never calls a model,
# never knows what backend produced the results, and never touches
# checkpoint/config plumbing.

from __future__ import annotations

import pandas as pd

from twoprompt.metrics.base import BaseMetric


class YourMetric(BaseMetric):
    """TODO: rename this class and describe what it measures.

    compute() receives a results DataFrame already filtered to one
    (method_name, model_name) pair — one row per question. Available
    columns include (at minimum): question_id, correct_option,
    parsed_choice, method_name, model_name, is_correct, score_status.
    See src/twoprompt/runners/base.py's _build_result_row() for the full
    column list any given run may have.
    """

    @property
    def name(self) -> str:
        # TODO: short identifier used in config.yaml's metrics: list and in
        # output tables, e.g. "my_metric".
        return "your_metric"

    def compute(self, results_df: pd.DataFrame) -> dict[str, float]:
        """TODO: implement your metric's computation here.

        Returns:
            dict of {sub_metric_name: float}. Return more than one key if
            your metric naturally produces several related numbers (e.g.
            a point estimate plus a standard error), the way MAD returns
            both "mad" and "mad_std".
        """
        # --- Example skeleton ---
        # total = len(results_df)
        # if total == 0:
        #     return {self.name: 0.0}
        # value = ...  # compute from results_df["correct_option"],
        #               # results_df["parsed_choice"], etc.
        # return {self.name: value}

        raise NotImplementedError("TODO: implement compute().")
