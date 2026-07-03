# src/choicebench/metrics/order_sensitivity.py

import json

import numpy as np
import pandas as pd

from choicebench.metrics.base import BaseMetric


class OrderSensitivity(BaseMetric):
    """Causal answer-order sensitivity, from per-rotation logprob data.

    Unlike MAD (marginal answer-letter skew vs. the gold distribution), this
    metric measures the effect of option *position*: for each cyclic rotation
    of a question's options, does the model's pick track the correct content
    or the printed position? It requires per-rotation logprob data, which only
    `cyclic_logprob` currently persists (its `option_distributions_json`
    column: shape (n_options, n_options), row k / column j = P(observed picks
    printed position j | rotation k)). Rows from other methods are ignored;
    if no row in the slice carries this data, both sub-metrics are NaN.

    Rotation convention (matching PermutationRunner / Eq. 1): under rotation
    k, the canonical content originally at position `c` is printed at
    position `(c - k) % n`.

    Reports:
      - order_rstd: Zheng et al.'s RStd (arXiv:2309.03882, §4.1) — the
        standard deviation, across printed positions, of recall conditioned
        on the correct content being printed at that position. High RStd
        means accuracy depends heavily on where the correct answer happens
        to land, independent of its content — the signature of positional
        bias.
      - order_flip_rate: fraction of questions where the model's content pick
        (translated back to canonical space) is not the same across every
        rotation — i.e. moving the same content to a different position
        changed the model's answer.
    """

    @property
    def name(self) -> str:
        return "order_sensitivity"

    def compute(self, results_df: pd.DataFrame) -> dict[str, float]:
        correct_by_position: dict[int, int] = {}
        total_by_position: dict[int, int] = {}
        n_flipped = 0
        n_questions = 0

        for _, row in results_df.iterrows():
            mat_json = row.get("option_distributions_json")
            correct_index = row.get("correct_index")
            if not isinstance(mat_json, str) or correct_index is None:
                continue
            if isinstance(correct_index, float) and np.isnan(correct_index):
                continue

            mat = np.array(json.loads(mat_json), dtype=np.float64)
            n = mat.shape[0]
            correct_index = int(correct_index)

            canonical_picks = []
            for k in range(n):
                predicted_position = int(np.argmax(mat[k]))
                correct_position = (correct_index - k) % n
                total_by_position[correct_position] = (
                    total_by_position.get(correct_position, 0) + 1
                )
                if predicted_position == correct_position:
                    correct_by_position[correct_position] = (
                        correct_by_position.get(correct_position, 0) + 1
                    )
                canonical_picks.append((predicted_position + k) % n)

            n_questions += 1
            if len(set(canonical_picks)) > 1:
                n_flipped += 1

        if n_questions == 0:
            return {"order_rstd": float("nan"), "order_flip_rate": float("nan")}

        recalls = [
            correct_by_position.get(pos, 0) / total
            for pos, total in total_by_position.items()
        ]
        order_rstd = float(np.std(recalls)) if len(recalls) > 1 else 0.0
        order_flip_rate = n_flipped / n_questions

        return {"order_rstd": order_rstd, "order_flip_rate": order_flip_rate}
