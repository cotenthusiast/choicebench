# src/choicebench/metrics/order_sensitivity.py

import json

import numpy as np
import pandas as pd

from choicebench.metrics.base import BaseMetric


class OrderSensitivity(BaseMetric):
    """Causal answer-order sensitivity, from per-rotation trace data.

    Unlike MAD (marginal answer-letter skew vs. the gold distribution), this
    metric measures the effect of option *position*: for each cyclic rotation
    of a question's options, does the model's pick track the correct content
    or the printed position? It requires per-rotation data, in either of two
    forms a row may carry:

      - `option_distributions_json` (`cyclic_logprob` only): shape
        (n_options, n_options), row k / column j = P(observed picks printed
        position j | rotation k). Full probability mass.
      - `per_rotation_choices_json` (`cyclic_permutation`, `two_stage`'s
        rotation rerun, `text_extraction`'s rotation rerun, and the other
        rotation-based flip-rate methods): a plain list of canonical letters
        (or null for a rotation that failed to produce an answer), one per
        rotation. Lighter -- sufficient for both sub-metrics below, since
        RStd only needs the argmax per rotation, not the full distribution.

    A row carrying neither is ignored; if no row in the slice carries either,
    both sub-metrics are NaN. A row with `per_rotation_choices_json` whose
    every entry is null (every rotation failed) is also excluded entirely --
    there is no signal to contribute.

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

    @staticmethod
    def _picks_from_matrix(
            mat_json: str, correct_index: int,
            correct_by_position: dict[int, int], total_by_position: dict[int, int],
    ) -> list[int]:
        mat = np.array(json.loads(mat_json), dtype=np.float64)
        n = mat.shape[0]
        canonical_picks = []
        for k in range(n):
            predicted_position = int(np.argmax(mat[k]))
            correct_position = (correct_index - k) % n
            total_by_position[correct_position] = total_by_position.get(correct_position, 0) + 1
            if predicted_position == correct_position:
                correct_by_position[correct_position] = correct_by_position.get(correct_position, 0) + 1
            canonical_picks.append((predicted_position + k) % n)
        return canonical_picks

    @staticmethod
    def _picks_from_letters(
            per_rotation_json: str, correct_index: int,
            correct_by_position: dict[int, int], total_by_position: dict[int, int],
    ) -> list[int]:
        letters = json.loads(per_rotation_json)
        n = len(letters)
        canonical_picks = []
        for k, letter in enumerate(letters):
            if letter is None:
                continue  # this rotation's call failed -- no data point at all
            correct_position = (correct_index - k) % n
            total_by_position[correct_position] = total_by_position.get(correct_position, 0) + 1
            pred_canonical_index = ord(letter) - ord("A")
            predicted_position = (pred_canonical_index - k) % n
            if predicted_position == correct_position:
                correct_by_position[correct_position] = correct_by_position.get(correct_position, 0) + 1
            canonical_picks.append(pred_canonical_index)
        return canonical_picks

    def compute(self, results_df: pd.DataFrame) -> dict[str, float]:
        correct_by_position: dict[int, int] = {}
        total_by_position: dict[int, int] = {}
        n_flipped = 0
        n_questions = 0

        for _, row in results_df.iterrows():
            correct_index = row.get("correct_index")
            if correct_index is None:
                continue
            if isinstance(correct_index, float) and np.isnan(correct_index):
                continue
            correct_index = int(correct_index)

            mat_json = row.get("option_distributions_json")
            per_rotation_json = row.get("per_rotation_choices_json")

            if isinstance(mat_json, str):
                canonical_picks = self._picks_from_matrix(
                    mat_json, correct_index, correct_by_position, total_by_position,
                )
            elif isinstance(per_rotation_json, str):
                canonical_picks = self._picks_from_letters(
                    per_rotation_json, correct_index, correct_by_position, total_by_position,
                )
                if not canonical_picks:
                    continue  # every rotation failed -- no signal at all
            else:
                continue

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
