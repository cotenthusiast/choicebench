# src/mcq_eval/methods/templates/base_method.py
#
# Template for implementing a new MCQ evaluation method.
# Copy this file into src/mcq_eval/methods/<your_method_name>.py (or any
# importable module), fill in the TODOs, then register it in your config YAML:
#
#   methods:
#     - name: my_package.methods.my_method:MyRunner   # importlib path
#
# Built-in methods can instead be added to METHOD_REGISTRY in
# scripts/run_experiment.py, but the importlib path works without
# touching any framework file.
#
# NOTE: use self.backend.generate(prompt) -> str for plain text generation.
# Use self.backend.score_options(prompt, options) -> list[float] only for
# logprob-based methods (e.g. PriDe). Always check
# self.backend.supports_logprobs first — API backends raise NotImplementedError.

from __future__ import annotations

from typing import Any

from mcq_eval.methods.base import ExperimentRunner

"""
Method: <Your Method Name>
---------------------------
Description: <one paragraph — what does this method do, and why?>
Reference: <paper title and citation, or "n/a" if not from a paper>
Backend requirements: generate only | generate + score_options
Logprob support required: yes | no
"""


class YourMethodRunner(ExperimentRunner):
    """TODO: rename this class to describe your method, e.g. MyDebiasRunner."""

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute one question through this experimental condition.

        Args:
            question_row: Normalized question record with keys:
                question_id, subject, question_text,
                choice_a, choice_b, choice_c, choice_d, correct_option.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary. Use self._build_result_row() to produce
            the standard schema — do not construct this dict by hand.
        """
        # --- Build the prompt for this question ---
        # TODO: replace with your method's prompt-building logic
        prompt = question_row["question_text"]  # placeholder

        # --- Call the backend ---
        model_response = self._call_backend_generate(prompt)

        parsed_result = None
        score_result = None

        if model_response.is_success():
            # --- Example: parse the raw text and score against gold answer ---
            parsed_result, score_result = self._parse_and_score(
                raw_text=model_response.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )

            # --- Example: logprob-based scoring (if your method needs it) ---
            # if not self.backend.supports_logprobs:
            #     raise NotImplementedError(
            #         f"{self.__class__.__name__} requires score_options() support."
            #     )
            # scores = self.backend.score_options(prompt, list(options.values()))
            # final_choice = list(options.keys())[scores.index(max(scores))]

        return self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            model_response=model_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )
