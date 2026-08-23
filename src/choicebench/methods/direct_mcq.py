# src/choicebench/methods/library/direct_mcq.py

"""
Method: Direct MCQ (Baseline)
------------------------------
Description: Presents the model with the question and all four lettered options
in a single prompt and parses the letter it selects. This is the conventional,
unmitigated way of evaluating an LLM on multiple-choice questions — every other
method in this package exists to measure or reduce the positional bias this
baseline is susceptible to.
Reference: n/a (standard MCQ evaluation baseline; contrasted against by
Zheng et al., ICLR 2024, "Large Language Models Are Not Robust Multiple Choice
Selectors", arXiv:2309.03882).
Backend requirements: generate only
Logprob support required: no
"""

from typing import Any

from choicebench.pipeline.prompt_builder import build_direct_mcq_prompt
from choicebench.methods.base import ExperimentRunner


class DirectMCQRunner(ExperimentRunner):
    """Runner for the direct MCQ baseline condition.

    Presents the model with a standard multiple-choice question and
    expects a single letter response. One prompt, one backend call per
    question.
    """

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute one question through the direct MCQ baseline.

        Args:
            question_row: Normalized question record.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary containing trace, model output,
            parse, and score fields.
        """
        prompt = self._build_prompt(question_row)
        model_response = self._call_backend_generate(prompt)

        parsed_result = None
        score_result = None

        if model_response.is_success():
            parsed_result, score_result = self._parse_and_score(
                raw_text=model_response.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )

        return self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            model_response=model_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )

    def _build_prompt(self, question_row: Any) -> str:
        """Build a direct multiple-choice prompt from a question row."""
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=self._build_options(question_row),
            subject=question_row["subject"],
        )

    def _build_batch_prompt(self, question_row: Any) -> str:
        """Prompt builder for the async batch path — delegates to _build_prompt."""
        return self._build_prompt(question_row)
