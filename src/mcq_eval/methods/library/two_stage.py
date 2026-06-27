# src/mcq_eval/methods/two_stage.py

"""
Method: Two-Stage Prompting (Free-Text then Match)
-----------------------------------------------------
Description: Stage one asks the model to answer the question in free text,
without showing it the lettered options at all — removing any positional
anchoring at the point the model actually reasons about the answer. Stage two
shows the model its own free-text answer alongside the four lettered options
and asks it to pick the matching letter. An optional fallback re-issues a
direct MCQ prompt if stage two's match is unparseable.
Reference: this project's primary research method — tests whether eliciting
a free-text answer before the model ever sees option letters reduces the
positional bias documented by Zheng et al., ICLR 2024 (arXiv:2309.03882).
Backend requirements: generate only (2 calls per question, or 3 with fallback)
Logprob support required: no
"""

from typing import Any

from mcq_eval.pipeline.prompt_builder import (
    build_direct_mcq_prompt,
    build_free_text_prompt,
    build_option_matching_prompt,
)
from mcq_eval.methods.base import ExperimentRunner


class TwoStageRunner(ExperimentRunner):
    """Runner for the two-stage prompting condition.

    Stage one elicits a free-text answer without exposing options.
    Stage two asks the model to match that free-text answer to one
    of the four canonical options. The final letter from stage two
    is parsed and scored.

    The intermediate free-text response is preserved in the result
    row for downstream answer-matching evaluation.
    """

    def __init__(self, *args, fallback_on_parse_failure: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fallback_on_parse_failure = fallback_on_parse_failure

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute one question through the two-stage pipeline.

        Args:
            question_row: Normalized question record.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary containing trace, model output,
            parse, and score fields, plus the intermediate free-text
            response.
        """
        # Stage 1: free-text response
        free_text_prompt = build_free_text_prompt(
            template=self._prompts["free_text"],
            question=question_row["question_text"],
        )
        free_text_response = self._call_backend_generate(free_text_prompt)

        # If stage 1 fails, return early
        if not free_text_response.is_success():
            return self._build_result_row(
                question_row=question_row,
                prompt=free_text_prompt,
                sample_index=sample_index,
                model_response=free_text_response,
                parsed_result=None,
                score_result=None,
            )

        free_text_answer = free_text_response.raw_text

        # Stage 2: option matching using the free-text answer
        matching_prompt = build_option_matching_prompt(
            template=self._prompts["option_matching"],
            question=question_row["question_text"],
            free_text=free_text_answer,
            option_a=question_row["choice_a"],
            option_b=question_row["choice_b"],
            option_c=question_row["choice_c"],
            option_d=question_row["choice_d"],
        )
        matching_response = self._call_backend_generate(matching_prompt)

        # Parse and score the stage 2 response
        parsed_result = None
        score_result = None

        if matching_response.is_success():
            parsed_result, score_result = self._parse_and_score(
                raw_text=matching_response.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )

        # Fallback: if matching was unparseable, re-issue the direct MCQ prompt.
        # Goes through the normal backend.generate() path — caching, if any,
        # is the backend's responsibility (e.g. inside APIBackend), not
        # something this runner controls directly.
        fallback_used = False
        if (
            self._fallback_on_parse_failure
            and matching_response.is_success()
            and (parsed_result is None or parsed_result.final_choice is None)
        ):
            fallback_prompt = build_direct_mcq_prompt(
                template=self._prompts["direct_mcq"],
                question=question_row["question_text"],
                option_a=question_row["choice_a"],
                option_b=question_row["choice_b"],
                option_c=question_row["choice_c"],
                option_d=question_row["choice_d"],
            )
            fallback_response = self._call_backend_generate(fallback_prompt)
            if fallback_response.is_success():
                parsed_result, score_result = self._parse_and_score(
                    raw_text=fallback_response.raw_text,
                    correct_option=question_row["correct_option"],
                    options=self._build_options(question_row),
                )
                fallback_used = True

        result = self._build_result_row(
            question_row=question_row,
            prompt=matching_prompt,
            sample_index=sample_index,
            model_response=matching_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )

        # Preserve the intermediate free-text response for answer matching
        result["free_text_prompt"] = free_text_prompt
        result["free_text_response"] = free_text_answer
        result["free_text_latency"] = free_text_response.latency_seconds
        result["fallback_used"] = fallback_used

        return result
