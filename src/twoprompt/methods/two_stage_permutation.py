# src/twoprompt/methods/two_stage_permutation.py
# Migrated from src/twoprompt/runners/two_stage_permutation.py (Session 3) —
# already backend-based and synchronous as of that file's Session 1/2 wiring,
# so no logic changed in this move. The only change is the PermutationRunner
# import, repointed to its sibling methods/permutation.py instead of the
# soon-to-be-deprecated runners/permutation.py, so methods/ does not depend
# on runners/ for anything but the shared ExperimentRunner base. See
# runners/two_stage_permutation.py for the deprecation notice pointing here.

"""
Method: Two-Stage + Cyclic Permutation
-----------------------------------------
Description: Combines the two methods above. Stage one elicits one free-text
answer per question, exactly as in two_stage. Stage two reuses that same
free-text answer across every cyclic permutation of the option ordering,
issuing N option-matching prompts, un-permuting each parsed letter, and
taking a majority vote. Tests whether the two mitigations compose rather
than substitute for one another.
Reference: combination of this project's two-stage method with the
cyclic-permutation mitigation from Zheng et al., ICLR 2024 (arXiv:2309.03882).
Backend requirements: generate only (1 + N calls per question, or +1 with
fallback)
Logprob support required: no
"""

from typing import Any

from twoprompt.parsing.types import ParseResult, PARSE_OK, PARSE_MISSING
from twoprompt.pipeline.prompt_builder import (
    build_direct_mcq_prompt,
    build_free_text_prompt,
    build_option_matching_prompt,
)
from twoprompt.runners.base import ExperimentRunner
from twoprompt.methods.permutation import PermutationRunner


class TwoStagePermutationRunner(ExperimentRunner):
    """Runner for the combined two-stage + cyclic permutation condition.

    Stage one elicits a free-text answer without exposing options.
    Stage two generates N cyclic permutations of the option order,
    builds N option-matching prompts using the same free-text answer,
    makes N backend calls, un-permutes each parsed letter, and
    determines the final answer by majority vote.

    Reuses permutation helpers from PermutationRunner.
    """

    def __init__(self, *args, fallback_on_parse_failure: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fallback_on_parse_failure = fallback_on_parse_failure

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute one question through two-stage + permutation.

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
        free_text_request = self._build_model_request(
            question_row, free_text_prompt, sample_index,
        )
        free_text_response = self._call_backend_generate(free_text_request, free_text_prompt)

        # If stage 1 fails, return early
        if not free_text_response.is_success():
            return self._build_result_row(
                question_row=question_row,
                prompt=free_text_prompt,
                model_request=free_text_request,
                model_response=free_text_response,
                parsed_result=None,
                score_result=None,
            )

        free_text_answer = free_text_response.raw_text

        # Stage 2: permuted option matching
        canonical_options = self._build_options(question_row)
        permutations = PermutationRunner._generate_permutations(canonical_options)

        # Build option-matching prompts for each permutation
        prompts = [
            build_option_matching_prompt(
                template=self._prompts["option_matching"],
                question=question_row["question_text"],
                free_text=free_text_answer,
                option_a=perm["A"],
                option_b=perm["B"],
                option_c=perm["C"],
                option_d=perm["D"],
            )
            for perm in permutations
        ]
        requests = [
            self._build_model_request(question_row, prompt, sample_index)
            for prompt in prompts
        ]

        # One backend call per permutation (sequential — backend calls are
        # compute-bound local forward passes or already-blocking API calls).
        responses = [
            self._call_backend_generate(req, prompt)
            for req, prompt in zip(requests, prompts)
        ]

        # Parse each response and un-permute back to canonical ordering
        canonical_choices: list[str | None] = []
        for response, permutation in zip(responses, permutations):
            if response.is_success():
                parsed = self._parse(response.raw_text, permutation)
                if parsed.final_choice is not None:
                    canonical_choices.append(
                        PermutationRunner._unpermute_choice(
                            parsed.final_choice, permutation, canonical_options
                        )
                    )
                else:
                    canonical_choices.append(None)
            else:
                canonical_choices.append(None)

        # Majority vote across canonical answers
        voted_letter = PermutationRunner._majority_vote(canonical_choices)

        # Build a synthetic ParseResult from the voted answer
        voted_parse = ParseResult(
            final_choice=voted_letter,
            status=PARSE_OK if voted_letter else PARSE_MISSING,
            raw_text=None,
            normalized_text="",
            reason="majority_vote",
        )

        # Score the voted answer
        score_result = None
        if voted_letter:
            score_result = self._score(voted_parse, question_row["correct_option"])

        # Fallback: if majority vote produced no winner, re-issue the direct MCQ prompt.
        # Goes through the normal backend.generate() path — caching, if any,
        # is the backend's responsibility (e.g. inside APIBackend), not
        # something this runner controls directly.
        fallback_used = False
        if self._fallback_on_parse_failure and voted_letter is None:
            fallback_prompt = build_direct_mcq_prompt(
                template=self._prompts["direct_mcq"],
                question=question_row["question_text"],
                option_a=question_row["choice_a"],
                option_b=question_row["choice_b"],
                option_c=question_row["choice_c"],
                option_d=question_row["choice_d"],
            )
            fallback_request = self._build_model_request(
                question_row, fallback_prompt, sample_index
            )
            fallback_response = self._call_backend_generate(fallback_request, fallback_prompt)
            if fallback_response.is_success():
                fallback_parse, fallback_score = self._parse_and_score(
                    raw_text=fallback_response.raw_text,
                    correct_option=question_row["correct_option"],
                    options=canonical_options,
                )
                if fallback_parse.final_choice is not None:
                    voted_parse = fallback_parse
                    score_result = fallback_score
                    fallback_used = True

        # Use the first permutation's trace for the result row
        result = self._build_result_row(
            question_row=question_row,
            prompt=prompts[0],
            model_request=requests[0],
            model_response=responses[0],
            parsed_result=voted_parse,
            score_result=score_result,
        )

        # Preserve the intermediate free-text response for answer matching
        result["free_text_prompt"] = free_text_prompt
        result["free_text_response"] = free_text_answer
        result["free_text_latency"] = free_text_response.latency_seconds
        result["fallback_used"] = fallback_used

        return result
