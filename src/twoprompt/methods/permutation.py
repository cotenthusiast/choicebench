# src/twoprompt/methods/permutation.py
# Migrated from src/twoprompt/runners/permutation.py (Session 3) — already
# backend-based and synchronous as of that file's Session 1/2 wiring, so no
# logic changed in this move. See runners/permutation.py for the deprecation
# notice pointing back here.

"""
Method: Cyclic Permutation (Majority Vote)
-------------------------------------------
Description: Generates every cyclic rotation of the option ordering for a
question (4 rotations for a 4-option question), issues one prompt per
rotation, maps each parsed letter back to the canonical ordering, and takes
the majority vote across rotations as the final answer. Mitigates positional
bias by averaging a single question's answer over every position the correct
option could have been placed in.
Reference: Zheng et al., ICLR 2024, "Large Language Models Are Not Robust
Multiple Choice Selectors" (arXiv:2309.03882) — cyclic-permutation mitigation
(related to their Eq. 1).
Backend requirements: generate only
Logprob support required: no
"""

import collections
from typing import Any

from twoprompt.parsing.types import ParseResult, PARSE_OK, PARSE_MISSING
from twoprompt.pipeline.prompt_builder import build_direct_mcq_prompt
from twoprompt.runners.base import ExperimentRunner


class PermutationRunner(ExperimentRunner):
    """Runner for the cyclic permutation condition.

    Generates N cyclic permutations of the option order for each question,
    makes N backend calls, un-permutes each parsed answer back to
    canonical ordering, and determines the final answer by majority vote.
    """

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute one question through all cyclic permutations.

        Args:
            question_row: Normalized question record.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary containing trace, model output,
            parse, and score fields.
        """
        canonical_options = self._build_options(question_row)
        permutations = self._generate_permutations(canonical_options)

        # Build prompts and requests for each permutation
        prompts = [
            self._build_permuted_prompt(question_row, perm, self._prompts["direct_mcq"])
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
                        self._unpermute_choice(
                            parsed.final_choice, permutation, canonical_options
                        )
                    )
                else:
                    canonical_choices.append(None)
            else:
                canonical_choices.append(None)

        # Majority vote across canonical answers
        voted_letter = self._majority_vote(canonical_choices)

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

        # Use the first permutation's trace for the result row
        return self._build_result_row(
            question_row=question_row,
            prompt=prompts[0],
            model_request=requests[0],
            model_response=responses[0],
            parsed_result=voted_parse,
            score_result=score_result,
        )

    @staticmethod
    def _generate_permutations(
            options: dict[str, str],
    ) -> list[dict[str, str]]:
        """Generate cyclic permutations of the canonical option ordering.

        Each permutation maps canonical letters (A, B, C, D) to cyclically
        shifted answer texts. The number of permutations equals the number
        of options.

        Args:
            options: Canonical letter-to-text mapping.

        Returns:
            List of permuted option mappings.
        """
        keys = list(options.keys())
        values = list(options.values())
        return [
            dict(zip(keys, values[i:] + values[:i]))
            for i in range(len(options))
        ]

    @staticmethod
    def _build_permuted_prompt(
            question_row: Any,
            permuted_options: dict[str, str],
            template: str,
    ) -> str:
        """Build a direct MCQ prompt using a permuted option ordering.

        Args:
            question_row: Normalized question record.
            permuted_options: Permuted letter-to-text mapping.
            template: Raw direct_mcq template string.

        Returns:
            Fully formatted prompt string with permuted options.
        """
        vals = list(permuted_options.values())
        return build_direct_mcq_prompt(
            template=template,
            question=question_row["question_text"],
            option_a=vals[0],
            option_b=vals[1],
            option_c=vals[2],
            option_d=vals[3],
        )

    @staticmethod
    def _unpermute_choice(
            parsed_letter: str,
            permuted_options: dict[str, str],
            canonical_options: dict[str, str],
    ) -> str | None:
        """Map a parsed letter from permuted ordering back to canonical.

        Args:
            parsed_letter: Letter the model selected (in permuted space).
            permuted_options: The permuted mapping used for that call.
            canonical_options: The original canonical mapping.

        Returns:
            Canonical letter corresponding to the selected answer text,
            or None if no match is found.
        """
        selected_text = permuted_options[parsed_letter]
        for key, value in canonical_options.items():
            if value == selected_text:
                return key
        return None

    @staticmethod
    def _majority_vote(choices: list[str | None]) -> str | None:
        """Determine the final answer by majority vote.

        In the case of a tie, the first valid vote is used as the
        tiebreaker, which corresponds to the canonical option ordering.

        Args:
            choices: List of canonical letters from each permutation,
                with None for any that failed to parse.

        Returns:
            The most frequent letter, or None if no valid votes exist.
        """
        cleaned = [x for x in choices if x is not None]
        if not cleaned:
            return None

        top = collections.Counter(cleaned).most_common(2)
        if len(top) == 1 or top[0][1] != top[1][1]:
            return top[0][0]
        return cleaned[0]
