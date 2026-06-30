# src/choicebench/methods/library/permutation.py

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
from typing import Any, Sequence

from choicebench.clients.types import FAILURE_STATUS, SUCCESS_STATUS
from choicebench.parsing.types import ParseResult, PARSE_OK, PARSE_MISSING
from choicebench.pipeline.prompt_builder import build_direct_mcq_prompt
from choicebench.methods.base import ExperimentRunner


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

        # Build prompts for each permutation
        prompts = [
            self._build_permuted_prompt(question_row, perm, self._prompts["direct_mcq"])
            for perm in permutations
        ]

        # One backend call per permutation (sequential — backend calls are
        # compute-bound local forward passes or already-blocking API calls).
        responses = [
            self._call_backend_generate(prompt)
            for prompt in prompts
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

        # Use the first permutation's trace for the result row, but report
        # model_status by the *voted* outcome, not responses[0] (FSF-5 / PF-3):
        # a row must not be marked "failure" while carrying a valid voted answer
        # just because the first rotation's call failed.
        row = self._build_result_row(
            question_row=question_row,
            prompt=prompts[0],
            sample_index=sample_index,
            model_response=responses[0],
            parsed_result=voted_parse,
            score_result=score_result,
        )
        row["model_status"] = SUCCESS_STATUS if voted_letter else FAILURE_STATUS
        return row

    async def run_many_async(self, question_rows: Sequence[Any]) -> list[dict]:
        """Async batch execution for PermutationRunner.

        Fans all permutation prompts for all questions out in one
        generate_batch() call, then reassembles per-question results and
        runs majority vote — identical logic to run_one() but fully batched.
        """
        rows = question_rows.to_dict(orient="records")
        canonical_options_per_q = [self._build_options(row) for row in rows]
        permutations_per_q = [
            self._generate_permutations(opts) for opts in canonical_options_per_q
        ]

        # Flatten: build one prompt per (question, permutation) pair.
        all_prompts: list[str] = []
        # prompt_map[i] = (question_index, permutation_index)
        prompt_map: list[tuple[int, int]] = []
        for q_idx, (row, perms) in enumerate(zip(rows, permutations_per_q)):
            for p_idx, perm in enumerate(perms):
                all_prompts.append(
                    self._build_permuted_prompt(row, perm, self._prompts["direct_mcq"])
                )
                prompt_map.append((q_idx, p_idx))

        all_responses = await self.backend.generate_batch(all_prompts)

        # Group responses back per question and run majority vote.
        results = []
        for q_idx, (row, perms) in enumerate(zip(rows, permutations_per_q)):
            canonical_options = canonical_options_per_q[q_idx]
            q_flat_indices = [i for i, (qi, _) in enumerate(prompt_map) if qi == q_idx]

            canonical_choices: list[str | None] = []
            for flat_i in q_flat_indices:
                p_idx = prompt_map[flat_i][1]
                response = all_responses[flat_i]
                perm = perms[p_idx]
                if response.is_success():
                    parsed = self._parse(response.raw_text, perm)
                    if parsed.final_choice is not None:
                        canonical_choices.append(
                            self._unpermute_choice(
                                parsed.final_choice, perm, canonical_options
                            )
                        )
                    else:
                        canonical_choices.append(None)
                else:
                    canonical_choices.append(None)

            voted_letter = self._majority_vote(canonical_choices)
            voted_parse = ParseResult(
                final_choice=voted_letter,
                status=PARSE_OK if voted_letter else PARSE_MISSING,
                raw_text=None,
                normalized_text="",
                reason="majority_vote",
            )
            score_result = None
            if voted_letter:
                score_result = self._score(voted_parse, row["correct_option"])

            first_flat = q_flat_indices[0]
            result_row = self._build_result_row(
                question_row=row,
                prompt=all_prompts[first_flat],
                sample_index=q_idx,
                model_response=all_responses[first_flat],
                parsed_result=voted_parse,
                score_result=score_result,
            )
            result_row["model_status"] = SUCCESS_STATUS if voted_letter else FAILURE_STATUS
            results.append(result_row)

        return results

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
        return build_direct_mcq_prompt(
            template=template,
            question=question_row["question_text"],
            options=permuted_options,
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
