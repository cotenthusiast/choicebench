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

from choicebench.parsing.types import ParseResult, PARSE_OK, PARSE_MISSING
from choicebench.pipeline.prompt_builder import (
    Rotation,
    build_permuted_prompt,
    build_rotations,
)
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
        canonical_letters = list(canonical_options.keys())
        rotations = self._generate_rotations(canonical_options)

        # Build prompts for each rotation
        prompts = [
            self._build_permuted_prompt(question_row, rot.mapping, self._prompts["direct_mcq"])
            for rot in rotations
        ]

        # One backend call per rotation (sequential — backend calls are
        # compute-bound local forward passes or already-blocking API calls).
        responses = [
            self._call_backend_generate(prompt)
            for prompt in prompts
        ]

        # Parse each response and map back via the rotation's positional
        # bijection (text-independent — F1 fix).
        canonical_choices: list[str | None] = []
        for response, rotation in zip(responses, rotations):
            if response.is_success():
                parsed = self._parse(response.raw_text, rotation.mapping)
                if parsed.final_choice is not None:
                    canonical_choices.append(
                        self._canonical_letter(
                            parsed.final_choice, rotation, canonical_letters
                        )
                    )
                else:
                    canonical_choices.append(None)
            else:
                canonical_choices.append(None)

        # Majority vote across canonical answers
        voted_letter = self._majority_vote(canonical_choices, canonical_options)

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

        # Use the first permutation's trace for the result row. answer_status
        # is derived by _build_result_row from voted_parse (the *voted*
        # outcome), not responses[0] (FSF-5 / PF-3): a row must not be marked
        # answer-failure while carrying a valid voted answer just because the
        # first rotation's call failed. transport_status, in contrast, does
        # reflect responses[0] — it's a transport-only signal.
        row = self._build_result_row(
            question_row=question_row,
            prompt=prompts[0],
            sample_index=sample_index,
            model_response=responses[0],
            parsed_result=voted_parse,
            score_result=score_result,
        )
        return row

    async def run_many_async(self, question_rows: Sequence[Any]) -> list[dict]:
        """Async batch execution for PermutationRunner.

        Fans all permutation prompts for all questions out in one
        generate_batch() call, then reassembles per-question results and
        runs majority vote — identical logic to run_one() but fully batched.
        """
        rows = question_rows.to_dict(orient="records")
        canonical_options_per_q = [self._build_options(row) for row in rows]
        rotations_per_q = [
            self._generate_rotations(opts) for opts in canonical_options_per_q
        ]

        # Flatten: build one prompt per (question, rotation) pair.
        all_prompts: list[str] = []
        # prompt_map[i] = (question_index, rotation_index)
        prompt_map: list[tuple[int, int]] = []
        for q_idx, (row, rots) in enumerate(zip(rows, rotations_per_q)):
            for p_idx, rot in enumerate(rots):
                all_prompts.append(
                    self._build_permuted_prompt(row, rot.mapping, self._prompts["direct_mcq"])
                )
                prompt_map.append((q_idx, p_idx))

        all_responses = await self.backend.generate_batch(all_prompts)

        # Group responses back per question and run majority vote.
        results = []
        for q_idx, (row, rots) in enumerate(zip(rows, rotations_per_q)):
            canonical_options = canonical_options_per_q[q_idx]
            canonical_letters = list(canonical_options.keys())
            q_flat_indices = [i for i, (qi, _) in enumerate(prompt_map) if qi == q_idx]

            canonical_choices: list[str | None] = []
            for flat_i in q_flat_indices:
                p_idx = prompt_map[flat_i][1]
                response = all_responses[flat_i]
                rot = rots[p_idx]
                if response.is_success():
                    parsed = self._parse(response.raw_text, rot.mapping)
                    if parsed.final_choice is not None:
                        canonical_choices.append(
                            self._canonical_letter(
                                parsed.final_choice, rot, canonical_letters
                            )
                        )
                    else:
                        canonical_choices.append(None)
                else:
                    canonical_choices.append(None)

            voted_letter = self._majority_vote(canonical_choices, canonical_options)
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
            results.append(result_row)

        return results

    @staticmethod
    def _generate_rotations(
            options: dict[str, str],
    ) -> list[Rotation]:
        """Thin delegate to pipeline.prompt_builder.build_rotations().

        Each Rotation carries the rendered mapping plus its positional
        bijection, produced by the same operation (F1 fix).
        """
        return build_rotations(options)

    @staticmethod
    def _build_permuted_prompt(
            question_row: Any,
            permuted_options: dict[str, str],
            template: str,
    ) -> str:
        """Thin delegate to pipeline.prompt_builder.build_permuted_prompt()."""
        return build_permuted_prompt(question_row, permuted_options, template)

    @staticmethod
    def _canonical_letter(
            parsed_letter: str,
            rotation: Rotation,
            canonical_letters: list[str],
    ) -> str:
        """Positional inverse: display slot -> its single canonical position.

        Text-independent by construction (F1 fix): duplicate option texts can
        no longer redirect a parsed answer to an earlier twin.
        """
        display_index = canonical_letters.index(parsed_letter)
        return canonical_letters[rotation.slot_to_canonical[display_index]]

    @staticmethod
    def _majority_vote(
            choices: list[str | None],
            canonical_options: dict[str, str],
    ) -> str | None:
        """Determine the final answer by majority vote.

        In the case of a tie, the canonically-earliest letter among the tied
        candidates is used as the tiebreaker — not the first-encountered vote,
        which is ordered by rotation/permutation index and would reintroduce
        the positional correlation cyclic permutation exists to cancel.

        Args:
            choices: List of canonical letters from each permutation,
                with None for any that failed to parse.
            canonical_options: Canonical letter-to-text mapping for this
                question, in canonical letter order (A, B, C, ...); used to
                resolve ties by canonical index rather than vote order.

        Returns:
            The most frequent letter, or None if no valid votes exist.
        """
        cleaned = [x for x in choices if x is not None]
        if not cleaned:
            return None

        counts = collections.Counter(cleaned)
        top = counts.most_common(2)
        if len(top) == 1 or top[0][1] != top[1][1]:
            return top[0][0]

        max_count = top[0][1]
        tied_letters = {letter for letter, count in counts.items() if count == max_count}
        for letter in canonical_options:
            if letter in tied_letters:
                return letter
        return top[0][0]
