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

import json
from typing import Any, Sequence

from choicebench.parsing.types import ParseResult, PARSE_OK, PARSE_MISSING
from choicebench.pipeline.prompt_builder import (
    Rotation,
    build_permuted_prompt,
    build_rotations,
)
from choicebench.methods.base import ExperimentRunner
from choicebench.scoring.tiebreak import majority_vote_with_tiebreak


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
            sample_index: Caller-assigned index (see ExperimentRunner.run_one).

        Returns:
            Flat result dictionary containing trace, model output,
            parse, and score fields.
        """
        rotations = self._generate_rotations(self._build_options(question_row))
        prompts = [
            self._build_permuted_prompt(question_row, rot.mapping, self._prompts["direct_mcq"])
            for rot in rotations
        ]

        # One backend call per rotation (sequential — backend calls are
        # compute-bound local forward passes or already-blocking API calls).
        responses = [self._call_backend_generate(prompt) for prompt in prompts]

        return self._assemble_row(question_row, rotations, prompts, responses, sample_index)

    async def run_many_async(self, question_rows: Sequence[Any]) -> list[dict]:
        """Async batch execution for PermutationRunner.

        Fans all permutation prompts for all questions out in one
        generate_batch() call, then reassembles per-question results and
        runs majority vote through the same _assemble_row() as run_one().
        """
        rows = question_rows.to_dict(orient="records")
        rotations_per_q = [self._generate_rotations(self._build_options(row)) for row in rows]

        # Flatten: one prompt per (question, rotation), question by question, so a question's
        # prompts and responses are the contiguous slice [offset, offset + len(rotations)).
        all_prompts: list[str] = [
            self._build_permuted_prompt(row, rot.mapping, self._prompts["direct_mcq"])
            for row, rots in zip(rows, rotations_per_q)
            for rot in rots
        ]
        all_responses = await self.backend.generate_batch(all_prompts)

        results = []
        offset = 0
        for q_idx, (row, rots) in enumerate(zip(rows, rotations_per_q)):
            stop = offset + len(rots)
            results.append(self._assemble_row(
                row, rots, all_prompts[offset:stop], all_responses[offset:stop], q_idx,
            ))
            offset = stop
        return results

    def _assemble_row(
            self,
            question_row: Any,
            rotations: list[Rotation],
            prompts: list[str],
            responses: list[Any],
            sample_index: int,
    ) -> dict:
        """Decode every rotation's reply, majority-vote, score, and build the result row.

        Shared by run_one() (sequential calls) and run_many_async() (one batched call) so the two
        paths cannot drift apart. ``prompts`` and ``responses`` are in rotation order.
        """
        canonical_options = self._build_options(question_row)
        canonical_letters = list(canonical_options.keys())

        canonical_choices = [
            self._decode_rotation_response(response, rotation, canonical_letters)
            for response, rotation in zip(responses, rotations)
        ]

        # Majority vote across canonical answers
        voted_letter = self._majority_vote(
            canonical_choices, canonical_options,
            label_to_source_index=self._build_label_to_source_index(question_row),
            seed=self.seed, benchmark_id=self.benchmark_name,
            question_id=question_row["question_id"], method_name=self.method_name,
        )

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
        # Per-rotation canonical choices (rotation order), not just the
        # majority-voted winner -- required for baseline/cyclic flip-rate,
        # which needs to see every rotation's own answer to detect a flip.
        row["per_rotation_choices_json"] = json.dumps(canonical_choices)
        return row

    def _decode_rotation_response(
            self,
            response: Any,
            rotation: Rotation,
            canonical_letters: list[str],
    ) -> str | None:
        """One rotation's reply -> the canonical letter it picked, or None.

        None means no usable answer: the call failed or the reply did not parse to a letter.
        Parsing is against the rotation's *displayed* options; the inverse is positional.
        """
        if not response.is_success():
            return None
        parsed = self._parse(response.raw_text, rotation.mapping)
        if parsed.final_choice is None:
            return None
        return self._canonical_letter(parsed.final_choice, rotation, canonical_letters)

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
            *,
            label_to_source_index: dict[str, int],
            seed: int,
            benchmark_id: str,
            question_id: str,
            method_name: str,
    ) -> str | None:
        """Determine the final answer by majority vote.

        Ties are broken via the shared canonical tie-break utility
        (choicebench.scoring.tiebreak.resolve_tie) over each tied letter's
        stable ``source_index`` identity — never the first-encountered vote
        (which is ordered by rotation/permutation index and would
        reintroduce the positional correlation cyclic permutation exists to
        cancel) and never the display letter itself.

        Args:
            choices: List of canonical letters from each permutation,
                with None for any that failed to parse.
            canonical_options: Canonical letter-to-text mapping for this
                question (unused by the tie-break itself; kept for callers
                that pass it positionally).
            label_to_source_index: This question's letter->source_index map.
            seed: Experiment seed, part of the tie-break key.
            benchmark_id: Benchmark name, part of the tie-break key.
            question_id: This question's canonical ID, part of the tie-break key.
            method_name: Registry method name, part of the tie-break key.

        Returns:
            The most frequent letter, or None if no valid votes exist.
        """
        return majority_vote_with_tiebreak(
            choices, label_to_source_index=label_to_source_index, seed=seed,
            benchmark_id=benchmark_id, question_id=question_id, method_name=method_name,
        )
