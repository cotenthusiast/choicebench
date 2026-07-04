# src/choicebench/methods/library/shuffled_baseline.py

"""
Method: Shuffled Baseline
--------------------------
Description: Randomly permutes the option ordering for a question once
(seeded by the run seed and question_id, so the permutation is fixed across
repeated samples of the same question but reproducible across runs), then
issues a single direct-ask prompt against the shuffled order. The parsed
letter is mapped back from shuffled position to the canonical option
ordering before scoring. Unlike cyclic_permutation (N calls, majority vote
across every rotation), this makes exactly one backend call per question —
it isolates what a single random reordering does to the model's answer,
rather than averaging it out.
Reference: n/a (reference/demo method for the extensibility walkthrough;
mechanically the single-draw case of the reordering Zheng et al. exhaustively
enumerate in cyclic permutation, ICLR 2024, arXiv:2309.03882).
Backend requirements: generate only
Logprob support required: no
Calls per question: 1
"""

from __future__ import annotations

import random
from typing import Any

from choicebench.methods.base import ExperimentRunner
from choicebench.parsing.types import ParseResult
from choicebench.pipeline.prompt_builder import build_direct_mcq_prompt


class ShuffledBaselineRunner(ExperimentRunner):
    """Runner for the shuffled-baseline condition.

    Same single-call shape as DirectMCQRunner, but the option order is
    shuffled once per question before rendering. The shuffle is seeded from
    (self.seed, question_id) so run_one() is deterministic and reproducible;
    the parsed letter is looked up in the shuffle's label map to recover the
    canonical letter before _score() runs.
    """

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        canonical_options = self._build_options(question_row)
        rng = random.Random(f"{self.seed}:{question_row['question_id']}")
        shuffled_options, label_map = self._shuffle_options(canonical_options, rng)

        prompt = build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=shuffled_options,
        )
        model_response = self._call_backend_generate(prompt)

        parsed_result = None
        score_result = None

        if model_response.is_success():
            # Parse against shuffled_options — the letters and text the model
            # actually saw. Parsing against canonical_options here would pair
            # the wrong text with each letter wherever the shuffle moved it.
            shuffled_parse = self._parse(model_response.raw_text, shuffled_options)
            canonical_choice = (
                label_map[shuffled_parse.final_choice]
                if shuffled_parse.final_choice is not None
                else None
            )
            parsed_result = ParseResult(
                final_choice=canonical_choice,
                status=shuffled_parse.status,
                raw_text=shuffled_parse.raw_text,
                normalized_text=shuffled_parse.normalized_text,
                reason=shuffled_parse.reason,
            )
            score_result = self._score(parsed_result, question_row["correct_option"])

        return self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            model_response=model_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )

    @staticmethod
    def _shuffle_options(
            canonical_options: dict[str, str],
            rng: random.Random,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Permute option text across the canonical letter slots.

        Args:
            canonical_options: Canonical letter-to-text mapping, in order.
            rng: Seeded Random instance controlling the permutation.

        Returns:
            Tuple of (shuffled_options, label_map):
              - shuffled_options: the same letters, with text permuted — this
                is what gets rendered into the prompt.
              - label_map: shuffled letter -> canonical letter, so a letter
                parsed from the shuffled prompt can be mapped back to score.
        """
        labels = list(canonical_options.keys())
        order = list(range(len(labels)))
        rng.shuffle(order)
        shuffled_options = {
            labels[i]: canonical_options[labels[order[i]]] for i in range(len(labels))
        }
        label_map = {labels[i]: labels[order[i]] for i in range(len(labels))}
        return shuffled_options, label_map
