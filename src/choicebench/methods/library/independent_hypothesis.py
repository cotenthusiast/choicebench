# src/choicebench/methods/library/independent_hypothesis.py

"""
Method: Independent Hypothesis (IHS)
-------------------------------------------------------------------------
Paper-specific (eacl-2026-revision): ported natively into ChoiceBench as
the reference implementation for this revision's reasoning-treatment
comparison. Historically run via two external repos (~/Projects/
two-stage-prompting for the 4 API models, ~/Projects/model-generalization
for the 2 local models) -- this port replaces both for the revised matrix.

Description: Evaluates each real option as an isolated hypothesis rather
than presenting all options together. The model receives the question and
exactly one candidate option framed as a hypothesis, and returns a 0-100
confidence score for that hypothesis alone. One backend call per real
option (never N-1 padded to a fixed 4 -- an ARC-Challenge question with
only 3 real options makes exactly 3 calls, never a phantom "D" slot). No
option ever appears alongside another in the same prompt. The final
prediction is the option with the highest confidence score; ties are
broken via the shared canonical tie-break utility over each option's
stable source_index.
Reference: paper §4.7 / Appendix D (arXiv:2608.11947).
Backend requirements: generate only
Logprob support required: no
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from choicebench.methods.base import ExperimentRunner
from choicebench.parsing.types import PARSE_OK, PARSE_MISSING, ParseResult
from choicebench.pipeline.prompt_builder import build_independent_hypothesis_prompt
from choicebench.scoring.tiebreak import resolve_tie

# Matches "<score>X</score>" where X is an int or float, case-insensitive.
# Last occurrence wins -- consistent with the rest of the codebase's parser,
# which prefers a model's final restated answer over an earlier draft.
_SCORE_PATTERN = re.compile(r"<score>\s*(-?\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE)


def _parse_confidence_score(raw_text: str | None) -> tuple[float, bool]:
    """Extract a confidence score from raw model output via regex.

    Returns:
        (score, parse_ok). On parse failure, score is 0.0 and parse_ok is
        False -- but the caller still includes this 0.0 in the argmax; a
        parse failure is not the same as "this option should be excluded."
    """
    if not raw_text:
        return 0.0, False
    matches = _SCORE_PATTERN.findall(raw_text)
    if not matches:
        return 0.0, False
    try:
        return float(matches[-1]), True
    except ValueError:
        return 0.0, False


def _argmax_with_tiebreak(
        scores: dict[str, float],
        label_to_source_index: dict[str, int],
        *,
        seed: int,
        benchmark_id: str,
        question_id: str,
        method_name: str,
) -> str:
    """Pick the highest-scoring option letter, breaking exact ties via the
    shared canonical tie-break utility over source_index (never letter)."""
    best_score = max(scores.values())
    tied_letters = [letter for letter, s in scores.items() if s == best_score]
    if len(tied_letters) == 1:
        return tied_letters[0]
    tied_ids = [label_to_source_index[letter] for letter in tied_letters]
    winning_id = resolve_tie(
        seed=seed, benchmark_id=benchmark_id, question_id=question_id,
        method_name=method_name, tied_canonical_ids=tied_ids,
    )
    id_to_label = {v: k for k, v in label_to_source_index.items()}
    return id_to_label[winning_id]


class IndependentHypothesisRunner(ExperimentRunner):
    """Runner for the independent-hypothesis condition.

    Every real option for a question is evaluated independently: one
    backend call per option, never a call for a missing option, never two
    options in the same prompt. All raw responses and per-option scores are
    preserved in the result row so a different tie-break or parse rule
    could be applied post-hoc without rerunning inference.
    """

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        options = self._build_options(question_row)
        label_to_source_index = self._build_label_to_source_index(question_row)
        letters = list(options.keys())

        prompts = [
            build_independent_hypothesis_prompt(
                template=self._prompts["independent_hypothesis"],
                question=question_row["question_text"],
                option_text=options[letter],
            )
            for letter in letters
        ]
        responses = [self._call_backend_generate(prompt) for prompt in prompts]

        return self._assemble_result_row(
            question_row=question_row,
            letters=letters,
            prompts=prompts,
            responses=responses,
            label_to_source_index=label_to_source_index,
        )

    async def run_many_async(self, question_rows: Sequence[Any]) -> list[dict]:
        """Batched path: every (question, option) prompt in one generate_batch() call."""
        rows = question_rows.to_dict(orient="records")
        options_per_q = [self._build_options(row) for row in rows]
        letters_per_q = [list(opts.keys()) for opts in options_per_q]

        all_prompts: list[str] = []
        prompt_map: list[tuple[int, int]] = []  # flat index -> (question_index, option_index)
        for q_idx, (row, letters, options) in enumerate(zip(rows, letters_per_q, options_per_q)):
            for o_idx, letter in enumerate(letters):
                all_prompts.append(
                    build_independent_hypothesis_prompt(
                        template=self._prompts["independent_hypothesis"],
                        question=row["question_text"],
                        option_text=options[letter],
                    )
                )
                prompt_map.append((q_idx, o_idx))

        all_responses = await self.backend.generate_batch(all_prompts)

        results = []
        for q_idx, (row, letters) in enumerate(zip(rows, letters_per_q)):
            q_flat_indices = [i for i, (qi, _) in enumerate(prompt_map) if qi == q_idx]
            q_prompts = [all_prompts[i] for i in q_flat_indices]
            q_responses = [all_responses[i] for i in q_flat_indices]
            results.append(self._assemble_result_row(
                question_row=row,
                letters=letters,
                prompts=q_prompts,
                responses=q_responses,
                label_to_source_index=self._build_label_to_source_index(row),
            ))
        return results

    def _assemble_result_row(
            self,
            question_row: Any,
            letters: list[str],
            prompts: list[str],
            responses: list[Any],
            label_to_source_index: dict[str, int],
    ) -> dict:
        scores: dict[str, float] = {}
        parse_oks: dict[str, bool] = {}
        for letter, response in zip(letters, responses):
            if response.is_success():
                score, ok = _parse_confidence_score(response.raw_text)
            else:
                score, ok = 0.0, False
            scores[letter] = score
            parse_oks[letter] = ok

        final_letter = _argmax_with_tiebreak(
            scores, label_to_source_index,
            seed=self.seed, benchmark_id=self.benchmark_name,
            question_id=question_row["question_id"], method_name=self.method_name,
        )
        parsed_result = ParseResult(
            final_choice=final_letter,
            status=PARSE_OK,
            raw_text=None,
            normalized_text="",
            reason="argmax_of_independent_hypothesis_scores",
        )
        score_result = self._score(parsed_result, question_row["correct_option"])

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompts[0],
            sample_index=0,
            model_response=responses[0],
            parsed_result=parsed_result,
            score_result=score_result,
        )
        for letter, response in zip(letters, responses):
            suffix = letter.lower()
            row[f"option_{suffix}_raw_text"] = response.raw_text
            row[f"option_{suffix}_model_status"] = response.status
            row[f"option_{suffix}_score"] = scores[letter]
            row[f"option_{suffix}_score_parse_ok"] = parse_oks[letter]
        return row
