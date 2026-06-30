# src/choicebench/methods/library/cyclic_logprob.py

"""
Method: Logprob Cyclic Permutation (Eq. 1 averaging)
-----------------------------------------------------
Description: Generates all N cyclic permutations of the option ordering for a
question, calls score_options() on each permutation to obtain per-letter logprob
distributions, stacks them into an (N, N) matrix, and applies Eq. (1) from
Zheng et al. to average probability mass back to canonical content slots.
Argmax over content slots gives the final answer. Unlike the majority-vote
PermutationRunner, this method uses logprob arithmetic rather than text generation.
Reference: Zheng et al., ICLR 2024, "Large Language Models Are Not Robust
Multiple Choice Selectors" (arXiv:2309.03882) — §2.2, Eq. (1).
Backend requirements: score_options
Logprob support required: yes
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Sequence

import numpy as np

from choicebench.parsing.types import PARSE_OK, ParseResult
from choicebench.methods.base import ExperimentRunner
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.methods.library.pride_math import (
    equation1_cyclic_debiased_content_probs,
    logprob_map_to_label_distribution,
)

logger = logging.getLogger(__name__)


class CyclicLogprobRunner(ExperimentRunner):
    """Logprob cyclic permutation via Eq. (1) averaging (Zheng et al., ICLR 2024).

    Requires a backend that implements score_options() (e.g. HuggingFaceBackend
    or APIBackend wrapping VLLMClient). Check backend.supports_logprobs before
    constructing, or rely on the schema-layer requires_score_options guard.
    """

    requires_score_options: bool = True

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        canon = self._build_options(question_row)
        letters = list(canon.keys())
        n = len(letters)
        permutations = PermutationRunner._generate_permutations(canon)

        prompts = [
            PermutationRunner._build_permuted_prompt(
                question_row, perm, self._prompts["direct_mcq"]
            )
            for perm in permutations
        ]

        uni = np.ones(n, dtype=np.float64) / n
        dist_rows: list[np.ndarray] = []
        n_success = 0
        scoring_error: str | None = None

        for prompt in prompts:
            try:
                scores = self.backend.score_options(prompt, letters)
                lp_map = dict(zip(letters, scores))
                dist_rows.append(
                    logprob_map_to_label_distribution(lp_map, letters=letters)
                )
                n_success += 1
            except Exception as exc:
                logger.warning("CyclicLogprob: score_options failed — %s", exc)
                dist_rows.append(uni.copy())
                if scoring_error is None:
                    scoring_error = str(exc)

        mat = np.stack(dist_rows, axis=0).astype(np.float64)  # shape (N, N)
        content_probs = equation1_cyclic_debiased_content_probs(mat)

        final_letter: str | None = None
        score_result = None
        if n_success > 0:
            final_letter = letters[int(np.argmax(content_probs))]
            parse = ParseResult(
                final_choice=final_letter,
                status=PARSE_OK,
                raw_text=None,
                normalized_text=final_letter,
                reason="eq1_averaging",
            )
            score_result = self._score(parse, question_row["correct_option"])

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompts[0],
            sample_index=sample_index,
            model_response=None,
            parsed_result=None,
            score_result=None,
            error=scoring_error,
        )
        row["cyclic_logprob_inference_mode"] = "eq1_averaging"
        row["option_distributions_json"] = json.dumps(mat.tolist())
        if final_letter is not None:
            row["parsed_choice"] = final_letter
            row["parse_status"] = PARSE_OK
        if score_result is not None:
            row["is_correct"] = score_result.is_correct
            row["score_status"] = score_result.status

        return row

    async def run_many_async(self, question_rows: Sequence[Any]) -> list[dict]:
        """Async inference path for CyclicLogprobRunner.

        Fans out all N_questions × N_permutations score_options_async() calls in
        one asyncio.gather(), then reassembles per-question results and applies
        equation1_cyclic_debiased_content_probs(). Mirrors PriDeRunner.run_many_async()
        but flattens across permutations as well as questions for maximum concurrency.

        Raises:
            NotImplementedError: if the backend does not support score_options_async().
        """
        if not self.backend.supports_score_options():
            raise NotImplementedError(
                f"CyclicLogprobRunner.run_many_async() requires a backend with "
                f"score_options() support (e.g. vLLM). "
                f"{self.backend.__class__.__name__} does not support it. "
                "Check backend.supports_score_options() before calling."
            )

        rows = question_rows.to_dict(orient="records")

        # Build per-question metadata and flatten prompts for the gather.
        per_q: list[tuple[list[str], list[str]]] = []  # (letters, prompts)
        all_tasks: list[tuple[str, list[str]]] = []

        for row in rows:
            canon = self._build_options(row)
            letters = list(canon.keys())
            perms = PermutationRunner._generate_permutations(canon)
            q_prompts = [
                PermutationRunner._build_permuted_prompt(
                    row, perm, self._prompts["direct_mcq"]
                )
                for perm in perms
            ]
            per_q.append((letters, q_prompts))
            for prompt in q_prompts:
                all_tasks.append((prompt, letters))

        from choicebench.clients.vllm_client import VLLMClient
        raw: VLLMClient = self.backend._raw_client

        all_logprob_results = await asyncio.gather(
            *[raw.score_options_async(prompt, letters) for prompt, letters in all_tasks],
            return_exceptions=True,
        )

        results: list[dict] = []
        flat_idx = 0

        for q_idx, (row, (letters, q_prompts)) in enumerate(zip(rows, per_q)):
            n_perms = len(q_prompts)
            n = len(letters)
            uni = np.ones(n, dtype=np.float64) / n
            dist_rows: list[np.ndarray] = []
            n_success = 0
            scoring_error: str | None = None

            for _ in range(n_perms):
                result = all_logprob_results[flat_idx]
                flat_idx += 1
                if isinstance(result, Exception):
                    logger.warning(
                        "CyclicLogprob: score_options_async failed — %s", result
                    )
                    dist_rows.append(uni.copy())
                    if scoring_error is None:
                        scoring_error = str(result)
                else:
                    lp_map = {opt: result.get(opt, -100.0) for opt in letters}
                    dist_rows.append(
                        logprob_map_to_label_distribution(lp_map, letters=letters)
                    )
                    n_success += 1

            mat = np.stack(dist_rows, axis=0).astype(np.float64)
            content_probs = equation1_cyclic_debiased_content_probs(mat)

            final_letter: str | None = None
            score_result = None
            if n_success > 0:
                final_letter = letters[int(np.argmax(content_probs))]
                parse = ParseResult(
                    final_choice=final_letter,
                    status=PARSE_OK,
                    raw_text=None,
                    normalized_text=final_letter,
                    reason="eq1_averaging",
                )
                score_result = self._score(parse, row["correct_option"])

            result_row = self._build_result_row(
                question_row=row,
                prompt=q_prompts[0],
                sample_index=q_idx,
                model_response=None,
                parsed_result=None,
                score_result=None,
                error=scoring_error,
            )
            result_row["cyclic_logprob_inference_mode"] = "eq1_averaging"
            result_row["option_distributions_json"] = json.dumps(mat.tolist())
            if final_letter is not None:
                result_row["parsed_choice"] = final_letter
                result_row["parse_status"] = PARSE_OK
            if score_result is not None:
                result_row["is_correct"] = score_result.is_correct
                result_row["score_status"] = score_result.status

            results.append(result_row)

        return results
