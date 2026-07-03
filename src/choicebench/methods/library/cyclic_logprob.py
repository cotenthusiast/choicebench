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
Logprob support required: yes (HuggingFace or Dummy — no API provider exposes
score_options; see config/schema.py's model_supports_logprobs()).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from choicebench.clients.types import FAILURE_STATUS, SUCCESS_STATUS
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

    Requires a backend that implements score_options() (e.g. HuggingFaceBackend).
    Check backend.supports_logprobs before constructing, or rely on the
    schema-layer requires_score_options guard.
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
        # PF-4: surface how many permutations fell back to a uniform distribution
        # so a partially-degraded row is distinguishable in the CSV, not only in
        # the scrolled log.
        row["n_permutations_total"] = len(prompts)
        row["n_permutations_failed"] = len(prompts) - n_success
        if final_letter is not None:
            row["parsed_choice"] = final_letter
            row["parse_status"] = PARSE_OK
        # Logprob method: generate() is never called, so _build_result_row's
        # transport_status defaults to None (no call was made). answer_status
        # is set explicitly here (PF-3) since parsed_result=None was passed
        # above (the row is patched with parsed_choice/parse_status directly
        # instead) — success iff a final answer was produced.
        row["answer_status"] = SUCCESS_STATUS if final_letter is not None else FAILURE_STATUS
        if score_result is not None:
            row["is_correct"] = score_result.is_correct
            row["score_status"] = score_result.status

        return row
