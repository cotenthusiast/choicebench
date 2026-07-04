# src/choicebench/methods/library/direct_logprob.py

"""
Method: Direct Logprob (Default Baseline)
-------------------------------------------
Description: The paper's "Default" baseline for open-source models: reads the
next-token log-probabilities of the option-ID tokens (A, B, C, ...) at the
answer position via a single backend.score_options() call and predicts the
argmax, rather than generating free text and parsing it (see direct_mcq for
the generate-and-parse baseline). One call per question, no permutations, no
calibration.
Reference: Zheng et al., ICLR 2024, "Large Language Models Are Not Robust
Multiple Choice Selectors" (arXiv:2309.03882) — §2.1 ("Default" baseline).
Backend requirements: score_options
Logprob support required: yes (HuggingFace or Dummy — no API provider exposes
score_options; see config/schema.py's model_supports_logprobs()).
Calls per question: 1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from choicebench.backends.base import BaseBackend
from choicebench.clients.types import FAILURE_STATUS, SUCCESS_STATUS
from choicebench.config.providers import MAX_TOKENS, SEED, TEMPERATURE
from choicebench.parsing.types import PARSE_OK, ParseResult
from choicebench.methods.base import ExperimentRunner
from choicebench.methods.library.pride_math import logprob_map_to_label_distribution
from choicebench.pipeline.prompt_builder import build_direct_mcq_prompt

logger = logging.getLogger(__name__)


class DirectLogprobRunner(ExperimentRunner):
    """Runner for the Default (logprob) baseline (Zheng et al., ICLR 2024, §2.1).

    Requires a backend that implements score_options() (e.g. HuggingFaceBackend);
    checked eagerly in __init__, mirroring PriDeRunner's guard.
    """

    requires_score_options: bool = True

    def __init__(
            self,
            backend: BaseBackend,
            method_name: str,
            split_name: str,
            prompt_version: str,
            prompts_dir: Path,
            run_id: str,
            temperature: float = TEMPERATURE,
            max_tokens: int = MAX_TOKENS,
            seed: int | None = SEED,
            perturbation_name: str | None = None,
            model_label: str | None = None,
    ) -> None:
        super().__init__(
            backend=backend,
            method_name=method_name,
            split_name=split_name,
            prompt_version=prompt_version,
            prompts_dir=prompts_dir,
            run_id=run_id,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            perturbation_name=perturbation_name,
            model_label=model_label,
        )
        # direct_logprob needs per-letter logprobs, so it requires a backend
        # that implements score_options(). Gate on the capability flag rather
        # than a provider name, so any logprob-capable backend works.
        if not backend.supports_logprobs:
            raise ValueError(
                f"direct_logprob requires a backend with score_options() support; "
                f"{backend.__class__.__name__} does not "
                f"(supports_logprobs=False)."
            )

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        options = self._build_options(question_row)
        letters = list(options.keys())
        prompt = self._build_prompt(question_row)

        lp_map: dict[str, float] = {}
        scoring_error: str | None = None
        try:
            scores = self.backend.score_options(prompt, letters)
            lp_map = dict(zip(letters, scores))
        except Exception as exc:
            scoring_error = str(exc)
            logger.warning(
                "direct_logprob: score_options failed for question %s — %s",
                question_row["question_id"],
                exc,
            )

        final_letter: str | None = None
        score_result = None
        distribution: np.ndarray | None = None
        if lp_map:
            distribution = logprob_map_to_label_distribution(lp_map, letters=letters)
            final_letter = letters[int(np.argmax(distribution))]
            parse = ParseResult(
                final_choice=final_letter,
                status=PARSE_OK,
                raw_text=None,
                normalized_text=final_letter,
                reason="direct_logprob_argmax",
            )
            score_result = self._score(parse, question_row["correct_option"])

        # direct_logprob scores purely via score_options() — it never calls
        # backend.generate(), so there is no raw model text to record.
        row = self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            model_response=None,
            parsed_result=None,
            score_result=None,
            error=scoring_error,
        )
        row["direct_logprob_inference_mode"] = "argmax"
        # Persisted in the same column/format cyclic_logprob uses for its
        # per-permutation distributions (a JSON matrix, letters in canonical
        # order) — here shape (1, n_letters) since there is only one call, no
        # permutations. A downstream script recomputing PriDe's Eq.(8)
        # offline from these CSVs can read row 0 as this question's default
        # label distribution; keeping the matrix shape (rather than a bare
        # list) also means order_sensitivity's shape assumptions don't break
        # if it's ever pointed at a direct_logprob run.
        row["option_distributions_json"] = (
            json.dumps([distribution.tolist()]) if distribution is not None else None
        )
        if final_letter is not None:
            row["parsed_choice"] = final_letter
            row["parse_status"] = PARSE_OK
        # direct_logprob never calls generate(), so _build_result_row's
        # transport_status defaults to None (no call was made). answer_status
        # is set explicitly here since parsed_result=None was passed above —
        # success iff score_options yielded a letter distribution to argmax.
        row["answer_status"] = SUCCESS_STATUS if final_letter is not None else FAILURE_STATUS
        if score_result is not None:
            row["is_correct"] = score_result.is_correct
            row["score_status"] = score_result.status

        return row

    def _build_prompt(self, question_row: Any) -> str:
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=self._build_options(question_row),
        )
