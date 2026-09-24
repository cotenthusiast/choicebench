# src/choicebench/methods/library/text_extraction.py

"""
Method: Text Extraction (Visible Options, Text-Matched)
-------------------------------------------------------------------------
Paper-specific (eacl-2026-revision): one cell of the 2x2 Stage-1-visibility
x Stage-2-matcher-type diagnostic grid (visible options + embedding
matching). Options are shown to the model (unlike two_stage's hidden
Stage 1), the model is asked for the answer's text rather than a letter,
and that free text is matched back to a canonical option via the shared
exact->containment->cosine cascade (choicebench.scoring.text_matcher) --
never choicebench's letter-based free-form parser.
Reference: paper §3.4 (arXiv:2608.11947), "visible + embedding" cell.
Backend requirements: generate only
Logprob support required: no
"""

from __future__ import annotations

import json
from typing import Any

from choicebench.methods.base import ExperimentRunner
from choicebench.parsing.types import PARSE_MISSING, PARSE_OK, ParseResult
from choicebench.pipeline.prompt_builder import build_direct_mcq_prompt, build_rotations
from choicebench.scoring.text_matcher import EmbedFn, default_embed_fn, match_text_to_options
from choicebench.scoring.tiebreak import majority_vote_with_tiebreak


class TextExtractionRunner(ExperimentRunner):
    """Runner for the visible-options, text-matched condition.

    One call per question: options are visible, the model is asked for the
    answer's text (not a letter), and that response is matched back to a
    canonical option via the shared text-matching cascade.
    """

    def __init__(self, *args, embed_fn: EmbedFn | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # None (rather than defaulting eagerly to default_embed_fn) keeps
        # the real sentence-transformers model import lazy -- only paid for
        # the first time a question actually reaches the cosine stage.
        self._embed_fn = embed_fn

    def _effective_embed_fn(self) -> EmbedFn:
        if self._embed_fn is None:
            self._embed_fn = default_embed_fn
        return self._embed_fn

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        prompt = self._build_prompt(question_row)
        model_response = self._call_backend_generate(prompt)

        parsed_result = None
        score_result = None

        if model_response.is_success():
            parsed_result, score_result = self._match_and_score(
                free_text=model_response.raw_text,
                question_row=question_row,
            )

        return self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            model_response=model_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )

    def run_rotations(self, question_row: Any, sample_index: int) -> dict:
        """Full rerun under every cyclic rotation of the options -- one new
        call per rotation, since options are VISIBLE here (unlike
        two_stage's hidden stage 1), so a rotation genuinely changes what
        the model sees. Matching is always against this question's
        canonical (unrotated) option texts: the cascade matches on text
        content, not the letter/position the model saw it at, so the
        canonical answer a given response yields doesn't depend on which
        rotation elicited it.

        Args:
            question_row: Normalized question record.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary: the majority-voted answer across
            rotations, plus per_rotation_choices_json (one canonical letter,
            or null, per rotation, in rotation order).
        """
        canonical_options = self._build_options(question_row)
        rotations = build_rotations(canonical_options)

        prompts = [
            build_direct_mcq_prompt(
                template=self._prompts["text_extraction"],
                question=question_row["question_text"],
                options=rotation.mapping,
                subject=question_row["subject"],
            )
            for rotation in rotations
        ]
        responses = [self._call_backend_generate(prompt) for prompt in prompts]

        canonical_choices: list[str | None] = []
        per_rotation_raw_text: list[str | None] = []
        for response in responses:
            if response.is_success():
                parsed_result, _ = self._match_and_score(
                    free_text=response.raw_text, question_row=question_row,
                )
                canonical_choices.append(parsed_result.final_choice)
                per_rotation_raw_text.append(response.raw_text)
            else:
                canonical_choices.append(None)
                per_rotation_raw_text.append(None)

        voted_letter = majority_vote_with_tiebreak(
            canonical_choices,
            label_to_source_index=self._build_label_to_source_index(question_row),
            seed=self.seed, benchmark_id=self.benchmark_name,
            question_id=question_row["question_id"], method_name=self.method_name,
        )
        voted_parse = ParseResult(
            final_choice=voted_letter,
            status=PARSE_OK if voted_letter else PARSE_MISSING,
            raw_text=None,
            normalized_text="",
            reason="rotation_majority_vote",
        )
        score_result = None
        if voted_letter:
            score_result = self._score(voted_parse, question_row["correct_option"])

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompts[0],
            sample_index=sample_index,
            model_response=responses[0],
            parsed_result=voted_parse,
            score_result=score_result,
        )
        row["per_rotation_choices_json"] = json.dumps(canonical_choices)
        row["per_rotation_raw_text_json"] = json.dumps(per_rotation_raw_text)
        return row

    def _build_prompt(self, question_row: Any) -> str:
        return build_direct_mcq_prompt(
            template=self._prompts["text_extraction"],
            question=question_row["question_text"],
            options=self._build_options(question_row),
            subject=question_row["subject"],
        )

    def _build_batch_prompt(self, question_row: Any) -> str:
        return self._build_prompt(question_row)

    def _match_and_score(self, free_text: str, question_row: Any) -> tuple[ParseResult, Any]:
        options = self._build_options(question_row)
        matched_letter = match_text_to_options(
            free_text, options,
            embed_fn=self._effective_embed_fn(),
            label_to_source_index=self._build_label_to_source_index(question_row),
            seed=self.seed, benchmark_id=self.benchmark_name,
            question_id=question_row["question_id"], method_name=self.method_name,
        )
        parsed_result = ParseResult(
            final_choice=matched_letter,
            status=PARSE_OK if matched_letter is not None else PARSE_MISSING,
            raw_text=free_text,
            normalized_text=free_text,
            reason="text_matcher_cascade",
        )
        score_result = self._score(parsed_result, question_row["correct_option"])
        return parsed_result, score_result
