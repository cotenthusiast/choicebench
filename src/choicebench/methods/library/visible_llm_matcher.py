# src/choicebench/methods/library/visible_llm_matcher.py

"""
Method: Visible + LLM Matcher ("fourth cell")
-------------------------------------------------------------------------
Paper-specific (eacl-2026-revision): the fourth cell of the 2x2 Stage-1-
visibility x Stage-2-matcher-type diagnostic grid (visible options + LLM
matching) -- reconciled from the historical choicebench
worktree-visible-llm-matcher-cell branch into the canonical implementation.

Reuses text_extraction's Stage-1 output (the extracted answer text for a
question under visible options) rather than re-issuing that call --
question rows passed to this runner must already carry an
"extracted_text" column (populated upstream, e.g. by joining a saved
text_extraction result CSV on question_id; this runner does not read that
CSV itself, to keep it a plain single-call ExperimentRunner). The one NEW
call this method makes is Stage 2: an LLM match of that reused text
against the visible canonical options, via the same option_matching
prompt/parser two_stage's own Stage 2 uses.
Reference: paper §3.4 (arXiv:2608.11947), "visible + LLM matching" cell.
Backend requirements: generate only
Logprob support required: no
"""

import json
from typing import Any

from choicebench.methods.base import ExperimentRunner
from choicebench.parsing.types import PARSE_MISSING, PARSE_OK, ParseResult
from choicebench.pipeline.prompt_builder import build_option_matching_prompt, build_rotations
from choicebench.scoring.tiebreak import majority_vote_with_tiebreak


class VisibleLLMMatcherRunner(ExperimentRunner):
    """Runner for the visible-options, LLM-matched condition (fourth cell).

    Reuses a prior text_extraction run's extracted text (must already be
    present as question_row["extracted_text"]); makes exactly one new
    backend call to match that text to a canonical option via an LLM.
    """

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        if "extracted_text" not in question_row:
            raise KeyError(
                "VisibleLLMMatcherRunner requires question_row['extracted_text'] "
                "(text_extraction's reused Stage-1 output) -- join it in upstream "
                "before this method runs; this runner never re-issues that call."
            )
        self._validate_extracted_text_source(question_row)
        extracted_text = question_row["extracted_text"]

        prompt = self._build_prompt(question_row, extracted_text)
        model_response = self._call_backend_generate(prompt)

        parsed_result = None
        score_result = None
        if model_response.is_success():
            parsed_result, score_result = self._parse_and_score(
                raw_text=model_response.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            model_response=model_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )
        row["reused_extracted_text"] = extracted_text
        return row

    def run_matching_rotations(
            self,
            question_row: Any,
            per_rotation_extracted_text: list[str | None],
            sample_index: int,
    ) -> dict:
        """Rerun ONLY the Stage-2 LLM match under every cyclic rotation of
        the options, reusing text_extraction's OWN per-rotation extracted
        text -- never re-eliciting Stage 1.

        Args:
            question_row: Normalized question record.
            per_rotation_extracted_text: One extracted-text entry per
                rotation (in rotation order, matching build_rotations'
                output for this question's canonical options), from a
                prior text_extraction run_rotations call's
                per_rotation_raw_text_json. A None entry means that
                rotation's own Stage-1 call failed upstream -- skipped
                here, no new call made for it.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary: the majority-voted answer across
            rotations, plus per_rotation_choices_json (one canonical
            letter, or null, per rotation, in rotation order).
        """
        canonical_options = self._build_options(question_row)
        canonical_letters = list(canonical_options.keys())
        rotations = build_rotations(canonical_options)

        if len(per_rotation_extracted_text) != len(rotations):
            raise ValueError(
                "VisibleLLMMatcherRunner.run_matching_rotations: expected "
                f"{len(rotations)} per_rotation_extracted_text entries (one per "
                f"rotation, matching this question's {len(rotations)} options), "
                f"got {len(per_rotation_extracted_text)} -- refusing to silently "
                "zip/truncate a mismatched trace."
            )
        self._validate_extracted_text_source(question_row)

        canonical_choices: list[str | None] = []
        representative_prompt: str | None = None
        representative_response = None
        for rotation, extracted_text in zip(rotations, per_rotation_extracted_text):
            if extracted_text is None:
                canonical_choices.append(None)
                continue

            prompt = build_option_matching_prompt(
                template=self._prompts["option_matching"],
                question=question_row["question_text"],
                free_text=extracted_text,
                options=rotation.mapping,
            )
            response = self._call_backend_generate(prompt)
            if representative_prompt is None:
                representative_prompt = prompt
                representative_response = response

            if response.is_success():
                parsed = self._parse(response.raw_text, rotation.mapping)
                if parsed.final_choice is not None:
                    display_index = canonical_letters.index(parsed.final_choice)
                    canonical_choices.append(
                        canonical_letters[rotation.slot_to_canonical[display_index]]
                    )
                else:
                    canonical_choices.append(None)
            else:
                canonical_choices.append(None)

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
            prompt=representative_prompt or "",
            sample_index=sample_index,
            model_response=representative_response,
            parsed_result=voted_parse,
            score_result=score_result,
        )
        row["per_rotation_choices_json"] = json.dumps(canonical_choices)
        return row

    def _validate_extracted_text_source(self, question_row: Any) -> None:
        """Reject a reused extracted_text whose own recorded producer
        (model/benchmark/method) doesn't match this run -- cross-model or
        cross-condition artifacts must never silently mix. All three
        fields are OPTIONAL (additive, versioned convention): a
        question_row without them is unvalidated exactly as before this
        existed, since a real production join predating this field never
        populated it -- see run_visible_llm_matcher.py/
        run_visible_llm_matcher_rotations.py, which now stamp it (see their
        own headers).
        """
        source_model = question_row.get("extracted_text_source_model")
        if source_model is not None:
            expected_model = self.model_label or self.backend.model_name
            if str(source_model) != str(expected_model):
                raise ValueError(
                    "VisibleLLMMatcherRunner: question_row['extracted_text_source_model']="
                    f"{source_model!r} does not match this run's own model "
                    f"{expected_model!r} -- refusing to reuse a different model's "
                    "extracted_text as this run's Stage 1."
                )
        source_benchmark = question_row.get("extracted_text_source_benchmark")
        if source_benchmark is not None and str(source_benchmark) != str(self.benchmark_name):
            raise ValueError(
                "VisibleLLMMatcherRunner: question_row['extracted_text_source_benchmark']="
                f"{source_benchmark!r} does not match this run's own benchmark "
                f"{self.benchmark_name!r} -- refusing to reuse a different "
                "benchmark's extracted_text as this run's Stage 1."
            )
        source_method = question_row.get("extracted_text_source_method")
        if source_method is not None and str(source_method) != "text_extraction":
            raise ValueError(
                "VisibleLLMMatcherRunner: question_row['extracted_text_source_method']="
                f"{source_method!r} -- extracted_text must come from a "
                "text_extraction run, refusing to reuse a different method's output."
            )

    def _build_prompt(self, question_row: Any, extracted_text: str) -> str:
        return build_option_matching_prompt(
            template=self._prompts["option_matching"],
            question=question_row["question_text"],
            free_text=extracted_text,
            options=self._build_options(question_row),
        )

    def _build_batch_prompt(self, question_row: Any) -> str:
        return self._build_prompt(question_row, question_row["extracted_text"])
