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

from typing import Any

from choicebench.methods.base import ExperimentRunner
from choicebench.pipeline.prompt_builder import build_option_matching_prompt


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

    def _build_prompt(self, question_row: Any, extracted_text: str) -> str:
        return build_option_matching_prompt(
            template=self._prompts["option_matching"],
            question=question_row["question_text"],
            free_text=extracted_text,
            options=self._build_options(question_row),
        )

    def _build_batch_prompt(self, question_row: Any) -> str:
        return self._build_prompt(question_row, question_row["extracted_text"])
