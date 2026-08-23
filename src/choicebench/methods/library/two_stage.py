# src/choicebench/methods/library/two_stage.py

"""
Method: Two-Stage Prompting (Free-Text then Match)
-----------------------------------------------------
Description: Stage one asks the model to answer the question in free text,
without showing it the lettered options at all — removing any positional
anchoring at the point the model actually reasons about the answer. Stage two
shows the model its own free-text answer alongside the four lettered options
and asks it to pick the matching letter. An optional fallback re-issues a
direct MCQ prompt if stage two's match is unparseable.
Reference: a bundled example method (one of several shipped with ChoiceBench,
not the headline) — tests whether eliciting a free-text answer before the model
ever sees option letters reduces the positional bias documented by Zheng et al.,
ICLR 2024 (arXiv:2309.03882).
Backend requirements: generate only (2 calls per question, or 3 with fallback)
Logprob support required: no
"""

from typing import Any, Sequence

from choicebench.pipeline.prompt_builder import (
    build_direct_mcq_prompt,
    build_free_text_prompt,
    build_option_matching_prompt,
)
from choicebench.methods.base import ExperimentRunner


class TwoStageRunner(ExperimentRunner):
    """Runner for the two-stage prompting condition.

    Stage one elicits a free-text answer without exposing options.
    Stage two asks the model to match that free-text answer to one
    of the four canonical options. The final letter from stage two
    is parsed and scored.

    The intermediate free-text response is preserved in the result
    row for downstream answer-matching evaluation.
    """

    def __init__(self, *args, fallback_on_parse_failure: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fallback_on_parse_failure = fallback_on_parse_failure

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute one question through the two-stage pipeline.

        Args:
            question_row: Normalized question record.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary containing trace, model output,
            parse, and score fields, plus the intermediate free-text
            response.
        """
        # Stage 1: free-text response
        free_text_prompt = build_free_text_prompt(
            template=self._prompts["free_text"],
            question=question_row["question_text"],
        )
        free_text_response = self._call_backend_generate(free_text_prompt)

        # If stage 1 fails, return early
        if not free_text_response.is_success():
            return self._build_result_row(
                question_row=question_row,
                prompt=free_text_prompt,
                sample_index=sample_index,
                model_response=free_text_response,
                parsed_result=None,
                score_result=None,
            )

        free_text_answer = free_text_response.raw_text

        # Stage 2: option matching using the free-text answer
        matching_prompt = build_option_matching_prompt(
            template=self._prompts["option_matching"],
            question=question_row["question_text"],
            free_text=free_text_answer,
            options=self._build_options(question_row),
        )
        matching_response = self._call_backend_generate(matching_prompt)

        # Parse and score the stage 2 response
        parsed_result = None
        score_result = None

        if matching_response.is_success():
            parsed_result, score_result = self._parse_and_score(
                raw_text=matching_response.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )

        # Fallback: if matching was unparseable, re-issue the direct MCQ prompt.
        # Goes through the normal backend.generate() path — caching, if any,
        # is the backend's responsibility (e.g. inside APIBackend), not
        # something this runner controls directly.
        fallback_used = False
        if (
            self._fallback_on_parse_failure
            and matching_response.is_success()
            and (parsed_result is None or parsed_result.final_choice is None)
        ):
            fallback_prompt = build_direct_mcq_prompt(
                template=self._prompts["direct_mcq"],
                question=question_row["question_text"],
                options=self._build_options(question_row),
                subject=question_row["subject"],
            )
            fallback_response = self._call_backend_generate(fallback_prompt)
            if fallback_response.is_success():
                parsed_result, score_result = self._parse_and_score(
                    raw_text=fallback_response.raw_text,
                    correct_option=question_row["correct_option"],
                    options=self._build_options(question_row),
                )
                fallback_used = True

        result = self._build_result_row(
            question_row=question_row,
            prompt=matching_prompt,
            sample_index=sample_index,
            model_response=matching_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )

        # Preserve the intermediate free-text response for answer matching
        result["free_text_prompt"] = free_text_prompt
        result["free_text_response"] = free_text_answer
        result["free_text_latency"] = free_text_response.latency_seconds
        result["fallback_used"] = fallback_used

        return result

    async def _run_phases(
        self, rows: list[dict]
    ) -> tuple[
        list[str],
        list[Any],
        dict[int, str],
        dict[int, Any],
        dict[int, Any],
        dict[int, Any],
        set[int],
    ]:
        """Run the three batched phases for TwoStageRunner.

        Phase 1 — all free-text prompts in one generate_batch() call.
        Phase 2 — option-matching prompts for questions where phase 1
                  succeeded, in one generate_batch() call.
        Phase 3 (optional) — fallback direct-MCQ prompts for questions
                  where phase 2 was unparseable and fallback_on_parse_failure
                  is enabled, in one generate_batch() call.

        Returns:
            s1_prompts, s1_responses, s2_prompt_by_idx, s2_resp_by_idx,
            parsed_by_idx, scored_by_idx, fb_used_set
        """
        # Phase 1: free-text answers for all questions.
        s1_prompts = [
            build_free_text_prompt(self._prompts["free_text"], row["question_text"])
            for row in rows
        ]
        s1_responses = await self.backend.generate_batch(s1_prompts)

        # Phase 2: option-matching for questions where phase 1 succeeded.
        s2_prompt_by_idx: dict[int, str] = {}
        for i, (row, s1_resp) in enumerate(zip(rows, s1_responses)):
            if s1_resp.is_success():
                s2_prompt_by_idx[i] = build_option_matching_prompt(
                    template=self._prompts["option_matching"],
                    question=row["question_text"],
                    free_text=s1_resp.raw_text,
                    options=self._build_options(row),
                )

        s2_indices = sorted(s2_prompt_by_idx)
        s2_prompts = [s2_prompt_by_idx[i] for i in s2_indices]
        s2_resp_list = await self.backend.generate_batch(s2_prompts) if s2_prompts else []
        s2_resp_by_idx = dict(zip(s2_indices, s2_resp_list))

        # Parse phase-2 responses; collect fallback candidates.
        parsed_by_idx: dict[int, Any] = {}
        scored_by_idx: dict[int, Any] = {}
        fb_prompt_by_idx: dict[int, str] = {}

        for i in s2_indices:
            row = rows[i]
            s2_resp = s2_resp_by_idx[i]
            if s2_resp.is_success():
                parsed, scored = self._parse_and_score(
                    s2_resp.raw_text, row["correct_option"], self._build_options(row)
                )
                parsed_by_idx[i] = parsed
                scored_by_idx[i] = scored
                if (
                    self._fallback_on_parse_failure
                    and (parsed is None or parsed.final_choice is None)
                ):
                    fb_prompt_by_idx[i] = build_direct_mcq_prompt(
                        template=self._prompts["direct_mcq"],
                        question=row["question_text"],
                        options=self._build_options(row),
                        subject=row["subject"],
                    )
            else:
                parsed_by_idx[i] = None
                scored_by_idx[i] = None

        # Phase 3 (optional): fallback direct-MCQ.
        fb_indices = sorted(fb_prompt_by_idx)
        fb_prompts = [fb_prompt_by_idx[i] for i in fb_indices]
        fb_resp_list = await self.backend.generate_batch(fb_prompts) if fb_prompts else []
        fb_used_set: set[int] = set()

        for i, fb_resp in zip(fb_indices, fb_resp_list):
            if fb_resp.is_success():
                row = rows[i]
                parsed, scored = self._parse_and_score(
                    fb_resp.raw_text, row["correct_option"], self._build_options(row)
                )
                parsed_by_idx[i] = parsed
                scored_by_idx[i] = scored
                fb_used_set.add(i)

        return (
            s1_prompts,
            s1_responses,
            s2_prompt_by_idx,
            s2_resp_by_idx,
            parsed_by_idx,
            scored_by_idx,
            fb_used_set,
        )

    async def run_many_async(self, question_rows: Sequence[Any]) -> list[dict]:
        """Async batch execution for TwoStageRunner.

        Runs the three batched phases via `_run_phases`, then builds result
        rows in original question order.
        """
        rows = question_rows.to_dict(orient="records")

        (
            s1_prompts,
            s1_responses,
            s2_prompt_by_idx,
            s2_resp_by_idx,
            parsed_by_idx,
            scored_by_idx,
            fb_used_set,
        ) = await self._run_phases(rows)

        # Assemble results in original question order.
        results = []
        for i, (row, s1_resp, s1_prompt) in enumerate(zip(rows, s1_responses, s1_prompts)):
            if not s1_resp.is_success():
                result = self._build_result_row(
                    question_row=row,
                    prompt=s1_prompt,
                    sample_index=i,
                    model_response=s1_resp,
                    parsed_result=None,
                    score_result=None,
                )
                result["free_text_prompt"] = s1_prompt
                result["free_text_response"] = None
                result["free_text_latency"] = s1_resp.latency_seconds
                result["fallback_used"] = False
            else:
                s2_resp = s2_resp_by_idx[i]
                result = self._build_result_row(
                    question_row=row,
                    prompt=s2_prompt_by_idx[i],
                    sample_index=i,
                    model_response=s2_resp,
                    parsed_result=parsed_by_idx.get(i),
                    score_result=scored_by_idx.get(i),
                )
                result["free_text_prompt"] = s1_prompt
                result["free_text_response"] = s1_resp.raw_text
                result["free_text_latency"] = s1_resp.latency_seconds
                result["fallback_used"] = i in fb_used_set
            results.append(result)

        return results
