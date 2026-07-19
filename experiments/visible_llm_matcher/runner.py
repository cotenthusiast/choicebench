# experiments/visible_llm_matcher/runner.py
#
# VisibleLlmMatcherRunner: the "options visible + LLM matcher" missing 2x2
# cell. Fills the gap between the three existing cells:
#
#   options hidden  + LLM matcher       = TSP two_prompt / choicebench two_stage
#   options hidden  + embedding matcher = TSP twostage_semantic_match (post-hoc)
#   options visible + embedding matcher = TSP/MG text_extraction
#   options visible + LLM matcher       = THIS RUNNER (previously missing)
#
# Design constraint (explicit user decision): Stage 1 is NOT regenerated.
# This runner takes ALREADY-VALIDATED Stage-1 rows (see stage1_sources.py —
# reused text_extraction free-text answers, options visible) and executes
# ONLY a new Stage-2 LLM-matcher call, using the historical two-stage-
# prompting option_matching.txt protocol ported verbatim in
# historical_protocol.py (NOT choicebench.methods.library.two_stage or
# choicebench.pipeline.prompt_builder / choicebench.parsing.parser, which
# are verified NOT behavior-identical — see historical_protocol.py's module
# docstring).
#
# This is deliberately NOT registered in choicebench.registry.METHOD_REGISTRY
# and does not live under src/choicebench/methods/. Per the user's
# instruction to keep this as custom experiment code rather than expand the
# shipped method surface, it is loaded the way ChoiceBench's own README
# documents for external methods: "module.path:ClassName" in a YAML config,
# with zero changes to the installed choicebench package. This module is
# only reachable if experiments/ is on PYTHONPATH (e.g. `pip install -e .`
# from the repo root, which puts the repo root — and therefore
# experiments/ — on sys.path via the working directory / editable install
# layout; see experiments/visible_llm_matcher/README.md for the exact
# invocation once a run config exists).
#
# IMPORTANT: question_id compatibility. ChoiceBench's own benchmark
# normalizers (src/choicebench/benchmarks/arc.py, mmlu.py) hash
# question_id from subject+question+choices, and that hash does NOT agree
# with two-stage-prompting's own historical question_id hash for at least 6
# of the 1000 ARC-Challenge robustness-split questions (verified directly:
# 3 are the known missing-4th-option rows — see
# stage1_sources.KNOWN_3OPTION_ARC_QUESTION_IDS — and 3 more mismatch for an
# unidentified text-normalization reason, all with 4 real options). Because
# of this, VisibleLlmMatcherRunner does NOT go through ChoiceBench's normal
# benchmark-loading path (config `benchmarks:` -> normalized CSV -> question
# rows). It is driven directly by the validated Stage-1 DataFrame returned
# by stage1_sources.load_and_validate_stage1(), which already carries
# TSP's own question_id, question_text, choice_a..choice_d, and
# correct_option for all 1000 questions. This sidesteps the mismatch
# entirely rather than silently dropping or mis-joining 6 questions.

from __future__ import annotations

from typing import Any

from choicebench.methods.base import ExperimentRunner
from choicebench.parsing.types import ParseResult
from choicebench.scoring.scorer import score_prediction
from choicebench.scoring.types import ScoreResult

from experiments.visible_llm_matcher.historical_protocol import (
    build_option_matching_prompt,
    parse_model_answer,
)


class VisibleLlmMatcherRunner(ExperimentRunner):
    """Stage-2-only runner: options-visible free text (reused) -> LLM matcher.

    Unlike every built-in ChoiceBench method (and unlike TSP's own
    TwoStageRunner), this runner makes exactly ONE backend call per
    question — the Stage-2 matching call. Stage 1 is supplied at
    construction time via stage1_lookup, not generated.
    """

    def __init__(
        self,
        *args,
        stage1_lookup: dict[str, dict[str, Any]],
        **kwargs,
    ) -> None:
        """
        Args:
            stage1_lookup: question_id -> row dict with at least
                free_text_response, choice_a, choice_b, choice_c, choice_d.
                Must be the output of
                stage1_sources.load_and_validate_stage1() converted via
                .set_index("question_id").to_dict("index") (or equivalent) —
                i.e. already fail-closed validated. This runner does not
                re-validate contamination/completeness; that is
                stage1_sources.py's job, kept separate so this class stays
                a plain per-question ExperimentRunner.
            *args, **kwargs: forwarded to ExperimentRunner (backend,
                method_name, split_name, prompt_version, prompts_dir,
                run_id, temperature, max_tokens, seed, perturbation_name,
                model_label).
        """
        super().__init__(*args, **kwargs)
        self._stage1_lookup = stage1_lookup

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute Stage 2 only for one question.

        Args:
            question_row: A row from the validated Stage-1 DataFrame (see
                stage1_sources.load_and_validate_stage1()) — NOT a
                ChoiceBench-normalized benchmark row. Must contain
                question_id, question_text, choice_a..choice_d,
                correct_option, subject, free_text_response.
            sample_index: Repetition index (forwarded to _build_result_row).

        Returns:
            Flat result dictionary via _build_result_row(), plus
            free_text_response/free_text_source_repo/free_text_source_path
            carried through from the reused Stage 1 input for provenance.
        """
        question_id = question_row["question_id"]
        stage1_row = self._stage1_lookup.get(question_id)
        if stage1_row is None:
            raise KeyError(
                f"No reused Stage-1 row for question_id={question_id!r}. "
                "stage1_lookup must cover every question in the run — this "
                "should have been caught by stage1_sources.py's row-count "
                "validation before the runner was constructed."
            )

        free_text_answer = stage1_row["free_text_response"]

        matching_prompt = build_option_matching_prompt(
            template=self._prompts["option_matching"],
            question=question_row["question_text"],
            free_text=free_text_answer,
            option_a=question_row["choice_a"],
            option_b=question_row["choice_b"],
            option_c=question_row["choice_c"],
            option_d=question_row["choice_d"],
        )
        matching_response = self._call_backend_generate(matching_prompt)

        parsed_result = None
        score_result = None
        if matching_response.is_success():
            parsed_result, score_result = self._parse_and_score(
                raw_text=matching_response.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )

        result = self._build_result_row(
            question_row=question_row,
            prompt=matching_prompt,
            sample_index=sample_index,
            model_response=matching_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )

        # Provenance: which reused Stage-1 row (and source repo/file) fed
        # this Stage-2 call, so the result row is independently auditable
        # without re-joining against stage1_sources.py output.
        result["free_text_response"] = free_text_answer
        result["free_text_source_repo"] = stage1_row.get("_source_repo")
        result["free_text_source_path"] = stage1_row.get("_source_path")

        return result

    @staticmethod
    def _parse(raw_text: str, options: dict[str, str]) -> ParseResult:
        """Override: use the ported TSP parser, not choicebench.parsing.parser.

        See historical_protocol.py's module docstring for the exact
        behavioral differences this avoids inheriting.
        """
        return parse_model_answer(raw_text, options)

    @staticmethod
    def _score(parse_result: ParseResult, correct_option: str) -> ScoreResult:
        """choicebench.scoring.scorer IS reused unmodified here — verified
        functionally identical to twoprompt.scoring.scorer (docstring-only
        diff), so no port was needed (see historical_protocol.py header).
        """
        return score_prediction(parse_result, correct_option)
