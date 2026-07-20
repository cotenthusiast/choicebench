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
# EXCEPTION, audited repair only: the historical protocol's 4-slot
# serialization renders a missing 4th option as literal "D. nan" — a real
# prompt defect at Stage-1 elicitation time for the 3 genuinely-3-option ARC
# questions (see stage1_sources.KNOWN_3OPTION_ARC_QUESTION_IDS), which is
# exactly what corrected replacement Stage-1 rows exist to fix. For THESE 3
# question_ids only, Stage 2 renders via repaired_stage2.py's dedicated
# A/B/C-only template instead of historical_protocol.build_option_matching_prompt
# — otherwise Stage 2 would silently reintroduce the same phantom-option
# contamination one stage later. historical_protocol.py itself is untouched
# and its D.nan-producing output is retained only as provenance/regression
# evidence (tests/experiments/test_historical_protocol.py); it is not called
# by this runner for these 3 IDs. Every other row (all ordinary 4-option
# questions) still goes through the byte-faithful historical path — see
# _option_is_missing() and its use in run_one() below.
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
# IMPORTANT: question_id authority. ChoiceBench's own benchmark normalizers
# (src/choicebench/benchmarks/arc.py, mmlu.py) hash question_id from
# subject+question+choices independently of the historical repos, and an
# EARLIER version of this comment claimed that hash disagreed with TSP's for
# 6 of the 1000 ARC-Challenge robustness-split questions. That comparison
# used the wrong authority (ChoiceBench's current, independently
# re-normalized dataset) — see arc_freeze_validation.py. Re-verified against
# the actual authority for this paper experiment, the immutable Stage 1
# freeze (model-generalization/paper_data_freeze/raw/local_model_generalization/),
# there are ZERO id or content mismatches across all 1000 questions, for all
# 6 reused Stage-1 sources. VisibleLlmMatcherRunner still does NOT go
# through ChoiceBench's normal benchmark-loading path (config `benchmarks:`
# -> normalized CSV -> question rows) — not because of an id mismatch, but
# because Stage 1 is reused as-is and the freeze, not ChoiceBench's own
# normalizer, is the correct identifier authority; see
# arc_freeze_validation.validate_against_freeze(), which every ARC run must
# pass before this runner is invoked (see repairs/group3_fourth_cell/).
# It is driven directly by the validated Stage-1 DataFrame returned by
# stage1_sources.load_and_validate_stage1(), which already carries the
# freeze's own question_id, question_text, choice_a..choice_d, and
# correct_option for all 1000 questions, unchanged.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from choicebench.methods.base import ExperimentRunner
from choicebench.parsing.types import ParseResult
from choicebench.scoring.scorer import score_prediction
from choicebench.scoring.types import ScoreResult

from experiments.visible_llm_matcher.historical_protocol import (
    build_option_matching_prompt,
    parse_model_answer,
)
from experiments.visible_llm_matcher.repaired_stage2 import (
    REPAIRED_3OPTION_TEMPLATE_NAME,
    build_repaired_3option_matching_prompt,
)
from experiments.visible_llm_matcher.stage1_sources import KNOWN_3OPTION_ARC_QUESTION_IDS


def _option_is_missing(value: object) -> bool:
    """True for None, NaN, or an empty/whitespace-only string.

    Used only to decide which Stage-2 template to render (see run_one) —
    not a parsing/scoring change.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


class VisibleLlmMatcherRunner(ExperimentRunner):
    """Stage-2-only runner: options-visible free text (reused) -> LLM matcher.

    Unlike every built-in ChoiceBench method (and unlike TSP's own
    TwoStageRunner), this runner makes exactly ONE backend call per
    question — the Stage-2 matching call. Stage 1 is supplied at
    construction time via stage1_lookup, not generated.
    """

    def __init__(
        self,
        *,
        prompts_dir: Path,
        prompt_version: str,
        stage1_lookup: dict[str, dict[str, Any]],
        **kwargs,
    ) -> None:
        """
        Args:
            prompts_dir: Root prompt bundle directory, forwarded to
                ExperimentRunner AND used here to separately load the
                repaired 3-option template (see below) — kept as an
                explicit keyword-only param, rather than fished out of
                **kwargs, so this constructor fails loudly and immediately
                if it's missing, instead of failing confusingly later.
            prompt_version: Forwarded to ExperimentRunner and used the same
                way as prompts_dir above.
            stage1_lookup: question_id -> row dict with at least
                free_text_response, choice_a, choice_b, choice_c, choice_d.
                Must be the output of
                stage1_sources.load_and_validate_stage1() converted via
                .set_index("question_id").to_dict("index") (or equivalent) —
                i.e. already fail-closed validated. This runner does not
                re-validate contamination/completeness; that is
                stage1_sources.py's job, kept separate so this class stays
                a plain per-question ExperimentRunner.
            **kwargs: forwarded to ExperimentRunner (backend, method_name,
                split_name, run_id, temperature, max_tokens, seed,
                perturbation_name, model_label).
        """
        super().__init__(prompts_dir=prompts_dir, prompt_version=prompt_version, **kwargs)
        self._stage1_lookup = stage1_lookup
        # choicebench.methods.base.ExperimentRunner._prompts only loads the
        # 3 fixed built-in template names (direct_mcq, free_text,
        # option_matching). The repaired 3-option template is this
        # experiment's own addition, so it is loaded separately here rather
        # than by widening the shared base-class loader's fixed name list.
        repaired_template_path = (
            Path(prompts_dir) / prompt_version / f"{REPAIRED_3OPTION_TEMPLATE_NAME}.txt"
        )
        self._repaired_3option_template = repaired_template_path.read_text(encoding="utf-8")

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

        option_d_missing = _option_is_missing(question_row["choice_d"])
        if option_d_missing and question_id not in KNOWN_3OPTION_ARC_QUESTION_IDS:
            # Defense in depth: stage1_sources.load_and_validate_stage1()
            # should already have rejected this upstream. If a row somehow
            # reaches the runner with a missing 4th option outside the
            # audited repair set, refuse rather than silently rendering
            # "D. nan" (or an empty D line) for an unaudited question.
            raise ValueError(
                f"question_id={question_id!r} has a missing option_d but is "
                "not in KNOWN_3OPTION_ARC_QUESTION_IDS. Refusing to render "
                "either the historical 4-slot template (would produce "
                "'D. nan'/an empty D) or the repaired 3-option template "
                "(reserved for the audited repair set only) for an "
                "unaudited missing-option row."
            )

        if option_d_missing:
            # Audited repair path: exactly the 3 known 3-option ARC
            # questions. Renders A/B/C only — see repaired_stage2.py.
            matching_prompt = build_repaired_3option_matching_prompt(
                template=self._repaired_3option_template,
                question=question_row["question_text"],
                free_text=free_text_answer,
                option_a=question_row["choice_a"],
                option_b=question_row["choice_b"],
                option_c=question_row["choice_c"],
            )
        else:
            # Ordinary 4-option row: byte-faithful historical protocol.
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

    async def run_one_async(self, question_row: Any, sample_index: int) -> dict:
        """Async twin of run_one() — identical protocol, identical checks,
        identical result schema. The ONLY difference is how the backend is
        called: run_one() uses self._call_backend_generate(), a synchronous
        wrapper that calls self.backend.generate(prompt) directly — which
        works for DummyBackend/HuggingFaceBackend/MockBackend (all
        genuinely synchronous) but ALWAYS raises RuntimeError for
        choicebench's real APIBackend ("requires a running event loop").
        This was only discovered live, via a canary run against a real
        API backend (see group3_fourth_cell canary evidence) — every
        prior test used a synchronous mock, which never exercised this
        path. run_one() is kept exactly as it was (all of
        tests/experiments/test_runner.py's existing, passing coverage
        exercises it unchanged); this method exists alongside it for
        run_fourth_cell.py to call against real backends (API directly via
        backend.generate_single_async(); local via
        local_backend_adapter.LocalBackendAsyncAdapter, which gives
        HuggingFaceBackend the same async interface).
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

        option_d_missing = _option_is_missing(question_row["choice_d"])
        if option_d_missing and question_id not in KNOWN_3OPTION_ARC_QUESTION_IDS:
            raise ValueError(
                f"question_id={question_id!r} has a missing option_d but is "
                "not in KNOWN_3OPTION_ARC_QUESTION_IDS. Refusing to render "
                "either the historical 4-slot template (would produce "
                "'D. nan'/an empty D) or the repaired 3-option template "
                "(reserved for the audited repair set only) for an "
                "unaudited missing-option row."
            )

        if option_d_missing:
            matching_prompt = build_repaired_3option_matching_prompt(
                template=self._repaired_3option_template,
                question=question_row["question_text"],
                free_text=free_text_answer,
                option_a=question_row["choice_a"],
                option_b=question_row["choice_b"],
                option_c=question_row["choice_c"],
            )
        else:
            matching_prompt = build_option_matching_prompt(
                template=self._prompts["option_matching"],
                question=question_row["question_text"],
                free_text=free_text_answer,
                option_a=question_row["choice_a"],
                option_b=question_row["choice_b"],
                option_c=question_row["choice_c"],
                option_d=question_row["choice_d"],
            )
        matching_response = await self.backend.generate_single_async(matching_prompt)

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
