# experiments/visible_llm_matcher/stage1_sources.py
#
# Reused Stage-1 (options-visible, free-text) source registry and fail-closed
# validation for the "options visible + LLM matcher" missing cell.
#
# Design constraint (explicit user decision): this experiment must NOT
# generate a second Stage-1 sample. It reuses the free-text answers already
# produced by the historical text_extraction condition and adds only a new
# Stage-2 LLM-matcher call. This module is the boundary that enforces that:
# it loads pre-existing text_extraction result CSVs from two-stage-prompting
# and model-generalization (read-only, external reference repos — never
# written to) and refuses (raises) rather than silently proceeding whenever
# the input is incomplete or known-contaminated.
#
# Verified source files (row counts / contamination checked directly against
# the CSVs, not assumed):
#
#   API models (gpt-4.1-mini, gemini-2.5-flash, llama-3.1-8b-instant,
#   Qwen/Qwen2.5-7B-Instruct-Turbo), both benchmarks:
#     two-stage-prompting/paper_results/eval_ready/paper_api_main/
#       20260603_154649_text_extraction_<model>_mmlu.csv           (1000 rows, clean)
#       20260603_154649_text_extraction_<model>_arc_challenge.csv  (1000 rows,
#         3 rows have choice_d == NaN: question_id in
#         KNOWN_3OPTION_ARC_QUESTION_IDS — see contamination note below)
#
#   Local models (Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct),
#   MMLU only:
#     two-stage-prompting/paper_results/eval_ready/paper_local_main/
#       20260603_213613_text_extraction_Qwen_Qwen2.5-7B-Instruct_mmlu.csv            (1000 rows, clean)
#       20260603_214857_text_extraction_meta-llama_Llama-3.1-8B-Instruct_mmlu.csv    (1000 rows, clean)
#
#   Local models, ARC-Challenge: TSP's own paper_local_main ARC files
#   (20260603_213613/214857 ARC variants) are STALE — only 850 rows — and
#   must NOT be used (confirmed by direct read). The corrected, complete
#   1000-row source is a model-generalization run instead:
#     model-generalization/runs/20260617_162624/
#       20260617_162624_text_extraction_Qwen_Qwen2.5-7B-Instruct_arc_challenge.csv          (1000 rows)
#       20260617_162624_text_extraction_meta-llama_Llama-3.1-8B-Instruct_arc_challenge.csv  (1000 rows)
#     Both still contain the same 3 NaN-choice_d rows (KNOWN_3OPTION_ARC_QUESTION_IDS)
#     BUT — verified by reading the stored `prompt` column directly — MG's own
#     text_extraction runner correctly drops the missing option before
#     rendering Stage 1 (no "D. nan" in the Stage-1 prompt text), unlike TSP's
#     API path, which always interpolates all four option slots. So MG's
#     Stage-1 elicitation for these 3 rows is *not* contaminated the way TSP's
#     API Stage-1 elicitation is; nonetheless this module treats all 6 models
#     identically at the loader boundary (fail closed unless a replacement is
#     explicitly supplied) so the fix path is uniform and auditable rather
#     than silently trusting one source and not another. See
#     KNOWN_3OPTION_ARC_QUESTION_IDS.
#
# Contamination, precisely: these 3 ARC-Challenge robustness-split questions
# have only 3 real options (choice_d is genuinely absent from the source
# data, not a data error). TSP's text_extraction Stage-1 prompt-building code
# (twoprompt.pipeline.prompt_builder.build_text_extraction_prompt) always
# interpolates option_a..option_d, so the model was shown a literal "D. nan"
# 4th option when the API Stage-1 free-text answer for these 3 questions was
# elicited — a genuine prompt defect that may have influenced what the model
# answered, not a rendering artifact fixable after the fact. Reusing that
# tainted free-text answer for the new cell would carry the defect forward.
# The fix belongs at Stage 1 (regenerate a correctly-3-option Stage-1 prompt
# for exactly these rows), which is out of scope for this phase — see
# ReplacementStage1Loader below and the top-level report for how corrected
# rows will be supplied later.
#
# NOT in scope here: writing run configs or calling any backend. This module
# only loads and validates already-materialized CSVs from disk.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# The 3 ARC-Challenge robustness-split questions confirmed (by direct CSV
# read) to have only 3 real options, across every model and every historical
# source checked (TSP API text_extraction, TSP API two_prompt Stage 2, MG
# local text_extraction). Any OTHER row with a NaN choice_d is an unknown
# condition this loader has not been validated against, and is treated as a
# hard failure rather than silently accepted as "probably fine".
KNOWN_3OPTION_ARC_QUESTION_IDS: frozenset[str] = frozenset(
    {
        "79e8c959bbeb74a0",
        "ad6b5d46ae54842c",
        "c30e75b011696a95",
    }
)

EXPECTED_ROWS_PER_MODEL_BENCHMARK = 1000

REQUIRED_COLUMNS = (
    "question_id",
    "question_text",
    "choice_a",
    "choice_b",
    "choice_c",
    "choice_d",
    "correct_option",
    "free_text_response",
)

# TSP's own API text_extraction CSVs (TextExtractionRunner._build_result_row)
# store the Stage-1 free-text answer under raw_text only — there is no
# free_text_response column at all (verified against
# paper_results/eval_ready/paper_api_main/20260603_154649_text_extraction_*.csv).
# MG's local text_extraction CSVs have BOTH raw_text and an explicit
# free_text_response copy of the same value (verified against
# runs/20260617_162624/*.csv). _load_csv() below derives free_text_response
# from raw_text when the column is missing, rather than requiring every
# historical source to already have it — REQUIRED_COLUMNS still lists
# free_text_response because by the time _load_csv() returns, every row
# must have one; the derivation just means "missing column" isn't
# automatically fatal the way any other missing required column is.


class Stage1ValidationError(RuntimeError):
    """Raised when reused Stage-1 input fails a fail-closed check.

    Deliberately a plain RuntimeError subclass (not silently caught anywhere
    in this package) — any caller that wants to proceed past a validation
    failure must explicitly catch this and justify why.
    """


@dataclass(frozen=True, slots=True)
class Stage1Source:
    """One text_extraction result CSV, with enough metadata to validate it."""

    path: Path
    model_name: str
    benchmark: str  # "mmlu" | "arc_challenge"
    repo: str  # "two-stage-prompting" | "model-generalization" — provenance only


def _derive_free_text_response(df: pd.DataFrame) -> pd.DataFrame:
    """See REQUIRED_COLUMNS comment above: TSP's own API text_extraction
    CSVs have no free_text_response column, only raw_text. Derives it when
    missing so both TSP's and MG's CSV shapes are accepted identically.
    """
    if "free_text_response" not in df.columns and "raw_text" in df.columns:
        df = df.copy()
        df["free_text_response"] = df["raw_text"]
    return df


def load_replacement_rows(path: Path) -> pd.DataFrame:
    """Load a candidate replacement-rows CSV (e.g. a repaired
    text_extraction output) with the same free_text_response derivation
    _load_csv() applies to primary sources — for callers (like
    repairs/group3_fourth_cell/run_fourth_cell.py) that read a replacement
    CSV directly rather than through a Stage1Source.
    """
    if not path.is_file():
        raise Stage1ValidationError(f"Replacement rows file does not exist: {path}")
    return _derive_free_text_response(pd.read_csv(path))


def _load_csv(source: Stage1Source) -> pd.DataFrame:
    if not source.path.is_file():
        raise Stage1ValidationError(
            f"Stage-1 source file does not exist: {source.path} "
            f"(model={source.model_name!r}, benchmark={source.benchmark!r}, "
            f"repo={source.repo!r}). This loader never fabricates missing "
            "input — supply the correct historical path."
        )
    df = _derive_free_text_response(pd.read_csv(source.path))
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise Stage1ValidationError(
            f"{source.path} is missing required column(s) {missing_cols}. "
            "This is not a text_extraction-shaped Stage-1 CSV."
        )
    df = df.copy()
    df["_source_repo"] = source.repo
    df["_source_path"] = str(source.path)
    return df


def load_and_validate_stage1(
    sources: list[Stage1Source],
    *,
    replacement_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load one model+benchmark's reused text_extraction Stage-1 rows.

    Fails closed (raises Stage1ValidationError) if, after applying any
    supplied replacement_rows:
      - any row still has a NaN choice_d, OR
      - the row count for this (model, benchmark) is not exactly
        EXPECTED_ROWS_PER_MODEL_BENCHMARK, OR
      - question_id is not unique, OR
      - free_text_response is null/empty on any row (Stage 1 failed for
        that question in the historical run — this loader will not
        silently treat a missing free-text answer as a Stage-2 input).

    Args:
        sources: One or more Stage1Source entries for the SAME
            (model_name, benchmark) pair (a list to allow a future
            replacement source to be passed alongside the original, though
            the current call sites pass one primary source plus
            replacement_rows for the 3-row ARC fix — see module docstring).
        replacement_rows: Optional DataFrame with the same REQUIRED_COLUMNS
            schema, indexed by question_id, to substitute in place of known-
            contaminated rows (see KNOWN_3OPTION_ARC_QUESTION_IDS) before
            validation runs. Must not itself contain a NaN choice_d for a
            genuinely-4-option question — that would indicate a bad
            replacement rather than a legitimate 3-option question.

    Returns:
        The validated, concatenated Stage-1 DataFrame for this
        (model, benchmark) pair — always exactly EXPECTED_ROWS_PER_MODEL_BENCHMARK
        rows, one per question_id, with a non-null free_text_response.
    """
    if not sources:
        raise Stage1ValidationError("load_and_validate_stage1() called with no sources.")

    model_names = {s.model_name for s in sources}
    benchmarks = {s.benchmark for s in sources}
    if len(model_names) != 1 or len(benchmarks) != 1:
        raise Stage1ValidationError(
            f"All sources passed together must share one (model, benchmark); "
            f"got models={model_names}, benchmarks={benchmarks}."
        )
    model_name = next(iter(model_names))
    benchmark = next(iter(benchmarks))

    frames = [_load_csv(s) for s in sources]
    df = pd.concat(frames, ignore_index=True)

    dupes = df["question_id"][df["question_id"].duplicated()].unique()
    if len(dupes):
        raise Stage1ValidationError(
            f"{model_name}/{benchmark}: {len(dupes)} duplicate question_id "
            f"value(s) in reused Stage-1 input (e.g. {dupes[0]!r}). Refusing "
            "to guess which row is authoritative."
        )

    if replacement_rows is not None and not replacement_rows.empty:
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in replacement_rows.columns]
        if missing_cols:
            raise Stage1ValidationError(
                f"replacement_rows is missing required column(s) {missing_cols}."
            )
        bad_replacements = replacement_rows[
            replacement_rows["choice_d"].isna()
            & ~replacement_rows["question_id"].isin(KNOWN_3OPTION_ARC_QUESTION_IDS)
        ]
        if not bad_replacements.empty:
            raise Stage1ValidationError(
                "replacement_rows contains NaN choice_d for question_id(s) not "
                f"in KNOWN_3OPTION_ARC_QUESTION_IDS: "
                f"{bad_replacements['question_id'].tolist()}. A replacement "
                "row for a genuinely-4-option question must supply all 4 "
                "options."
            )
        replacement_rows = replacement_rows.copy()
        replacement_rows["_source_repo"] = "corrected_replacement"
        replacement_rows["_source_path"] = "<supplied at call time>"

        df = df[~df["question_id"].isin(set(replacement_rows["question_id"]))]
        df = pd.concat([df, replacement_rows], ignore_index=True)

    # Structural checks (choice_d present/absent) cannot by themselves tell
    # "already repaired" apart from "still contaminated": a replacement_rows
    # source that is the SAME underlying file as the primary source (e.g.
    # an in-place repair before it has actually landed) would otherwise
    # pass — self-referential NaN choice_d looks identical either way. The
    # phantom-D contamination has a known, literal textual signature in the
    # stored Stage-1 prompt ("D. nan"), so check for it directly wherever a
    # prompt column exists, for every known-3-option-id row regardless of
    # whether it came from a replacement or the primary source.
    if "prompt" in df.columns:
        known_id_rows = df[df["question_id"].isin(KNOWN_3OPTION_ARC_QUESTION_IDS)]
        still_phantom_d = known_id_rows[
            known_id_rows["prompt"].astype(str).str.contains("D. nan", regex=False)
        ]
        if not still_phantom_d.empty:
            raise Stage1ValidationError(
                f"{model_name}/{benchmark}: {len(still_phantom_d)} row(s) "
                f"still show the literal phantom-D contamination signature "
                f"('D. nan') in their stored Stage-1 prompt text "
                f"(question_id(s) {still_phantom_d['question_id'].tolist()}), "
                "even though a replacement was supplied. The replacement "
                "source has not actually been repaired yet — do not "
                "proceed with this input."
            )

    # A NaN choice_d from a "corrected_replacement" row is a legitimate
    # 3-option question (already checked above via bad_replacements) and is
    # not contamination. Only a NaN choice_d still coming from an original,
    # unreplaced source row counts as "still contaminated".
    still_contaminated = df[df["choice_d"].isna() & (df["_source_repo"] != "corrected_replacement")]
    if not still_contaminated.empty:
        ids = still_contaminated["question_id"].tolist()
        unknown_ids = [i for i in ids if i not in KNOWN_3OPTION_ARC_QUESTION_IDS]
        if unknown_ids:
            raise Stage1ValidationError(
                f"{model_name}/{benchmark}: unexpected NaN choice_d for "
                f"question_id(s) not in KNOWN_3OPTION_ARC_QUESTION_IDS: "
                f"{unknown_ids}. This loader only tolerates the 3 known "
                "3-option ARC questions, and only via an explicit "
                "replacement — investigate before proceeding."
            )
        raise Stage1ValidationError(
            f"{model_name}/{benchmark}: {len(ids)} row(s) still have the "
            f"known contaminated Stage-1 free-text answer (question_id(s) "
            f"{ids}). Supply replacement_rows with corrected Stage-1 "
            "free-text answers for these IDs before running the Stage-2 "
            "matcher — this experiment must not silently reuse a free-text "
            "answer elicited while the model was shown a bogus 'D. nan' "
            "option."
        )

    if len(df) != EXPECTED_ROWS_PER_MODEL_BENCHMARK:
        raise Stage1ValidationError(
            f"{model_name}/{benchmark}: expected exactly "
            f"{EXPECTED_ROWS_PER_MODEL_BENCHMARK} rows, got {len(df)}. "
            "Refusing to run the missing cell on an incomplete input set "
            "(e.g. do not silently fall back to a 997-row subset)."
        )

    missing_free_text = df[
        df["free_text_response"].isna()
        | (df["free_text_response"].astype(str).str.strip() == "")
    ]
    if not missing_free_text.empty:
        raise Stage1ValidationError(
            f"{model_name}/{benchmark}: {len(missing_free_text)} row(s) have "
            f"a null/empty free_text_response (Stage 1 failed in the "
            f"historical run) for question_id(s) "
            f"{missing_free_text['question_id'].tolist()[:5]}... . This "
            "experiment reuses Stage 1 as-is and does not regenerate it."
        )

    return df.sort_values("question_id").reset_index(drop=True)


def build_stage1_lookup(df: pd.DataFrame) -> dict[str, dict]:
    """Convert a load_and_validate_stage1() DataFrame into the
    question_id -> row dict shape runner.VisibleLlmMatcherRunner expects
    for its stage1_lookup constructor argument.

    Pure reshaping — no validation here; df must already be the output of
    load_and_validate_stage1(). Separated out (rather than folded into
    load_and_validate_stage1() or the runner itself) so config-driven
    execution (a YAML pointing at CSV paths) and direct/programmatic
    construction (an in-memory DataFrame, as all of this experiment's
    existing tests use) share one conversion path without either one
    depending on the other's calling convention.
    """
    return df.set_index("question_id", drop=False).to_dict(orient="index")
