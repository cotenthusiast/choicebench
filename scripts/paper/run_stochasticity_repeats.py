# scripts/paper/run_stochasticity_repeats.py
#
# Paper-specific (eacl-2026-revision): the frozen stochasticity protocol --
# for the 100-MMLU / 100-ARC frozen question subsets
# (data/manifests/{mmlu,arc_challenge}_stochasticity_v2.csv), API models
# only, runs N observations per question, in canonical (unrotated) option
# order. Methods: baseline (direct_mcq), two_stage_v1 (two_stage),
# reasoning_mcq, reasoning_two_stage.
#
# RESOLVED (was previously flagged as an unresolved scientific question --
# see git history for the earlier "UNRESOLVED SCIENTIFIC QUESTION" version
# of this comment): observation 0 REUSES the main accuracy run's own
# already-collected result for that question (passed in via
# --canonical-obs0-csv), rather than making a fresh call. Observations
# 1..n_repetitions-1 are fully fresh, independent calls. This matches the
# frozen spec's stated total of 14,400 additional calls: 200 questions x 4
# API models x (1+2+1+2=6 calls/question across the 4 methods) x 3 FRESH
# repetitions = 14,400 (not 19,200, which would be 4 fresh repetitions).
# If the canonical artifact is missing entirely, or doesn't cover a
# question this run needs observation 0 for, this script fails loudly
# (raises ValueError) rather than silently generating a fresh replacement
# -- see _load_canonical_obs0_rows().
#
# For two_stage_v1 and reasoning_two_stage, every FRESH repetition (1..
# n_repetitions-1) is still a COMPLETE independent Stage1->Stage2 repeat --
# stage 1 is never reused across repetitions (unlike the flip-rate rotation
# scripts, which deliberately reuse stage 1 across rotations of the SAME
# call). Observation 0's reused row already carries whatever stage-1/
# stage-2 provenance the main accuracy run recorded for it.
#
# Batch policy: direct_mcq/reasoning_mcq FRESH repetitions are batch-safe
# (every repetition's call is independently constructible up front -- it's
# the same canonical prompt repeated, not a dependency chain) and default
# to execution_mode="batch". two_stage/reasoning_two_stage FRESH
# repetitions must stay synchronous (each repetition's own stage-1/stage-2
# dependency) -- call with execution_mode="sync" for those. Either way
# this always goes through run_many_async(): for a plain "sync" APIBackend
# that's ordinary concurrent dispatch (never a real provider Batch API job
# -- no multi-wave batching is built or used here), for a "batch"
# BatchAPIBackend it's a real batch job. TwoStageRunner.run_many_async's
# own two generate_batch() calls per repetition (stage 1 wave, then stage
# 2 wave) are therefore ordinary concurrent HTTP dispatch under "sync",
# not two provider batch jobs.
#
# Each FRESH repetition gets a DISTINCT model_identity (-> distinct
# cache_dir / batch_state_dir), so repetitions never share a cache bucket.
# This is not an optimization -- without it, repetition 2/3 would silently
# return repetition 1's cached response for the byte-identical canonical
# prompt, measuring nothing. Resumable across (question_id,
# repetition_index) pairs, one repetition's batch of pending questions at
# a time.
#
# ⚠️ OPEN DESIGN QUESTION, deliberately NOT resolved here (flagging per
# Karl's instruction rather than guessing at a scientific protocol change)
# -- investigated 2026-09-25 (see clients/openrouter_client.py:142-143,
# clients/together_client.py:134-135/174-175 vs. openai_client.py,
# anthropic_client.py, deepinfra_client.py, none of which reference
# request.seed at all): ONLY the OpenRouter (Llama sync) and Together
# (Qwen sync+batch) clients forward ModelRequest.seed to the provider's
# own API call; OpenAI/Anthropic/DeepInfra never send a seed (Anthropic's
# Messages API has no seed parameter at all; OpenAI's Chat Completions API
# does support one but this codebase's OpenAIClient never wires it).
# Separately: this script passes the SAME run_seed (e.g. 42) to
# build_backend() for every repetition -- only model_identity varies, not
# run_seed -- so APIBackend._seed (and therefore the seed value actually
# sent over the wire) is IDENTICAL across all fresh repetitions for
# whichever providers DO forward it. Net effect: for Llama-sync and Qwen,
# every fresh repetition requests the SAME seed=42 from the provider; for
# GPT/Claude/Llama-batch, no seed is ever sent at all. Whether stochastic
# repeatability should be measured under a FIXED seed sent identically
# every time (current behavior, preserved as-is), an OMITTED seed, or a
# VARIED seed per repetition is a genuine scientific protocol question
# with more than one defensible answer -- NOT changed here. This is
# orthogonal to (and does not compromise) the model_identity-based cache
# isolation above: a "fresh" repetition here always means a genuinely new
# API call was made (never served from a prior repetition's cache), even
# on the providers where that new call happens to request the same seed
# as the previous one.
#
# The output file mixes two row shapes: observation 0's reused rows carry
# whatever condition_metadata columns the main accuracy run's grid
# execution stamped on them (experiment_id, condition_id, ...), while
# freshly computed rows (1..n_repetitions-1) don't. Rather than a single
# incremental DictWriter with one fixed header (which can't hold both),
# writes go through a pandas-concat-based rewrite of the whole (small,
# <=400-row) file -- see _append_rows_with_schema_union().
#
# Takes a plain, full-content questions CSV (same shape
# run_text_extraction_rotations.py takes) -- not the raw 2-column
# question_id/subject manifest at data/manifests/mmlu_stochasticity_v2.csv
# directly. Export it once per benchmark with:
#   load_benchmark_selection(benchmark_cfg, run_seed).questions.to_csv(...)
# using a benchmark config whose question_id_manifest points at the
# frozen stochasticity manifest (data/manifests/{mmlu,arc_challenge}_
# stochasticity_v2.csv), so the exported rows are exactly that 100-question
# subset with full content (question_text, choices_json, correct_option).
#
# --canonical-obs0-csv is that same method/model's own saved main accuracy
# run output CSV -- the file run_experiment.py (or the relevant standalone
# script) already wrote for this method_name/model/benchmark combination.
#
# --model-config takes an EXISTING grid config purely as a model source
# (same pattern every other standalone script here uses) -- reuse
# config/paper/mmlu_core_methods.yaml (API models 0-3: openai, anthropic,
# openrouter-pinned-Llama, together) for execution_mode=sync repeats, or
# config/paper/mmlu_core_methods_cyclic_batch.yaml (same 4 API models,
# but index 2 is direct-deepinfra-fp8-Turbo Llama) for execution_mode=
# batch repeats -- no dedicated stochasticity config file needed. Swap in
# the arc_* equivalents for ARC.
#
# Run with (baseline/reasoning_mcq repeats, batch-safe):
#   python scripts/paper/run_stochasticity_repeats.py \
#       --questions-csv <exported mmlu stochasticity questions CSV> \
#       --canonical-obs0-csv runs/<main_run_id>/direct_mcq_<model>_mmlu.csv \
#       --model-config config/paper/mmlu_core_methods_cyclic_batch.yaml --model-index 0 \
#       --method-name direct_mcq --prompt-version v1 \
#       --run-id <id> --output runs/<id>/stochasticity_direct_mcq_<model>_mmlu.csv \
#       --execution-mode batch
#
# Run with (two_stage/reasoning_two_stage repeats, must stay synchronous):
#   python scripts/paper/run_stochasticity_repeats.py \
#       --questions-csv <exported mmlu stochasticity questions CSV> \
#       --canonical-obs0-csv runs/<main_run_id>/two_stage_<model>_mmlu.csv \
#       --model-config config/paper/mmlu_core_methods.yaml --model-index 0 \
#       --method-name two_stage --prompt-version v1 \
#       --run-id <id> --output runs/<id>/stochasticity_two_stage_<model>_mmlu.csv \
#       --execution-mode sync

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from choicebench.cli.run_experiment import build_backend
from choicebench.config.schema import load_config
from choicebench.infra.resumable_csv import check_resume_compatible
from choicebench.registry import METHOD_REGISTRY


def _load_completed_pairs(output_path: Path) -> set[tuple[str, int]]:
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    if "question_id" not in existing.columns or "repetition_index" not in existing.columns:
        return set()
    return set(zip(
        existing["question_id"].astype(str), existing["repetition_index"].astype(int),
    ))


# two_stage/reasoning_two_stage's own stage-1/stage-2 dependency means no
# multi-wave batching is built for them (see module header) -- running
# them under execution_mode="batch" wouldn't corrupt data, but it would
# silently violate the frozen policy and submit real, unbuilt-for provider
# batch jobs for a dependency chain this script never designed batching
# around. direct_mcq/reasoning_mcq are batch-safe either way (every call
# is independently constructible), so only the sync-only side is enforced.
_SYNC_ONLY_METHODS = frozenset({"two_stage", "reasoning_two_stage"})


def _validate_execution_mode(method_name: str, execution_mode: str) -> None:
    if method_name in _SYNC_ONLY_METHODS and execution_mode != "sync":
        raise ValueError(
            f"method_name={method_name!r} has its own stage-1/stage-2 dependency "
            "per repetition and must run under execution_mode='sync' (the frozen "
            "batch policy: two_stage/reasoning_two_stage stay synchronous) -- got "
            f"execution_mode={execution_mode!r}. No multi-wave batching is built "
            "for this dependency chain; running it under 'batch' would silently "
            "submit unintended real provider batch jobs."
        )


_OBS0_IDENTITY_COLUMNS = ("provider", "model_name", "benchmark_name", "prompt_version")
_OBS0_NUMERIC_IDENTITY_COLUMNS = ("temperature", "max_tokens")


def _load_canonical_obs0_rows(
        canonical_obs0_csv: Path, method_name: str, question_ids: list[str],
        *, expected_identity: dict[str, object],
) -> list[dict]:
    """Load observation 0's rows by REUSING the main accuracy run's own
    saved result for method_name -- never generating a fresh replacement.
    Fails loudly (instead of silently falling back to a fresh call) if the
    canonical artifact is missing entirely, is missing any of the
    question_ids this run needs observation 0 for, has no row for a
    question_id under this exact method_name, contains AMBIGUOUS/
    conflicting duplicate rows for a needed question_id (never silently
    resolved via keep="first"), or the matched rows' own recorded
    provider/model/benchmark/prompt/generation-settings identity doesn't
    match what THIS stochasticity run itself is about to use -- e.g. a
    canonical Llama row saved under the sync OpenRouter->DeepInfra route
    must never be reused as observation 0 for a batch DeepInfra-direct
    stochasticity run, since those are different deployments even though
    "Llama" is the same nominal model name.

    Args:
        expected_identity: this run's own {"provider", "model_name",
            "benchmark_name", "prompt_version", "temperature",
            "max_tokens"} -- every matched canonical row must agree with
            all of these exactly.
    """
    if not canonical_obs0_csv.exists():
        raise ValueError(
            f"Canonical observation-0 artifact {canonical_obs0_csv} does not "
            "exist -- observation 0 must reuse the main accuracy run's own "
            "saved result for this method/model, never a freshly generated "
            "replacement. Run the main accuracy run for this method/model "
            "first, or pass the correct --canonical-obs0-csv path."
        )
    canonical_df = pd.read_csv(canonical_obs0_csv)
    required_columns = ("method_name", "question_id") + _OBS0_IDENTITY_COLUMNS + _OBS0_NUMERIC_IDENTITY_COLUMNS
    missing_columns = [c for c in required_columns if c not in canonical_df.columns]
    if missing_columns:
        raise ValueError(
            f"Canonical observation-0 artifact {canonical_obs0_csv} is missing "
            f"required identity column(s) {missing_columns} -- cannot verify it "
            f"is the correct method/model/benchmark/prompt/generation-settings "
            "result for this run."
        )
    canonical_df = canonical_df[canonical_df["method_name"].astype(str) == str(method_name)]

    # Reject ambiguous duplicate rows for a needed question -- never
    # silently resolved via keep="first". A canonical artifact should have
    # exactly one row per question_id under this method_name; more than
    # one means the file is corrupt, was concatenated from incompatible
    # runs, or otherwise can't be trusted to pick "the" right one.
    dup_counts = canonical_df["question_id"].astype(str).value_counts()
    ambiguous_needed = sorted(
        qid for qid in dup_counts[dup_counts > 1].index if qid in question_ids
    )
    if ambiguous_needed:
        raise ValueError(
            f"Canonical observation-0 artifact {canonical_obs0_csv} has more "
            f"than one row for method_name={method_name!r} for question_id(s) "
            f"{ambiguous_needed[:5]} -- refusing to silently pick one among "
            "conflicting candidates. Deduplicate the canonical artifact first."
        )

    canonical_df = canonical_df.set_index(canonical_df["question_id"].astype(str), drop=False)

    missing = [qid for qid in question_ids if qid not in canonical_df.index]
    if missing:
        raise ValueError(
            f"Canonical observation-0 artifact {canonical_obs0_csv} is missing "
            f"{len(missing)} question_id(s) required for observation 0 of "
            f"method_name={method_name!r} (e.g. {missing[:5]}) -- refusing to "
            "silently generate a fresh replacement for observation 0."
        )

    needed_rows = canonical_df.loc[question_ids]
    for column in _OBS0_IDENTITY_COLUMNS:
        expected = str(expected_identity[column])
        mismatched = needed_rows[needed_rows[column].astype(str) != expected]
        if len(mismatched) > 0:
            bad_qids = sorted(mismatched["question_id"].astype(str).unique())[:5]
            raise ValueError(
                f"Canonical observation-0 artifact {canonical_obs0_csv} has "
                f"{column}={sorted(mismatched[column].astype(str).unique())} for "
                f"question_id(s) {bad_qids}, but this stochasticity run expects "
                f"{column}={expected!r} -- refusing to reuse an observation from "
                "a different model/provider/deployment/benchmark/prompt as "
                "observation 0."
            )
    for column in _OBS0_NUMERIC_IDENTITY_COLUMNS:
        expected_val = float(expected_identity[column])
        mismatched = needed_rows[needed_rows[column].astype(float) != expected_val]
        if len(mismatched) > 0:
            bad_qids = sorted(mismatched["question_id"].astype(str).unique())[:5]
            raise ValueError(
                f"Canonical observation-0 artifact {canonical_obs0_csv} has "
                f"{column}={sorted(mismatched[column].unique())} for question_id(s) "
                f"{bad_qids}, but this stochasticity run expects "
                f"{column}={expected_val!r} -- refusing to reuse an observation "
                "generated under different generation settings as observation 0."
            )

    return [canonical_df.loc[qid].to_dict() for qid in question_ids]


def _append_rows_with_schema_union(output_path: Path, new_rows: list[dict], file_exists: bool) -> int:
    """Reused observation-0 rows (carrying extra condition_metadata
    columns from the main grid run) and freshly computed rows (which
    don't) are legitimately heterogeneous -- a single incremental
    DictWriter with one fixed header can't hold both. Rewriting the whole
    (small, <=400-row) file via pandas keeps every column any row needs,
    filling the rest with NaN, and stays resumable across process restarts
    the same as before.
    """
    new_df = pd.DataFrame(new_rows)
    if file_exists:
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    else:
        combined_df = new_df
    combined_df.to_csv(output_path, index=False)
    return len(new_df)


def run(
        questions_csv: Path,
        model_config,
        run_id: str,
        output_path: Path,
        method_name: str,
        prompt_version: str,
        canonical_obs0_csv: Path,
        n_repetitions: int = 4,
        run_seed: int = 42,
        resume: bool = True,
        execution_mode: str = "batch",
) -> int:
    """Run n_repetitions observations of method_name over every row in
    questions_csv: observation 0 is reused from canonical_obs0_csv,
    observations 1..n_repetitions-1 are fresh, independent calls.

    Returns:
        Number of new (question_id, repetition_index) rows written
        (excludes pairs already present on resume).

    Raises:
        ValueError: if execution_mode is incompatible with method_name --
            see _validate_execution_mode().
    """
    _validate_execution_mode(method_name, execution_mode)
    source_df = pd.read_csv(questions_csv)
    if "question_text" not in source_df.columns:
        raise ValueError(
            f"{questions_csv} has no 'question_text' column -- expected an "
            "exported benchmark questions CSV."
        )

    runner_cls = METHOD_REGISTRY[method_name]
    completed_pairs = _load_completed_pairs(output_path) if resume else set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    if file_exists:
        check_resume_compatible(
            output_path, required_columns=["question_id", "repetition_index"],
            expected_method_name=method_name,
        )

    new_rows: list[dict] = []

    # Observation 0: reused from the canonical main-run artifact, never a
    # fresh call.
    pending_mask_0 = ~source_df["question_id"].astype(str).apply(
        lambda qid: (qid, 0) in completed_pairs
    )
    pending_df_0 = source_df[pending_mask_0]
    if len(pending_df_0) > 0:
        expected_identity = {
            "provider": model_config.provider,
            "model_name": model_config.model_name_or_path,
            "benchmark_name": source_df["benchmark_name"].iloc[0] if len(source_df) else "",
            "prompt_version": prompt_version,
            "temperature": model_config.generation_kwargs.temperature,
            "max_tokens": model_config.generation_kwargs.max_new_tokens,
        }
        obs0_rows = _load_canonical_obs0_rows(
            canonical_obs0_csv, method_name, pending_df_0["question_id"].astype(str).tolist(),
            expected_identity=expected_identity,
        )
        for row in obs0_rows:
            row["repetition_index"] = 0
        new_rows.extend(obs0_rows)

    # Observations 1..n_repetitions-1: fully fresh, independent calls --
    # for two_stage/reasoning_two_stage, a complete Stage1->Stage2 repeat
    # every time, never mixing stages across repetitions.
    for repetition_index in range(1, n_repetitions):
        pending_mask = ~source_df["question_id"].astype(str).apply(
            lambda qid: (qid, repetition_index) in completed_pairs
        )
        pending_df = source_df[pending_mask].reset_index(drop=True)
        if len(pending_df) == 0:
            continue

        backend = build_backend(
            model_config, run_id, run_seed=run_seed, execution_mode=execution_mode,
            model_identity=f"stochasticity_{method_name}_rep{repetition_index}",
        )
        runner = runner_cls(
            backend=backend, method_name=method_name, split_name="test",
            prompt_version=prompt_version, prompts_dir=Path("prompts"), run_id=run_id,
            seed=run_seed, benchmark_name=source_df["benchmark_name"].iloc[0] if len(source_df) else "",
            temperature=model_config.generation_kwargs.temperature,
            max_tokens=model_config.generation_kwargs.max_new_tokens,
        )

        results = asyncio.run(runner.run_many_async(pending_df))
        for result in results:
            result["repetition_index"] = repetition_index
        new_rows.extend(results)

    if not new_rows:
        return 0

    return _append_rows_with_schema_union(output_path, new_rows, file_exists)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-csv", required=True, type=Path)
    parser.add_argument("--canonical-obs0-csv", required=True, type=Path,
                         help="The main accuracy run's own saved result CSV for this "
                              "method/model -- observation 0 reuses its rows rather than "
                              "making a fresh call.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--method-name", required=True,
                         choices=["direct_mcq", "two_stage", "reasoning_mcq", "reasoning_two_stage"])
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--n-repetitions", type=int, default=4)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--execution-mode", choices=["batch", "sync"], default=None,
                         help="Defaults to the frozen policy's own mode for --method-name "
                              "if omitted (sync for two_stage/reasoning_two_stage, batch "
                              "otherwise) -- an explicit value incompatible with "
                              "--method-name is rejected, never silently honored.")
    args = parser.parse_args()

    config = load_config(str(args.model_config))
    model_config = config.models[args.model_index]

    execution_mode = args.execution_mode
    if execution_mode is None:
        execution_mode = "sync" if args.method_name in _SYNC_ONLY_METHODS else "batch"

    n_written = run(
        args.questions_csv, model_config, args.run_id, args.output,
        method_name=args.method_name, prompt_version=args.prompt_version,
        canonical_obs0_csv=args.canonical_obs0_csv,
        n_repetitions=args.n_repetitions, run_seed=args.seed,
        resume=not args.no_resume, execution_mode=execution_mode,
    )
    print(f"Wrote {n_written} new rows -> {args.output}")


if __name__ == "__main__":
    main()
