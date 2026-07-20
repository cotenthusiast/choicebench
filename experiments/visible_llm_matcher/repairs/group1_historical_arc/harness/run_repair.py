# experiments/visible_llm_matcher/repairs/group1_historical_arc/harness/run_repair.py
#
# CLI driver: given a cell_id from manifest_summary.json, run the
# appropriate method repair for its 3 audited question_ids, verify
# freshness, and write a NEW staged 1000-row artifact (997 keep rows from
# the seeded checkpoint + 3 freshly-repaired rows) — never overwriting the
# historical CSV or the immutable freeze.
#
# API keys are read from process environment variables only (OPENAI_API_KEY
# etc.) — this script never reads, prints, copies, or writes any .env file
# or secret value anywhere. Source two-stage-prompting's .env into your
# shell environment before running (`set -a; source .env; set +a`) rather
# than pointing this script at a file.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from choicebench.registry import CLIENT_REGISTRY

from experiments.visible_llm_matcher.arc_freeze_validation import load_frozen_arc_dataset
from experiments.visible_llm_matcher.stage1_sources import KNOWN_3OPTION_ARC_QUESTION_IDS

from .baseline_repair import repair_one_baseline_question
from .cyclic_repair import repair_one_cyclic_question
from .repair_infra import (
    PROVENANCE_ROOT,
    REPAIR_CACHE_ROOT,
    STAGING_ROOT,
    ProtocolDiffRecord,
    assert_cache_grew_by,
    build_repair_backend,
    count_cache_entries,
)
from .stage1_reuse_sources import load_reused_free_text, resolve_stage1_source_path
from .text_extraction_repair import repair_one_text_extraction_question
from .two_stage_repair import STAGE1_REUSE_DECISION, STAGE1_REUSE_REASON, repair_one_two_stage_question

import os

REPO_ROOT = Path(__file__).resolve().parents[5]
MANIFEST_PATH = REPO_ROOT / "experiments" / "visible_llm_matcher" / "repairs" / "group1_historical_arc" / "manifest_summary.json"
# Overridable via VLM_FREEZE_ROOT (e.g. on Kelvin2, where the freeze lives
# at a synced copy, not the local sandbox's model-generalization checkout
# path) -- the freeze files themselves are checksum-verified on every load
# (arc_freeze_validation.py), so a copy at a different path is only ever
# trusted if it is byte-identical to the original.
FREEZE_ROOT = Path(
    os.environ.get(
        "VLM_FREEZE_ROOT",
        "/home/cotenthusiast/Projects/model-generalization/paper_data_freeze/raw/local_model_generalization",
    )
)

_CALLS_PER_QUESTION = {
    "baseline": 1,
    "cyclic_generation_majority": 3,  # repaired: 1 call per REAL option, not the historical 4
    "text_extraction": 1,
    "two_stage_v1": 1,  # Stage 2 only — Stage 1 reused, not called
    "two_stage_v2": 1,
    "two_stage_v3": 1,
}

_ENV_KEY_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
}


class MissingApiKeyError(RuntimeError):
    pass


def _load_manifest_cell(cell_id: str) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    matches = [c for c in manifest if c["cell_id"] == cell_id]
    if not matches:
        raise ValueError(f"cell_id {cell_id!r} not found in {MANIFEST_PATH}")
    return matches[0]


def _build_client(provider: str, model_name: str, concurrency_limit: int):
    env_key = _ENV_KEY_BY_PROVIDER.get(provider)
    api_key = os.environ.get(env_key) if env_key else None
    if not api_key:
        raise MissingApiKeyError(
            f"{env_key} is not set in the process environment. This provider "
            "will be skipped — export it (e.g. `set -a; source "
            "/home/cotenthusiast/Projects/two-stage-prompting/.env; set +a`) "
            "before retrying just this provider."
        )
    client_cls = CLIENT_REGISTRY[provider]
    return client_cls(model_name=model_name, concurrency_limit=concurrency_limit)


def _three_repaired_rows(model_name: str) -> list[dict]:
    """The 3 audited questions' clean content (question_text, real A/B/C,
    correct_option, subject) from the checksummed immutable freeze — the
    most authoritative source available, independent of any historical
    repo's own (possibly contaminated) copy.
    """
    frozen = load_frozen_arc_dataset(FREEZE_ROOT)
    rows = []
    for qid in sorted(KNOWN_3OPTION_ARC_QUESTION_IDS):
        r = frozen.loc[qid]
        rows.append(
            {
                "question_id": qid,
                "subject": r["subject"],
                "question_text": r["question_text"],
                "choice_a": r["choice_a"],
                "choice_b": r["choice_b"],
                "choice_c": r["choice_c"],
                "correct_option": r["correct_option"],
                "model_name": model_name,
            }
        )
    return rows


async def run_repair_for_cell_async(cell_id: str) -> dict:
    cell = _load_manifest_cell(cell_id)
    method = cell["method"]
    job_method = cell["job_matrix_method"]
    model_name = cell["model"]
    is_local = cell["is_local"]

    cfg_path = REPO_ROOT / "experiments" / "visible_llm_matcher" / "repairs" / "group1_historical_arc" / "configs" / f"{cell_id}.yaml"
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())
    model_cfg = cfg["models"][model_name]
    temperature = cfg["run"]["temperature"]
    max_tokens = cfg["run"]["max_tokens"]
    seed = cfg["run"]["seed"]

    cache_namespace = f"{method}__{model_name.replace('/', '_')}"
    n_before = 0  # local path has no cache layer at all — see below

    if is_local:
        # Local/Kelvin2: HuggingFaceBackend has a completely different
        # (sync, no-cache) interface — see local_backend_adapter.py. Must
        # run on a GPU node (Kelvin2 sbatch), not this driver's own host.
        from choicebench.backends.hf_backend import HuggingFaceBackend

        from experiments.visible_llm_matcher.local_backend_adapter import LocalBackendAsyncAdapter

        hf_backend = HuggingFaceBackend(
            model_name_or_path=model_cfg["model_path"],
            device=model_cfg.get("device", "cuda"),
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=False,
        )
        hf_backend.load()
        backend = LocalBackendAsyncAdapter(hf_backend)
    else:
        provider = model_cfg["provider"]
        concurrency = model_cfg.get("concurrency", 10)
        client = _build_client(provider, model_name, concurrency)
        backend = build_repair_backend(
            provider=provider,
            model_name=model_name,
            client=client,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            concurrency_limit=concurrency,
            cache_namespace=cache_namespace,
        )
        n_before = count_cache_entries(cache_namespace)

    rows = _three_repaired_rows(model_name)

    repaired: list[dict] = []
    stage1_disposition = "not_applicable"
    stage1_reason = "Method has no separate hidden-options Stage 1 (single-call or visible-options-only method)."

    if job_method == "baseline":
        for row in rows:
            repaired.append(await repair_one_baseline_question(backend, row))
    elif job_method == "cyclic":
        for row in rows:
            repaired.append(await repair_one_cyclic_question(backend, row))
    elif job_method == "text_extraction":
        for row in rows:
            repaired.append(await repair_one_text_extraction_question(backend, row))
    elif job_method == "two_prompt":
        stage1_disposition = STAGE1_REUSE_DECISION
        stage1_reason = STAGE1_REUSE_REASON
        for row in rows:
            reused_ft = load_reused_free_text(method, model_name, row["question_id"])
            repaired.append(
                await repair_one_two_stage_question(
                    backend, row, reused_free_text_response=reused_ft, method_name=method
                )
            )
    else:
        raise ValueError(f"Unhandled job_matrix_method: {job_method!r}")

    n_calls_per_question = _CALLS_PER_QUESTION[method]
    expected_new_cache_entries = n_calls_per_question * len(rows)
    if is_local:
        # HuggingFaceBackend has no caching layer at all (every call is a
        # genuine forward pass) -- the cache-file-growth check is
        # meaningless here, not skipped out of laziness. Freshness for
        # local calls is instead evidenced by positive latency (checked
        # per-response by assert_fresh_call already) and, for a real GPU
        # run, plausible multi-second-per-call latency reflecting actual
        # model inference.
        pass
    else:
        assert_cache_grew_by(cache_namespace, before=n_before, expected_new_entries=expected_new_cache_entries)

    failed = [r for r in repaired if r["model_status"] != "success"]
    if failed:
        details = "; ".join(
            f"{r['question_id']}: {r.get('error_type')} - {r.get('error_message')}"
            for r in failed
        )
        raise RuntimeError(
            f"{len(failed)}/{len(repaired)} repair calls failed for "
            f"cell_id={cell_id!r} — refusing to stage a repaired artifact "
            f"built from failed generations: {details}"
        )

    # Merge with the 997 seeded "keep" rows into a NEW staged artifact.
    checkpoint_seed_path = REPO_ROOT / cell["generated_checkpoint_seed"]
    checkpoint = json.loads(checkpoint_seed_path.read_text())
    keep_rows = checkpoint["results"]

    staged_df = pd.concat([pd.DataFrame(keep_rows), pd.DataFrame(repaired)], ignore_index=True)
    staging_dir = STAGING_ROOT / cell_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staging_dir / f"{cell_id}_repaired_1000.csv"
    staged_df.to_csv(staged_path, index=False)

    source_config_ref = resolve_stage1_source_path(method, model_name) if job_method == "two_prompt" else Path(cell["source_config"])

    provenance = ProtocolDiffRecord(
        cell_id=cell_id,
        method=method,
        model=model_name,
        benchmark="arc_challenge",
        source_repo="model-generalization" if is_local else "two-stage-prompting",
        source_commit=(
            "24403d8da9c7c11189bf56439b4040bc7cad1efd"
            if is_local
            else "b8e784f3eb5d2a727a97eb675140b383a34584fa"
        ),
        source_config_path=cell["source_config"],
        source_run_id=cell["run_id"],
        prompt_template_historical=_historical_template_name(job_method),
        prompt_template_repaired=_repaired_template_name(job_method),
        repaired_question_ids=sorted(KNOWN_3OPTION_ARC_QUESTION_IDS),
        correction_description=(
            "Real option count only (A/B/C) for the 3 audited 3-option ARC "
            "questions; never render/parse/permute/match a synthetic D. nan."
        ),
        stage1_disposition=stage1_disposition,
        stage1_disposition_reason=stage1_reason,
    )
    provenance_path = provenance.write()

    return {
        "cell_id": cell_id,
        "staged_output": str(staged_path),
        "provenance": str(provenance_path),
        "n_keep_rows": len(keep_rows),
        "n_repaired_rows": len(repaired),
        "total_rows": len(staged_df),
        "cache_namespace": cache_namespace,
        "cache_entries_created": expected_new_cache_entries,
    }


def run_repair_for_cell(cell_id: str) -> dict:
    """Sync entry point — wraps run_repair_for_cell_async in asyncio.run()
    for CLI/test callers that aren't already inside an event loop."""
    import asyncio as _asyncio

    return _asyncio.run(run_repair_for_cell_async(cell_id))


def _historical_template_name(job_method: str) -> str:
    return {
        "baseline": "prompts/v1/direct_mcq.txt",
        "cyclic": "prompts/v1/direct_mcq.txt",
        "text_extraction": "prompts/v1/text_extraction.txt",
        "two_prompt": "prompts/v1/option_matching.txt (Stage 2); free_text.txt (Stage 1, reused unchanged)",
    }[job_method]


def _repaired_template_name(job_method: str) -> str:
    return {
        "baseline": "experiments/visible_llm_matcher/prompts/v1/direct_mcq_repaired_3option.txt",
        "cyclic": "experiments/visible_llm_matcher/prompts/v1/direct_mcq_repaired_3option.txt",
        "text_extraction": "experiments/visible_llm_matcher/prompts/v1/text_extraction_repaired_3option.txt",
        "two_prompt": "experiments/visible_llm_matcher/prompts/v1/option_matching_repaired_3option.txt (Stage 2 only)",
    }[job_method]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    args = parser.parse_args()
    result = run_repair_for_cell(args.cell_id)
    print(json.dumps(result, indent=2))
