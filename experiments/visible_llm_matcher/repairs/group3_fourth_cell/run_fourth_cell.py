# experiments/visible_llm_matcher/repairs/group3_fourth_cell/run_fourth_cell.py
#
# Standalone driver for one (model, benchmark) cell of the fourth condition
# (options visible + LLM matcher). NOT wired through choicebench's own
# scripts/run_experiment.py CLI, deliberately: that CLI feeds question rows
# from ChoiceBench's own prepared benchmark artifact
# (config `benchmarks:` -> data/processed/<name>_normalized.csv), whose
# question_id is independently re-hashed and does NOT reliably agree with
# the immutable Stage 1 freeze this experiment must stay authoritative to
# (see runner.py's module docstring and arc_freeze_validation.py). Using
# the standard CLI path would silently break that design. This driver
# instead iterates over stage1_sources.load_and_validate_stage1()'s own
# DataFrame — exactly what runner.py's tests already exercise it with — as
# the question source, and reuses ChoiceBench's own backend construction
# (build_backend) and checkpoint manager (CheckpointManager, legacy
# calling convention) rather than reimplementing them.
#
# ChoiceBench's registry/method surface is untouched: VisibleLlmMatcherRunner
# is constructed directly in Python here, not looked up via
# METHOD_REGISTRY or "module.path:ClassName" resolution.
#
# Usage (NOT invoked by this config-generation phase):
#   python run_fourth_cell.py --config api/configs/gpt-4.1-mini__arc_challenge.yaml
#
# Config schema — see api/configs/*.yaml and local/configs/*.yaml for real
# examples:
#   run_id: str
#   model: {backend, provider?, model_name_or_path, generation_kwargs: {temperature, max_new_tokens}, concurrency_limit?, device?}
#   benchmark: "mmlu" | "arc_challenge"
#   seed: int
#   prompt_version: str
#   prompts_dir: str (relative to this ChoiceBench worktree root)
#   stage1_sources: [{repo, path, model_name, benchmark, is_local}]  (Stage1Source fields)
#   stage1_replacement_source: {repo, path, model_name, benchmark, is_local} | null
#   output_csv: str
#   checkpoint_dir: str

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from choicebench.config.schema import GenerationKwargsConfig, ModelConfig
from choicebench.infra.checkpoint import CheckpointManager
from choicebench.cli.run_experiment import build_backend

from experiments.visible_llm_matcher.arc_freeze_validation import validate_against_freeze
from experiments.visible_llm_matcher.stage1_sources import (
    KNOWN_3OPTION_ARC_QUESTION_IDS,
    Stage1Source,
    build_stage1_lookup,
    load_and_validate_stage1,
    load_replacement_rows,
)
from experiments.visible_llm_matcher.runner import VisibleLlmMatcherRunner

REPO_ROOT = Path(__file__).resolve().parents[4]
# Overridable via VLM_FREEZE_ROOT -- see harness/run_repair.py's identical
# override for why (Kelvin2 has no local model-generalization checkout at
# this sandbox's path; the freeze files are checksum-verified on every
# load regardless of where they're read from).
FREEZE_ROOT = Path(
    os.environ.get(
        "VLM_FREEZE_ROOT",
        "/home/cotenthusiast/Projects/model-generalization/paper_data_freeze/raw/local_model_generalization",
    )
)
PROMPTS_DIR = REPO_ROOT / "experiments" / "visible_llm_matcher" / "prompts"

EXPECTED_ROWS = 1000


def _load_source(entry: dict) -> Stage1Source:
    from experiments.visible_llm_matcher.source_path_rewrite import rewrite_source_path

    return Stage1Source(
        path=rewrite_source_path(entry["path"]),
        model_name=entry["model_name"],
        benchmark=entry["benchmark"],
        repo=entry["repo"],
    )


def load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def build_model_config(cfg: dict) -> ModelConfig:
    m = cfg["model"]
    gk = m.get("generation_kwargs", {})
    return ModelConfig(
        backend=m["backend"],
        model_name_or_path=m["model_name_or_path"],
        provider=m.get("provider"),
        device=m.get("device", "cuda"),
        generation_kwargs=GenerationKwargsConfig(
            temperature=gk.get("temperature", 0.0),
            max_new_tokens=gk.get("max_new_tokens", 500),
            do_sample=gk.get("do_sample", False),
        ),
        concurrency_limit=m.get("concurrency_limit"),
    )


def prepare_stage1(cfg: dict) -> pd.DataFrame:
    """Load + validate this cell's reused Stage-1 rows. Fails closed
    (raises) on incomplete/contaminated input — see stage1_sources.py.
    For ARC, additionally re-validates all 1000 rows against the immutable
    Stage 1 freeze (arc_freeze_validation.py) before returning.
    """
    sources = [_load_source(e) for e in cfg["stage1_sources"]]
    replacement_rows = None
    if cfg.get("stage1_replacement_source"):
        rep_source = _load_source(cfg["stage1_replacement_source"])
        rep_df = load_replacement_rows(rep_source.path)
        rep_df = rep_df[rep_df["question_id"].isin(KNOWN_3OPTION_ARC_QUESTION_IDS)].copy()
        replacement_rows = rep_df

    df = load_and_validate_stage1(sources, replacement_rows=replacement_rows)

    if cfg["benchmark"] == "arc_challenge":
        validate_against_freeze(df, FREEZE_ROOT)

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Stage-1 input has {len(df)} rows, expected exactly {EXPECTED_ROWS}."
        )
    return df


async def run_cell_async(config_path: Path, *, canary_limit: int | None = None) -> None:
    """
    Args:
        canary_limit: if set, only the first N Stage-1 rows (still drawn
            from the full, freeze-validated 1000-row Stage-1 input — the
            validation itself is never skipped) are actually sent through
            the runner. Output goes to a CANARY_-prefixed path under
            group3_fourth_cell/CANARY_outputs/, and the checkpoint uses a
            run_id suffixed "_CANARY" — both deliberately distinct from
            the real run_id/output_csv so a canary can never be mistaken
            for, merged into, or accidentally imported as the real
            1000-row artifact.

    Async: VisibleLlmMatcherRunner.run_one_async() is used (not run_one())
    because choicebench's real APIBackend.generate() always raises
    RuntimeError ("requires a running event loop") -- discovered live via
    the very first canary run against gpt-4.1-mini, not assumed. Local
    (HuggingFaceBackend) calls are wrapped in LocalBackendAsyncAdapter so
    both paths share one call site.
    """
    cfg = load_config(config_path)
    run_id = cfg["run_id"]
    benchmark = cfg["benchmark"]
    model_name = cfg["model"]["model_name_or_path"]

    stage1_df = prepare_stage1(cfg)
    stage1_lookup = build_stage1_lookup(stage1_df)

    model_config = build_model_config(cfg)
    is_canary = canary_limit is not None
    effective_run_id = f"{run_id}_CANARY" if is_canary else run_id
    raw_backend = build_backend(
        model_config=model_config,
        run_id=effective_run_id,
        run_seed=cfg["seed"],
        default_concurrency_limit=cfg["model"].get("concurrency_limit") or 10,
        model_identity=f"{effective_run_id}__{model_name.replace('/', '_')}",
    )
    if model_config.backend == "huggingface":
        from experiments.visible_llm_matcher.local_backend_adapter import LocalBackendAsyncAdapter

        backend = LocalBackendAsyncAdapter(raw_backend)
    else:
        backend = raw_backend

    runner = VisibleLlmMatcherRunner(
        backend=backend,
        method_name="visible_llm_matcher",
        split_name="robustness",
        prompt_version=cfg["prompt_version"],
        prompts_dir=Path(cfg["prompts_dir"]),
        run_id=effective_run_id,
        temperature=model_config.generation_kwargs.temperature,
        max_tokens=model_config.generation_kwargs.max_new_tokens,
        seed=cfg["seed"],
        model_label=model_name,
        stage1_lookup=stage1_lookup,
    )

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=(
            (REPO_ROOT / "experiments" / "visible_llm_matcher" / "repairs" / "group3_fourth_cell" / "CANARY_outputs" / "checkpoints")
            if is_canary
            else (REPO_ROOT / cfg["checkpoint_dir"])
        ),
        run_id=effective_run_id,
        condition="visible_llm_matcher",
        model=model_name,
        benchmark=benchmark,
    )
    state = checkpoint_mgr.load()
    completed_ids = set(state["completed_ids"]) if state else set()
    accumulated: list[dict] = state["results"] if state else []

    rows = stage1_df.to_dict(orient="records")
    if is_canary:
        rows = rows[:canary_limit]
    remaining = [r for r in rows if r["question_id"] not in completed_ids]

    expected = len(rows) if is_canary else EXPECTED_ROWS
    print(f"{effective_run_id}: {len(completed_ids)}/{expected} already complete, {len(remaining)} remaining.")

    stage1_calls_made = 0  # this driver never calls a backend for Stage 1 — always 0, asserted below
    stage2_calls_made = 0
    for i, row in enumerate(remaining):
        result = await runner.run_one_async(row, sample_index=i)
        stage2_calls_made += 1
        accumulated.append(result)
        completed_ids.add(row["question_id"])
        if (i + 1) % 50 == 0:
            checkpoint_mgr.save(list(completed_ids), accumulated, started_at="")

    checkpoint_mgr.save(list(completed_ids), accumulated, started_at="")

    if is_canary:
        failed = [r for r in accumulated if r.get("transport_status") != "success"]
        if failed:
            details = "; ".join(
                f"{r['question_id']}: {r.get('error_type')} - {r.get('error_message')}"
                for r in failed
            )
            raise RuntimeError(
                f"{len(failed)}/{len(accumulated)} canary rows failed for "
                f"{effective_run_id!r} — refusing to report a successful "
                f"canary built from failed generations: {details}"
            )
        out_dir = REPO_ROOT / "experiments" / "visible_llm_matcher" / "repairs" / "group3_fourth_cell" / "CANARY_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"CANARY_{config_path.stem}_{canary_limit}row.csv"
    else:
        out_path = REPO_ROOT / cfg["output_csv"]
        out_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(accumulated).to_csv(out_path, index=False)
    print(f"Wrote {len(accumulated)} rows to {out_path}")
    print(f"stage1_calls_made={stage1_calls_made} stage2_calls_made={stage2_calls_made}")


def run_cell(config_path: Path, *, canary_limit: int | None = None) -> None:
    """Sync entry point — wraps run_cell_async in asyncio.run()."""
    import asyncio

    asyncio.run(run_cell_async(config_path, canary_limit=canary_limit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--canary-limit",
        type=int,
        default=None,
        help="If set, only process this many Stage-1 rows and write to an "
        "isolated CANARY_-prefixed output/checkpoint that can't be confused "
        "with the real 1000-row run.",
    )
    args = parser.parse_args()
    run_cell(args.config, canary_limit=args.canary_limit)
