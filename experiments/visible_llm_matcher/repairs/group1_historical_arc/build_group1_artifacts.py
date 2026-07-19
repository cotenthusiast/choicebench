# experiments/visible_llm_matcher/repairs/group1_historical_arc/build_group1_artifacts.py
#
# Config-and-execution-preparation only. Makes ZERO network/API/model calls.
# Reads already-materialized historical CSVs and config.yaml SNAPSHOTS
# (read-only; nothing in two-stage-prompting or model-generalization is
# written to) and writes, into THIS ChoiceBench worktree only:
#   - a scoped run config per cell (job matrix trimmed to exactly one
#     (model, method, benchmark) job, everything else byte-identical to the
#     historical run's own frozen config.yaml snapshot)
#   - a seeded checkpoint per cell (997 "keep" rows marked completed with
#     their full original row data; the 3 contaminated question_ids
#     deliberately NOT marked completed, so TSP's/MG's own
#     run_experiment.py --run-id <same run_id> --yes will process ONLY
#     those 3 on resume)
#
# The checkpoint format (completed_ids, results, started_at,
# last_checkpoint_at) matches twoprompt.infra.checkpoint.CheckpointManager /
# modelgen's equivalent exactly (verified against source, see both repos'
# src/*/infra/checkpoint.py — byte-identical schema). These seeded
# checkpoints are STAGED here, not copied into place — see
# group1_historical_arc/README.md and the top-level repair report for the
# exact target path each must be copied to before running
# scripts/run_experiment.py, and note that the 4 local cells' target is a
# Kelvin2 path this machine cannot write to directly.
#
# Scope: the 28 cells in IN_SCOPE_CELLS below were derived by filtering
# model-generalization/paper_data_freeze/rerun_specs/*.json (32 files) to:
#   benchmark == "arc_challenge"
#   AND method in {baseline, cyclic_generation_majority, text_extraction,
#                   two_stage_v1, two_stage_v2, two_stage_v3}
#   AND (model in the 4 API models
#        OR (model, method) in the 4 explicit local (model, two_stage_v2/v3) pairs)
# This excludes exactly 4 rerun_specs: 1 independent_hypothesis (gemini),
# 2 mmlu (gemini baseline/cyclic — wrong benchmark), 1 pride (Qwen-Turbo) —
# see build_and_verify_scope() below, which re-derives and asserts this
# count every time the script runs rather than trusting the hardcoded list
# alone.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

FREEZE_ROOT = Path("/home/cotenthusiast/Projects/model-generalization/paper_data_freeze")
RERUN_SPECS_DIR = FREEZE_ROOT / "rerun_specs"

THIS_DIR = Path(__file__).resolve().parent
CONFIGS_OUT_DIR = THIS_DIR / "configs"
CHECKPOINTS_OUT_DIR = THIS_DIR / "checkpoint_seeds"

_APPROVED_METHODS = {
    "baseline",
    "cyclic_generation_majority",
    "text_extraction",
    "two_stage_v1",
    "two_stage_v2",
    "two_stage_v3",
}
_API_MODELS = {
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "llama-3.1-8b-instant",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
}
_LOCAL_TARGETS = {
    ("Qwen/Qwen2.5-7B-Instruct", "two_stage_v2"),
    ("Qwen/Qwen2.5-7B-Instruct", "two_stage_v3"),
    ("meta-llama/Llama-3.1-8B-Instruct", "two_stage_v2"),
    ("meta-llama/Llama-3.1-8B-Instruct", "two_stage_v3"),
}

# Historical-method name (as used in rerun_specs) -> job-matrix method key
# actually used inside config.yaml `run.jobs[].methods` (verified against
# scripts/run_experiment.py::_METHOD_TO_RUNNER in both repos).
_METHOD_TO_JOB_MATRIX_KEY = {
    "baseline": "baseline",
    "cyclic_generation_majority": "cyclic",
    "text_extraction": "text_extraction",
    "two_stage_v1": "two_prompt",
    "two_stage_v2": "two_prompt",
    "two_stage_v3": "two_prompt",
}

EXPECTED_IN_SCOPE_COUNT = 28


class ScopeError(RuntimeError):
    pass


def build_and_verify_scope() -> list[dict]:
    """Re-derive the 28 in-scope cells from the rerun_specs directory and
    assert the count, rather than trusting a hardcoded list alone.
    """
    specs = [json.loads(p.read_text()) for p in sorted(RERUN_SPECS_DIR.glob("*.json"))]
    in_scope = []
    for d in specs:
        if d["benchmark"] != "arc_challenge":
            continue
        key = (d["model"], d["method"])
        if d["model"] in _API_MODELS and d["method"] in _APPROVED_METHODS:
            in_scope.append(d)
        elif key in _LOCAL_TARGETS:
            in_scope.append(d)
    if len(in_scope) != EXPECTED_IN_SCOPE_COUNT:
        raise ScopeError(
            f"Expected {EXPECTED_IN_SCOPE_COUNT} in-scope cells, derived "
            f"{len(in_scope)}. Do not proceed until this is reconciled."
        )
    return in_scope


@dataclass(frozen=True, slots=True)
class ResolvedCellPaths:
    csv_path: Path
    config_path: Path
    is_local: bool


def _resolve_paths(spec: dict) -> ResolvedCellPaths:
    """Pick the locally-readable (non-'kelvin2:'-prefixed) raw-run CSV and
    config.yaml snapshot from a rerun_spec's path lists. Raises if none of
    the listed candidates exist on this machine.
    """
    is_local = spec["provider_backend"] == "local_huggingface"

    csv_candidates = [p for p in spec["supporting_artifact_paths"] if not p.startswith("kelvin2:")]
    # Prefer the raw run-folder copy (contains config.yaml alongside it),
    # not the eval_ready/organized derived copies.
    csv_candidates.sort(key=lambda p: 0 if "/runs/" in p else 1)
    csv_path = next((Path(p) for p in csv_candidates if Path(p).is_file()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"No locally-readable CSV found for {spec['cell_id']} among {csv_candidates}"
        )

    cfg_candidates = [p for p in spec["source_configuration"] if not p.startswith("kelvin2:")]
    config_path = next((Path(p) for p in cfg_candidates if Path(p).is_file()), None)
    if config_path is None:
        raise FileNotFoundError(
            f"No locally-readable config.yaml found for {spec['cell_id']} among {cfg_candidates}"
        )

    return ResolvedCellPaths(csv_path=csv_path, config_path=config_path, is_local=is_local)


def _safe_model(model_name: str) -> str:
    return model_name.replace("/", "_")


def build_scoped_config(spec: dict, source_config_path: Path) -> dict:
    """Load the historical run's own frozen config.yaml snapshot and trim
    `run.jobs` to exactly one entry for this cell — everything else
    (models:, temperature/max_tokens/seed/prompt_version, paths:) is left
    byte-identical to what actually produced the historical data, so the
    repair uses the exact same generation settings by construction, not by
    transcription.
    """
    cfg = yaml.safe_load(source_config_path.read_text())
    job_method = _METHOD_TO_JOB_MATRIX_KEY[spec["method"]]
    scoped_job = {
        "model": spec["model"],
        "methods": [job_method],
        "benchmark": spec["benchmark"],
        "split": "robustness",
    }
    cfg["run"]["jobs"] = [scoped_job]
    return cfg


def build_seeded_checkpoint(csv_path: Path, exact_question_ids: list[str]) -> tuple[dict, int, int]:
    """Split the historical CSV into keep/repair rows and build the
    checkpoint payload.

    Returns:
        (checkpoint_dict, n_keep, n_repair)
    """
    df = pd.read_csv(csv_path)
    if len(df) != 1000:
        raise ValueError(f"{csv_path}: expected 1000 rows, found {len(df)}")

    repair_ids = set(exact_question_ids)
    keep_df = df[~df["question_id"].isin(repair_ids)]
    repair_df = df[df["question_id"].isin(repair_ids)]
    if len(repair_df) != len(repair_ids):
        raise ValueError(
            f"{csv_path}: expected {len(repair_ids)} repair rows, found {len(repair_df)}"
        )
    if len(keep_df) != 1000 - len(repair_ids):
        raise ValueError(f"{csv_path}: keep-row count mismatch after split")

    # JSON-safe: NaN -> None, everything else passed through as native types.
    records = json.loads(keep_df.to_json(orient="records"))

    checkpoint = {
        "completed_ids": keep_df["question_id"].tolist(),
        "results": records,
        "started_at": "2026-07-19T00:00:00+00:00",
        "last_checkpoint_at": "2026-07-19T00:00:00+00:00",
    }
    return checkpoint, len(keep_df), len(repair_df)


def build_all() -> list[dict]:
    """Generate every Group 1 config + checkpoint seed. Returns a list of
    per-cell summary dicts for the report (no side effects beyond writing
    files under CONFIGS_OUT_DIR / CHECKPOINTS_OUT_DIR).
    """
    CONFIGS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    in_scope = build_and_verify_scope()
    summaries = []

    for spec in in_scope:
        paths = _resolve_paths(spec)
        job_method = _METHOD_TO_JOB_MATRIX_KEY[spec["method"]]

        scoped_cfg = build_scoped_config(spec, paths.config_path)
        config_out_path = CONFIGS_OUT_DIR / f"{spec['cell_id']}.yaml"
        config_out_path.write_text(yaml.safe_dump(scoped_cfg, sort_keys=False))

        checkpoint, n_keep, n_repair = build_seeded_checkpoint(
            paths.csv_path, spec["exact_question_ids"]
        )
        run_id = paths.csv_path.parent.name
        checkpoint_out_dir = CHECKPOINTS_OUT_DIR / run_id
        checkpoint_out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_filename = f"{job_method}__{_safe_model(spec['model'])}__{spec['benchmark']}.json"
        checkpoint_out_path = checkpoint_out_dir / checkpoint_filename
        checkpoint_out_path.write_text(json.dumps(checkpoint, indent=2))

        target_checkpoint_rel = (
            scoped_cfg["paths"]["checkpoints_dir"] + f"/{run_id}/{checkpoint_filename}"
        )

        summaries.append(
            {
                "cell_id": spec["cell_id"],
                "model": spec["model"],
                "method": spec["method"],
                "job_matrix_method": job_method,
                "provider_backend": spec["provider_backend"],
                "is_local": paths.is_local,
                "run_id": run_id,
                "n_keep": n_keep,
                "n_repair": n_repair,
                "source_csv": str(paths.csv_path),
                "source_config": str(paths.config_path),
                "generated_config": str(config_out_path.relative_to(THIS_DIR.parent.parent.parent.parent)),
                "generated_checkpoint_seed": str(checkpoint_out_path.relative_to(THIS_DIR.parent.parent.parent.parent)),
                "target_checkpoint_path_relative_to_repo_root": target_checkpoint_rel,
            }
        )

    return summaries


if __name__ == "__main__":
    summaries = build_all()
    print(f"Generated {len(summaries)} scoped configs + checkpoint seeds.")
    total_repair = sum(s["n_repair"] for s in summaries)
    total_keep = sum(s["n_keep"] for s in summaries)
    print(f"Total repair calls: {total_repair}  (expected 84 = 28 x 3)")
    print(f"Total keep rows carried through checkpoints: {total_keep}")
    (THIS_DIR / "manifest_summary.json").write_text(json.dumps(summaries, indent=2))
