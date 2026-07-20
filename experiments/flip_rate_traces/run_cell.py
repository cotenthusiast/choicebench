# experiments/flip_rate_traces/run_cell.py
#
# Full-cell driver with checkpoint/resume. Iterates the frozen 1000-question
# set for one (model, benchmark) cell, executes the trace-retaining cyclic
# runner per question, and writes results to a fresh output directory (never
# into the immutable 142-artifact bundle or any existing run/ directory).
#
# Usage:
#   python -m experiments.flip_rate_traces.run_cell --config configs/flip_rate_traces/<name>.yaml

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from choicebench.infra.checkpoint import CheckpointManager  # noqa: E402

from experiments.flip_rate_traces.data_source import load_frozen_questions  # noqa: E402
from experiments.flip_rate_traces.runner import TraceRetainingCyclicRunner  # noqa: E402
from experiments.flip_rate_traces.trace_schema import TRACE_COLUMNS, validate_question_traces  # noqa: E402

CACHE_NAMESPACE = "flip_rate_traces_v1"
OUTPUT_ROOT = REPO_ROOT / "runs" / "flip_rate_traces"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "flip_rate_traces"
FRESH_CACHE_DIR = REPO_ROOT / ".cache" / "flip_rate_traces_v1"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if v and k not in os.environ:
            os.environ[k] = v


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    tmp.replace(path)


async def _run_api(cfg: dict, runner: TraceRetainingCyclicRunner, q, sample_index: int) -> list[dict]:
    return await runner.run_one_async(q, sample_index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N questions (testing)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    backend_kind = cfg["backend"]  # "api" | "local"
    cell_id = cfg["cell_id"]
    model_name = cfg["model_name"]
    provider = cfg["provider"]
    benchmark = cfg["benchmark"]
    split_name = cfg.get("split_name", "robustness")
    run_id = cfg["run_id"]

    questions = load_frozen_questions(cell_id)
    if args.limit:
        questions = questions[: args.limit]

    output_dir = OUTPUT_ROOT / run_id
    output_path = output_dir / f"{run_id}_{cell_id}_traced.csv"
    ckpt = CheckpointManager(
        checkpoint_dir=CHECKPOINT_ROOT, run_id=run_id, condition="cyclic_generation_majority_traced_v1",
        model=model_name, benchmark=benchmark,
    )
    state = ckpt.load() or {"completed_ids": [], "results": [], "started_at": datetime.now(timezone.utc).isoformat()}
    completed = set(state["completed_ids"])
    all_rows: list[dict] = state["results"]

    remaining = [q for q in questions if q.question_id not in completed]
    print(f"[run_cell] {cell_id}: {len(completed)} already done, {len(remaining)} remaining")

    if backend_kind == "api":
        _load_env_file(REPO_ROOT.parent / "two-stage-prompting" / ".env")
        from choicebench.backends.api_backend import APIBackend
        from choicebench.clients.openai_client import OpenAIClient

        client = OpenAIClient(model_name=model_name)
        backend = APIBackend(
            provider=provider, model_name=model_name, client=client,
            cache_dir=FRESH_CACHE_DIR, temperature=0.0, max_tokens=500, seed=42,
            cache_identity=CACHE_NAMESPACE,
        )
        runner = TraceRetainingCyclicRunner(
            backend=backend, model_name=model_name, provider=provider,
            benchmark=benchmark, split_name=split_name,
            cell_id=f"{cell_id}_traced_v1", run_id=run_id,
        )

        async def _drive():
            for i, q in enumerate(remaining):
                rows = await runner.run_one_async(q, sample_index=i)
                validate_question_traces(q.question_id, rows)
                all_rows.extend(rows)
                completed.add(q.question_id)
                if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
                    ckpt.save(list(completed), all_rows, state["started_at"])
                    _write_csv(all_rows, output_path)
                    print(f"[run_cell] {cell_id}: {len(completed)}/{len(questions)} questions done")

        asyncio.run(_drive())
    else:
        from choicebench.backends.hf_backend import HuggingFaceBackend

        backend = HuggingFaceBackend(model_name, device=cfg.get("device", "cuda"))
        backend.load()
        runner = TraceRetainingCyclicRunner(
            backend=backend, model_name=model_name, provider=provider,
            benchmark=benchmark, split_name=split_name,
            cell_id=f"{cell_id}_traced_v1", run_id=run_id,
        )
        for i, q in enumerate(remaining):
            rows = runner.run_one_sync(q, sample_index=i)
            validate_question_traces(q.question_id, rows)
            all_rows.extend(rows)
            completed.add(q.question_id)
            if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
                ckpt.save(list(completed), all_rows, state["started_at"])
                _write_csv(all_rows, output_path)
                print(f"[run_cell] {cell_id}: {len(completed)}/{len(questions)} questions done")

    _write_csv(all_rows, output_path)
    if len(completed) == len(questions):
        ckpt.delete()
    print(f"[run_cell] DONE: {output_path} ({len(all_rows)} trace rows, {len(completed)} questions)")


if __name__ == "__main__":
    main()
