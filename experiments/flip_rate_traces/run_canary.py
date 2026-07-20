# experiments/flip_rate_traces/run_canary.py
#
# Real-backend canary driver. Writes ONLY to runs/flip_rate_traces/_canary/
# (never to a cell's authoritative output directory) and stamps every row
# with is_diagnostic_canary=True, so canary output can never be mistaken for
# or accidentally merged into authoritative run output.
#
# Usage:
#   python -m experiments.flip_rate_traces.run_canary --cell <cell_id> --question-kind normal
#   python -m experiments.flip_rate_traces.run_canary --cell <cell_id> --question-kind damaged_arc
#
# "damaged_arc" selects one of the 3 known 3-real-option ARC questions
# (KNOWN_3OPTION_ARC_QUESTION_IDS); only valid for an arc_challenge cell.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.flip_rate_traces.data_source import load_frozen_questions  # noqa: E402
from experiments.flip_rate_traces.historical_protocol import (  # noqa: E402
    KNOWN_3OPTION_ARC_QUESTION_IDS,
    historical_majority_vote,
)
from experiments.flip_rate_traces.runner import TraceRetainingCyclicRunner  # noqa: E402
from experiments.flip_rate_traces.trace_schema import validate_question_traces  # noqa: E402

CANARY_OUTPUT_DIR = REPO_ROOT / "runs" / "flip_rate_traces" / "_canary"
FRESH_CACHE_DIR = REPO_ROOT / ".cache" / "flip_rate_traces_v2_exact_historical_prompt"
CACHE_NAMESPACE = "flip_rate_traces_v2_exact_historical_prompt"


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


def _select_question(cell_id: str, benchmark: str, kind: str):
    questions = load_frozen_questions(cell_id)
    if kind == "damaged_arc":
        if benchmark != "arc_challenge":
            raise ValueError("damaged_arc is only meaningful for arc_challenge cells")
        for q in questions:
            if q.question_id in KNOWN_3OPTION_ARC_QUESTION_IDS:
                return q
        raise ValueError("No known 3-option ARC question found in frozen source")
    for q in questions:
        if q.question_id not in KNOWN_3OPTION_ARC_QUESTION_IDS:
            return q
    raise ValueError("No ordinary (4-option) question found")


def _cache_file_count(cache_dir: Path) -> int:
    if not cache_dir.exists():
        return 0
    return sum(1 for _ in cache_dir.rglob("*.json"))


async def run_api_canary(cell_id: str, model_name: str, benchmark: str, split_name: str, kind: str) -> list[dict]:
    _load_env_file(REPO_ROOT.parent / "two-stage-prompting" / ".env")
    from choicebench.backends.api_backend import APIBackend
    from choicebench.clients.openai_client import OpenAIClient

    q = _select_question(cell_id, benchmark, kind)
    before = _cache_file_count(FRESH_CACHE_DIR)

    client = OpenAIClient(model_name=model_name)
    backend = APIBackend(
        provider="openai",
        model_name=model_name,
        client=client,
        cache_dir=FRESH_CACHE_DIR,
        temperature=0.0,
        max_tokens=500,
        seed=42,
        cache_identity=CACHE_NAMESPACE,
    )
    runner = TraceRetainingCyclicRunner(
        backend=backend, model_name=model_name, provider="openai",
        benchmark=benchmark, split_name=split_name,
        cell_id=f"{cell_id}_traced_v2_CANARY", run_id="canary",
    )
    rows = await runner.run_one_async(q, sample_index=0, is_diagnostic_canary=True)
    after = _cache_file_count(FRESH_CACHE_DIR)
    print(f"[canary] cache files before={before} after={after} (expect after > before for a fresh call)")
    return rows


def run_local_canary(cell_id: str, model_name: str, benchmark: str, split_name: str, kind: str, device: str) -> list[dict]:
    from choicebench.backends.hf_backend import HuggingFaceBackend

    q = _select_question(cell_id, benchmark, kind)
    backend = HuggingFaceBackend(model_name, device=device)
    backend.load()
    runner = TraceRetainingCyclicRunner(
        backend=backend, model_name=model_name, provider="huggingface",
        benchmark=benchmark, split_name=split_name,
        cell_id=f"{cell_id}_traced_v2_CANARY", run_id="canary",
    )
    return runner.run_one_sync(q, sample_index=0, is_diagnostic_canary=True)


def verify_and_report(rows: list[dict], q_id: str) -> None:
    validate_question_traces(q_id, rows)
    recomputed = historical_majority_vote([r["semantic_parsed_choice"] for r in rows])
    runner_vote = rows[0]["majority_semantic_choice"]
    assert recomputed == runner_vote, f"recomputed vote {recomputed!r} != runner vote {runner_vote!r}"
    for r in rows:
        assert "nan" not in (r["displayed_options_json"] or "").lower()
        assert r["is_diagnostic_canary"] is True
    print(f"[canary] OK: {len(rows)} permutation rows, n_options={rows[0]['n_options']}, "
          f"vote={runner_vote}, correct_option={rows[0]['correct_option']}, "
          f"majority_is_correct={rows[0]['majority_is_correct']}")
    for r in rows:
        print(f"  perm={r['permutation_index']} transport={r['transport_status']} "
              f"displayed={r['displayed_parsed_choice']} semantic={r['semantic_parsed_choice']} "
              f"is_correct={r['is_correct']} cache_hit={r['cache_hit']} latency={r['latency_seconds']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["api", "local"], required=True)
    ap.add_argument("--cell", required=True, help="canonical cell_id, e.g. cbp__gpt-4-1-mini__arc_challenge__cyclic_generation_majority")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--benchmark", required=True, choices=["arc_challenge", "mmlu"])
    ap.add_argument("--split-name", default="robustness")
    ap.add_argument("--question-kind", choices=["normal", "damaged_arc"], required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.backend == "api":
        rows = asyncio.run(run_api_canary(
            args.cell, args.model_name, args.benchmark, args.split_name, args.question_kind,
        ))
    else:
        rows = run_local_canary(
            args.cell, args.model_name, args.benchmark, args.split_name, args.question_kind, args.device,
        )

    verify_and_report(rows, rows[0]["question_id"])

    CANARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CANARY_OUTPUT_DIR / f"v2_{args.backend}_{args.benchmark}_{args.question_kind}.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"[canary] wrote {out_path}")


if __name__ == "__main__":
    main()
