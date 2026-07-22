"""Real Stage-2 LLM-matching per-permutation traces for two_prompt (gpt-4.1-mini/API).

Stage 1 (free-text, option-blind) is reused verbatim from the existing
historical run -- it cannot depend on option order, so no new Stage-1 calls
are made. Stage 2 IS order-sensitive (the model sees the permuted options and
picks a letter), so this makes NEW real API calls: one per (question,
permutation). This is the approved-cost part of Task 1a.

Run with --canary N to test on the first N questions only before the full run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/cotenthusiast/Projects/choicebench-flip-rate-traces")
sys.path.insert(0, "/home/cotenthusiast/Projects/two-stage-prompting/src")

from experiments.flip_rate_traces.data_source import load_frozen_questions
from experiments.flip_rate_traces.historical_protocol import (
    generate_permutations,
    historical_majority_vote,
    unpermute_choice,
)
from experiments.flip_rate_traces.trace_schema import build_trace_row

from twoprompt.clients.openai_client import OpenAIClient
from twoprompt.clients.types import ModelRequest, RequestMetadata
from twoprompt.config.models import OPENAI_API_KEY, SEED, TEMPERATURE, MAX_TOKENS
from twoprompt.parsing.parser import parse_model_answer
from twoprompt.pipeline.prompt_builder import build_option_matching_prompt, load_prompt_templates

OUT_DIR = Path(__file__).resolve().parent / "traces"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR = Path("/home/cotenthusiast/Projects/two-stage-prompting/prompts")

CELLS = {
    "cbp__gpt-4-1-mini__arc_challenge__cyclic_generation_majority": (
        "/home/cotenthusiast/Projects/two-stage-prompting/runs/20260601_183346/"
        "20260601_183346_two_prompt_gpt-4.1-mini_arc_challenge.csv",
        "arc_challenge",
    ),
    "cbp__gpt-4-1-mini__mmlu__cyclic_generation_majority": (
        "/home/cotenthusiast/Projects/two-stage-prompting/runs/20260601_183346/"
        "20260601_183346_two_prompt_gpt-4.1-mini_mmlu.csv",
        "mmlu",
    ),
}
MODEL_NAME = "gpt-4.1-mini"
PROVIDER = "openai"


def _question_options(q) -> dict[str, str]:
    opts = {"A": q.choice_a, "B": q.choice_b, "C": q.choice_c, "D": q.choice_d}
    return {k: v for k, v in opts.items() if pd.notna(v) and str(v).strip() != ""}


async def run_cell(cell_id: str, stage1_csv: str, benchmark: str, run_id: str, canary_n: int | None) -> list[dict]:
    templates = load_prompt_templates("v1", PROMPTS_DIR)
    questions = load_frozen_questions(cell_id)
    if canary_n:
        questions = questions[:canary_n]
    stage1 = pd.read_csv(stage1_csv).set_index("question_id")["free_text_response"]

    client = OpenAIClient(model_name=MODEL_NAME, api_key=OPENAI_API_KEY, concurrency_limit=20)
    out_cell_id = cell_id.replace("cyclic_generation_majority", "two_prompt_stage2") + "_traced_v1"

    all_rows = []
    for qi, q in enumerate(questions):
        canonical_options = _question_options(q)
        free_text = stage1.get(q.question_id)
        permutations = generate_permutations(canonical_options)
        n = len(permutations)

        if pd.isna(free_text):
            # Stage 1 failed historically for this question -- no Stage 2 possible.
            continue

        prompts = [
            build_option_matching_prompt(
                template=templates["option_matching"],
                question=q.question_text,
                free_text=free_text,
                option_a=perm.get("A", ""), option_b=perm.get("B", ""),
                option_c=perm.get("C", ""), option_d=perm.get("D", ""),
            )
            for perm in permutations
        ]
        requests = [
            ModelRequest(
                provider=PROVIDER, model_name=MODEL_NAME, payload=prompt,
                metadata=RequestMetadata(
                    question_id=q.question_id, split_name="robustness",
                    method_name="two_prompt_stage2_task1a", subject=q.subject,
                    run_id=run_id, prompt_version="v1", perturbation_name=f"perm{idx}",
                    sample_index=0,
                ),
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=SEED,
            )
            for idx, prompt in enumerate(prompts)
        ]

        t0 = time.monotonic()
        responses = await client.generate_batch(requests)
        elapsed = time.monotonic() - t0
        per_call_latency = elapsed / len(requests)

        semantic_choices = []
        parsed_meta = []
        for idx, (perm, resp) in enumerate(zip(permutations, responses)):
            if resp.is_success():
                parsed = parse_model_answer(resp.raw_text, perm)
                sem = unpermute_choice(parsed.final_choice, perm, canonical_options) if parsed.final_choice else None
            else:
                parsed = None
                sem = None
            semantic_choices.append(sem)
            parsed_meta.append((perm, resp, parsed, sem))

        voted_letter = historical_majority_vote(semantic_choices)
        cleaned = [x for x in semantic_choices if x is not None]
        is_tie = False
        if cleaned:
            import collections
            counts = collections.Counter(cleaned)
            top = counts.most_common(2)
            is_tie = len(top) > 1 and top[0][1] == top[1][1]
        majority_is_correct = (voted_letter == q.correct_option) if voted_letter else False

        for idx, (perm, resp, parsed, sem) in enumerate(parsed_meta):
            is_correct = (sem == q.correct_option) if sem is not None else False
            err = resp.error if not resp.is_success() else None
            row = build_trace_row(
                schema_version="flip_rate_traces.trace.v1",
                run_id=run_id, cell_id=out_cell_id, question_id=q.question_id,
                benchmark=benchmark, split_name="robustness", subject=q.subject,
                model_name=MODEL_NAME, provider=PROVIDER,
                n_options=n, permutation_index=idx, n_permutations=n,
                canonical_options_json=json.dumps(canonical_options, sort_keys=True),
                displayed_options_json=json.dumps(perm, sort_keys=True),
                displayed_to_semantic_json=json.dumps(
                    {letter: unpermute_choice(letter, perm, canonical_options) for letter in perm},
                    sort_keys=True,
                ),
                correct_option=q.correct_option,
                prompt=prompts[idx], prompt_sha256=None,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=SEED, prompt_version="v1",
                raw_text=resp.raw_text if resp.is_success() else None,
                finish_reason=getattr(resp, "finish_reason", None),
                displayed_parsed_choice=(parsed.final_choice if parsed else None),
                semantic_parsed_choice=sem,
                parse_status=(parsed.status if parsed else None),
                parse_reason=(parsed.reason if parsed else None),
                normalized_text=(parsed.normalized_text if parsed else None),
                is_correct=is_correct,
                score_status=("score_correct" if is_correct else "score_incorrect"),
                transport_status=resp.status,
                error_type=(err.error_type if err else None),
                error_message=(err.message if err else None),
                error_stage=(err.stage if err else None),
                error_retryable=(err.retryable if err else None),
                latency_seconds=per_call_latency,
                timestamp_utc=pd.Timestamp.utcnow().isoformat(),
                cache_hit=None,
                majority_semantic_choice=voted_letter,
                majority_is_tie=is_tie,
                majority_tie_break_used=(is_tie and voted_letter is not None),
                majority_is_correct=majority_is_correct,
                is_diagnostic_canary=(canary_n is not None),
            )
            all_rows.append(row)

        if (qi + 1) % 50 == 0:
            print(f"  {cell_id}: {qi + 1}/{len(questions)} questions done")

    return all_rows


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary", type=int, default=None)
    args = ap.parse_args()

    run_id = "task1a_two_prompt_stage2_canary" if args.canary else "task1a_two_prompt_stage2_v1"

    for cell_id, (stage1_csv, benchmark) in CELLS.items():
        rows = await run_cell(cell_id, stage1_csv, benchmark, run_id, args.canary)
        df = pd.DataFrame(rows)
        suffix = f"_canary{args.canary}" if args.canary else ""
        out_cell_id = cell_id.replace("cyclic_generation_majority", "two_prompt_stage2")
        dest = OUT_DIR / f"two_prompt_stage2_api_{out_cell_id}{suffix}.csv"
        df.to_csv(dest, index=False)
        n_fail = (df["transport_status"] != "success").sum() if len(df) else 0
        print(f"{cell_id}: {len(rows)} rows, {n_fail} transport failures -> {dest}")


if __name__ == "__main__":
    asyncio.run(main())
