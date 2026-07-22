"""Real Stage-2 LLM-matching per-permutation traces for two_prompt
(Qwen/Qwen2.5-7B-Instruct, local, Kelvin2).

Mirrors gen_two_prompt_stage2_api.py but uses model-generalization's own
local backend/prompt-builder (modelgen), matching the historically-faithful
local Stage-2 protocol (dynamic {options} block template, HFCausalLMBackend
generate(), not twoprompt's API template/client). Stage 1 free-text is reused
verbatim from the historical local two_prompt run (20260529_145812) --
option-blind, so it cannot depend on permutation order.

Run on Kelvin2 via slurm/flip_rate_traces/local_qwen7b_two_prompt_stage2*.sbatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

CHOICEBENCH_REPO = os.environ.get(
    "CHOICEBENCH_FLIP_RATE_REPO",
    "/home/cotenthusiast/Projects/choicebench-flip-rate-traces",
)
MODELGEN_REPO = os.environ.get(
    "MODELGEN_REPO", "/home/cotenthusiast/Projects/model-generalization",
)
sys.path.insert(0, CHOICEBENCH_REPO)
sys.path.insert(0, str(Path(MODELGEN_REPO) / "src"))

from experiments.flip_rate_traces.data_source import load_frozen_questions
from experiments.flip_rate_traces.historical_protocol import (
    generate_permutations,
    historical_majority_vote,
    unpermute_choice,
)
from experiments.flip_rate_traces.trace_schema import build_trace_row

from modelgen.backends.qwen import QwenBackend
from modelgen.backends.types import LocalGenerationConfig
from modelgen.parsing.parser import parse_model_answer
from modelgen.pipeline.prompt_builder import build_option_matching_prompt, load_prompt_templates

OUT_DIR = Path(__file__).resolve().parent / "traces"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR = Path(MODELGEN_REPO) / "prompts"

CELL_ID = "cbp__qwen-qwen2-5-7b-instruct__{bench}__cyclic_generation_majority"
STAGE1_CSV = str(
    Path(MODELGEN_REPO)
    / "paper_data_freeze/canonical/cbp__qwen-qwen2-5-7b-instruct__{bench}__two_stage_v1"
    / "20260529_145812_two_prompt_Qwen_Qwen2.5-7B-Instruct_{bench}.csv"
)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PROVIDER = "huggingface"
GEN_CONFIG = LocalGenerationConfig(max_new_tokens=500, temperature=0.0, do_sample=False, seed=42)


def _question_options(q) -> dict[str, str]:
    opts = {"A": q.choice_a, "B": q.choice_b, "C": q.choice_c, "D": q.choice_d}
    return {k: v for k, v in opts.items() if pd.notna(v) and str(v).strip() != ""}


def run_cell(benchmark: str, backend, run_id: str, canary_n: int | None) -> list[dict]:
    templates = load_prompt_templates("v1", PROMPTS_DIR)
    cell_id = CELL_ID.format(bench=benchmark)
    questions = load_frozen_questions(cell_id)
    if canary_n:
        questions = questions[:canary_n]
    stage1 = pd.read_csv(STAGE1_CSV.format(bench=benchmark)).set_index("question_id")["free_text_response"]

    out_cell_id = cell_id.replace("cyclic_generation_majority", "two_prompt_stage2") + "_traced_v1"
    all_rows = []

    for qi, q in enumerate(questions):
        canonical_options = _question_options(q)
        free_text = stage1.get(q.question_id)
        permutations = generate_permutations(canonical_options)
        n = len(permutations)

        if pd.isna(free_text):
            continue

        semantic_choices = []
        perm_meta = []
        for idx, perm in enumerate(permutations):
            prompt = build_option_matching_prompt(
                template=templates["option_matching"],
                question=q.question_text,
                free_text=free_text,
                options=perm,
            )
            t0 = time.monotonic()
            try:
                result = backend.generate(prompt, config=GEN_CONFIG)
                latency = time.monotonic() - t0
                raw_text = result.raw_text
                finish_reason = result.finish_reason
                transport_status = "success"
                error_type = error_msg = error_stage = None
            except Exception as exc:  # noqa: BLE001
                latency = time.monotonic() - t0
                raw_text = None
                finish_reason = None
                transport_status = "failure"
                error_type = type(exc).__name__
                error_msg = str(exc)
                error_stage = "backend_generate"

            if raw_text is not None:
                parsed = parse_model_answer(raw_text, perm)
                sem = unpermute_choice(parsed.final_choice, perm, canonical_options) if parsed.final_choice else None
            else:
                parsed = None
                sem = None
            semantic_choices.append(sem)
            perm_meta.append((idx, perm, prompt, raw_text, finish_reason, parsed, sem, latency,
                               transport_status, error_type, error_msg, error_stage))

        voted_letter = historical_majority_vote(semantic_choices)
        cleaned = [x for x in semantic_choices if x is not None]
        is_tie = False
        if cleaned:
            import collections
            counts = collections.Counter(cleaned)
            top = counts.most_common(2)
            is_tie = len(top) > 1 and top[0][1] == top[1][1]
        majority_is_correct = (voted_letter == q.correct_option) if voted_letter else False

        for idx, perm, prompt, raw_text, finish_reason, parsed, sem, latency, transport_status, error_type, error_msg, error_stage in perm_meta:
            is_correct = (sem == q.correct_option) if sem is not None else False
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
                prompt=prompt, prompt_sha256=None,
                temperature=0.0, max_tokens=500, seed=42, prompt_version="v1",
                raw_text=raw_text, finish_reason=finish_reason,
                displayed_parsed_choice=(parsed.final_choice if parsed else None),
                semantic_parsed_choice=sem,
                parse_status=(parsed.status if parsed else None),
                parse_reason=(parsed.reason if parsed else None),
                normalized_text=(parsed.normalized_text if parsed else None),
                is_correct=is_correct,
                score_status=("score_correct" if is_correct else "score_incorrect"),
                transport_status=transport_status,
                error_type=error_type, error_message=error_msg, error_stage=error_stage,
                error_retryable=(False if transport_status == "failure" else None),
                latency_seconds=latency,
                timestamp_utc=pd.Timestamp.now("UTC").isoformat(),
                cache_hit=None,
                majority_semantic_choice=voted_letter,
                majority_is_tie=is_tie,
                majority_tie_break_used=(is_tie and voted_letter is not None),
                majority_is_correct=majority_is_correct,
                is_diagnostic_canary=(canary_n is not None),
            )
            all_rows.append(row)

        if (qi + 1) % 25 == 0:
            print(f"  {benchmark}: {qi + 1}/{len(questions)} questions done", flush=True)

    return all_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["arc_challenge", "mmlu"])
    ap.add_argument("--canary", type=int, default=None)
    args = ap.parse_args()

    backend = QwenBackend(model_path=MODEL_NAME)
    backend.load()

    run_id = "task1a_two_prompt_stage2_local_canary" if args.canary else "task1a_two_prompt_stage2_local_v1"
    rows = run_cell(args.benchmark, backend, run_id, args.canary)

    df = pd.DataFrame(rows)
    suffix = f"_canary{args.canary}" if args.canary else ""
    dest = OUT_DIR / f"two_prompt_stage2_local_qwen7b_{args.benchmark}{suffix}.csv"
    df.to_csv(dest, index=False)
    n_fail = (df["transport_status"] != "success").sum() if len(df) else 0
    print(f"{args.benchmark}: {len(rows)} rows, {n_fail} transport failures -> {dest}")


if __name__ == "__main__":
    main()
