"""Build per-permutation twostage_semantic_match traces.

No new model calls. Reuses the existing Stage-1 free_text_response (already
collected, order-independent since Stage 1 never shows options) and applies
the existing local sentence-transformer matcher
(twoprompt.parsing.text_matcher.match_text_to_options) once per permutation
of displayed option order.

Expected result: near-zero flip rate. match_text_to_options matches free
text against option TEXT content, never against displayed position -- the
winning content is the same regardless of which letter it's displayed under,
so the unpermuted (canonical) answer should be permutation-invariant except
for tie-break artifacts driven by dict iteration order. This script exists
to confirm that empirically, not to assume it.
"""
from __future__ import annotations

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

from twoprompt.parsing.text_matcher import match_text_to_options

OUT_DIR = Path(__file__).resolve().parent / "traces"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# cell_id (frozen question source) -> (stage1 CSV with free_text_response, model_name, provider)
STAGE1_SOURCES = {
    "cbp__gpt-4-1-mini__arc_challenge__cyclic_generation_majority": (
        "/home/cotenthusiast/Projects/two-stage-prompting/runs/20260601_183346/"
        "20260601_183346_two_prompt_gpt-4.1-mini_arc_challenge.csv",
        "gpt-4.1-mini", "openai", "arc_challenge",
    ),
    "cbp__gpt-4-1-mini__mmlu__cyclic_generation_majority": (
        "/home/cotenthusiast/Projects/two-stage-prompting/runs/20260601_183346/"
        "20260601_183346_two_prompt_gpt-4.1-mini_mmlu.csv",
        "gpt-4.1-mini", "openai", "mmlu",
    ),
    "cbp__qwen-qwen2-5-7b-instruct__arc_challenge__cyclic_generation_majority": (
        "/home/cotenthusiast/Projects/model-generalization/paper_data_freeze/canonical/"
        "cbp__qwen-qwen2-5-7b-instruct__arc_challenge__two_stage_v1/"
        "20260529_145812_two_prompt_Qwen_Qwen2.5-7B-Instruct_arc_challenge.csv",
        "Qwen/Qwen2.5-7B-Instruct", "huggingface", "arc_challenge",
    ),
    "cbp__qwen-qwen2-5-7b-instruct__mmlu__cyclic_generation_majority": (
        "/home/cotenthusiast/Projects/model-generalization/paper_data_freeze/canonical/"
        "cbp__qwen-qwen2-5-7b-instruct__mmlu__two_stage_v1/"
        "20260529_145812_two_prompt_Qwen_Qwen2.5-7B-Instruct_mmlu.csv",
        "Qwen/Qwen2.5-7B-Instruct", "huggingface", "mmlu",
    ),
}


def _question_options(q) -> dict[str, str]:
    opts = {"A": q.choice_a, "B": q.choice_b, "C": q.choice_c, "D": q.choice_d}
    return {k: v for k, v in opts.items() if pd.notna(v) and str(v).strip() != ""}


def run_cell(cell_id: str, stage1_csv: str, model_name: str, provider: str, benchmark: str) -> None:
    questions = load_frozen_questions(cell_id)
    stage1 = pd.read_csv(stage1_csv).set_index("question_id")["free_text_response"]

    run_id = "task1a_twostage_semantic_match_v1"
    out_cell_id = cell_id.replace(
        "cyclic_generation_majority", "twostage_semantic_match"
    ) + "_traced_v1"

    all_rows = []
    for q in questions:
        canonical_options = _question_options(q)
        free_text = stage1.get(q.question_id)
        permutations = generate_permutations(canonical_options)
        n = len(permutations)

        semantic_choices = []
        perm_rows_meta = []
        for idx, perm in enumerate(permutations):
            t0 = time.monotonic()
            if pd.isna(free_text):
                displayed_letter = None
                match_score = None
                transport_status = "failure"
                error_type, error_msg, error_stage = "MissingStage1", "no free_text_response for question", "stage1_lookup"
            else:
                result = match_text_to_options(free_text, perm)
                displayed_letter = result.label
                match_score = result.score
                transport_status = "success"
                error_type = error_msg = error_stage = None
            latency = time.monotonic() - t0

            sem = unpermute_choice(displayed_letter, perm, canonical_options) if displayed_letter else None
            semantic_choices.append(sem)
            perm_rows_meta.append((idx, perm, displayed_letter, sem, match_score, latency,
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

        for idx, perm, displayed_letter, sem, match_score, latency, transport_status, error_type, error_msg, error_stage in perm_rows_meta:
            is_correct = (sem == q.correct_option) if sem is not None else False
            row = build_trace_row(
                schema_version="flip_rate_traces.trace.v1",
                run_id=run_id,
                cell_id=out_cell_id,
                question_id=q.question_id,
                benchmark=benchmark,
                split_name="robustness",
                subject=q.subject,
                model_name=model_name,
                provider=provider,
                n_options=n,
                permutation_index=idx,
                n_permutations=n,
                canonical_options_json=json.dumps(canonical_options, sort_keys=True),
                displayed_options_json=json.dumps(perm, sort_keys=True),
                displayed_to_semantic_json=json.dumps({
                    letter: unpermute_choice(letter, perm, canonical_options) for letter in perm
                }, sort_keys=True),
                correct_option=q.correct_option,
                prompt=f"[local_semantic_match] free_text={free_text!r} vs options={perm!r}",
                prompt_sha256=None,
                temperature=None,
                max_tokens=None,
                seed=None,
                prompt_version="semantic_match_v1",
                raw_text=free_text if pd.notna(free_text) else None,
                finish_reason=None,
                displayed_parsed_choice=displayed_letter,
                semantic_parsed_choice=sem,
                parse_status=("parse_ok" if displayed_letter else "parse_missing"),
                parse_reason=(f"cosine_similarity={match_score:.4f}" if match_score is not None else "no_stage1_free_text"),
                normalized_text=None,
                is_correct=is_correct,
                score_status=("score_correct" if is_correct else "score_incorrect"),
                transport_status=transport_status,
                error_type=error_type,
                error_message=error_msg,
                error_stage=error_stage,
                error_retryable=False if transport_status == "failure" else None,
                latency_seconds=latency,
                timestamp_utc=pd.Timestamp.utcnow().isoformat(),
                cache_hit=None,
                majority_semantic_choice=voted_letter,
                majority_is_tie=is_tie,
                majority_tie_break_used=(is_tie and voted_letter is not None),
                majority_is_correct=majority_is_correct,
                is_diagnostic_canary=False,
            )
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    dest = OUT_DIR / f"semantic_match_{out_cell_id}.csv"
    df.to_csv(dest, index=False)
    n_missing = sum(1 for r in all_rows if r["transport_status"] == "failure")
    print(f"{cell_id}: {len(all_rows)} rows ({len(questions)} questions), {n_missing} missing-stage1 rows -> {dest}")


def main() -> None:
    for cell_id, (stage1_csv, model_name, provider, benchmark) in STAGE1_SOURCES.items():
        run_cell(cell_id, stage1_csv, model_name, provider, benchmark)


if __name__ == "__main__":
    main()
