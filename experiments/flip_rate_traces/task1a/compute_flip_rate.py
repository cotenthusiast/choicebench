"""Compute the any-flip flip-rate metric across baseline / two_prompt /
twostage_semantic_match / cyclic trace sets, per model x benchmark.

Flip definition: a question "flips" if semantic_parsed_choice takes >= 2
distinct values across that question's permutation rows. Denominator is the
question's own permutation count (3 for the 3 known 3-option ARC questions,
4 otherwise).

Before printing any flip-rate table, validates 3 questions per cell by
printing their raw per-permutation semantic_parsed_choice values, so the
metric can be eyeballed against the underlying trace data.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

TRACES_DIR = Path(__file__).resolve().parent / "traces"

# (glob pattern, method_label, model, benchmark)
CELLS = [
    ("baseline_flip_rate_traces_v2_api_gpt41mini_arc_challenge_*.csv", "baseline", "gpt-4.1-mini", "arc_challenge"),
    ("baseline_flip_rate_traces_v2_api_gpt41mini_mmlu_*.csv", "baseline", "gpt-4.1-mini", "mmlu"),
    ("baseline_flip_rate_traces_v2_local_qwen7b_arc_challenge_*.csv", "baseline", "Qwen/Qwen2.5-7B-Instruct", "arc_challenge"),
    ("baseline_flip_rate_traces_v2_local_qwen7b_mmlu_*.csv", "baseline", "Qwen/Qwen2.5-7B-Instruct", "mmlu"),

    ("semantic_match_cbp__gpt-4-1-mini__arc_challenge__*.csv", "twostage_semantic_match", "gpt-4.1-mini", "arc_challenge"),
    ("semantic_match_cbp__gpt-4-1-mini__mmlu__*.csv", "twostage_semantic_match", "gpt-4.1-mini", "mmlu"),
    ("semantic_match_cbp__qwen-qwen2-5-7b-instruct__arc_challenge__*.csv", "twostage_semantic_match", "Qwen/Qwen2.5-7B-Instruct", "arc_challenge"),
    ("semantic_match_cbp__qwen-qwen2-5-7b-instruct__mmlu__*.csv", "twostage_semantic_match", "Qwen/Qwen2.5-7B-Instruct", "mmlu"),

    ("two_prompt_stage2_api_cbp__gpt-4-1-mini__arc_challenge__two_prompt_stage2.csv", "two_prompt", "gpt-4.1-mini", "arc_challenge"),
    ("two_prompt_stage2_api_cbp__gpt-4-1-mini__mmlu__two_prompt_stage2.csv", "two_prompt", "gpt-4.1-mini", "mmlu"),
    ("two_prompt_stage2_local_qwen7b_arc_challenge.csv", "two_prompt", "Qwen/Qwen2.5-7B-Instruct", "arc_challenge"),
    ("two_prompt_stage2_local_qwen7b_mmlu.csv", "two_prompt", "Qwen/Qwen2.5-7B-Instruct", "mmlu"),

    # cyclic: same source data as baseline (see gen_baseline_traces.py), kept
    # under its own label per instruction not to conflate the two.
    ("baseline_flip_rate_traces_v2_api_gpt41mini_arc_challenge_*.csv", "cyclic", "gpt-4.1-mini", "arc_challenge"),
    ("baseline_flip_rate_traces_v2_api_gpt41mini_mmlu_*.csv", "cyclic", "gpt-4.1-mini", "mmlu"),
    ("baseline_flip_rate_traces_v2_local_qwen7b_arc_challenge_*.csv", "cyclic", "Qwen/Qwen2.5-7B-Instruct", "arc_challenge"),
    ("baseline_flip_rate_traces_v2_local_qwen7b_mmlu_*.csv", "cyclic", "Qwen/Qwen2.5-7B-Instruct", "mmlu"),
]


def flip_rate_for_df(df: pd.DataFrame) -> tuple[int, int, float, pd.DataFrame]:
    per_q = df.groupby("question_id")["semantic_parsed_choice"].agg(
        n_perm="count", n_distinct=lambda s: s.dropna().nunique(),
    )
    per_q["flipped"] = per_q["n_distinct"] >= 2
    n_questions = len(per_q)
    n_flipped = int(per_q["flipped"].sum())
    rate = n_flipped / n_questions if n_questions else float("nan")
    return n_questions, n_flipped, rate, per_q


def validate_sample(df: pd.DataFrame, per_q: pd.DataFrame, label: str, n_sample: int = 3) -> None:
    flipped_ids = per_q.index[per_q["flipped"]].tolist()
    unflipped_ids = per_q.index[~per_q["flipped"]].tolist()
    sample_ids = (flipped_ids[:n_sample] + unflipped_ids[: max(0, n_sample - len(flipped_ids))])[:n_sample]
    print(f"  --- validation sample ({label}); {len(flipped_ids)} flipped available ---")
    for qid in sample_ids:
        rows = df[df["question_id"] == qid].sort_values("permutation_index")
        answers = rows["semantic_parsed_choice"].tolist()
        flipped = per_q.loc[qid, "flipped"]
        print(f"    q={qid} perms={answers} -> flipped={flipped} (n_distinct={per_q.loc[qid, 'n_distinct']})")


def main() -> None:
    results = []
    for pattern, method, model, benchmark in CELLS:
        matches = sorted(glob.glob(str(TRACES_DIR / pattern)))
        if not matches:
            print(f"MISSING: {method} / {model} / {benchmark} (pattern {pattern} matched nothing)")
            continue
        df = pd.read_csv(matches[0])
        n_q, n_flipped, rate, per_q = flip_rate_for_df(df)
        print(f"{method} / {model} / {benchmark}: n_questions={n_q} n_flipped={n_flipped} flip_rate={rate:.4f}")
        validate_sample(df, per_q, f"{method}/{model}/{benchmark}")
        results.append({
            "model": model, "benchmark": benchmark, "method": method,
            "n_questions": n_q, "n_flipped": n_flipped, "flip_rate": round(rate, 4),
        })

    out = pd.DataFrame(results)
    out = out.sort_values(["model", "benchmark", "method"])
    print()
    print("=== FINAL TABLE ===")
    print(out.to_string(index=False))
    out.to_csv(TRACES_DIR / "flip_rate_table.csv", index=False)


if __name__ == "__main__":
    main()
