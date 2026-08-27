"""
Phase 4c whitespace causal-effect variant: scores the 809 questions whose
stems carry leading/trailing whitespace our template doesn't strip, under
BOTH the original (unstripped) and reference-style (.strip()'d) prompt, using
the current pipeline's scoring rule only (full-vocab log_softmax, dominant
token form) — the scoring-mechanism variant is already independently ruled
out (job 9708113), so this isolates whitespace alone.

The other 12,755 unaffected rows are not re-run here — their prompts are
byte-identical before/after, so their job-9708113 pred_current values are
reused directly (see examples/pride_whitespace_variant_prediction.md).

Input CSV columns required: question_id, subject, correct_option,
original_prompt, stripped_prompt (see examples/pride_reproduction_results/
whitespace_affected_rows.csv, built locally from the original run's persisted
prompts).

Usage:
    python pride_whitespace_variant_run.py \
        --input whitespace_affected_rows.csv \
        --output whitespace_variant_results.csv \
        --model huggyllama/llama-13b \
        --revision bf57045473f207bb1de1ed035ace226f4d9f9bba
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

csv.field_size_limit(10**7)

LETTERS = ["A", "B", "C", "D"]


def logprob_map_to_label_distribution(logps: np.ndarray) -> np.ndarray:
    probs = np.exp(logps - logps.max())
    probs = probs / probs.sum()
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    return probs / probs.sum()


def score_prompt(model, tok, space_ids, prompt: str) -> tuple[str, np.ndarray]:
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]
    log_probs_full = F.log_softmax(last_logits, dim=-1)
    logps_4 = np.array([log_probs_full[t].item() for t in space_ids])
    dist = logprob_map_to_label_distribution(logps_4)
    pred = LETTERS[int(np.argmax(dist))]
    return pred, dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="huggyllama/llama-13b")
    ap.add_argument("--revision", default="bf57045473f207bb1de1ed035ace226f4d9f9bba")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap row count")
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.model}@{args.revision}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)

    print("Loading model (fp16, cuda)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.to("cuda")
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    space_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    print(f"space_ids={dict(zip(LETTERS, space_ids))}", flush=True)

    with open(args.input, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows from {args.input}", flush=True)

    out_fields = [
        "question_id",
        "subject",
        "correct_option",
        "pred_original",
        "pred_stripped",
        "dist_original_json",
        "dist_stripped_json",
    ]

    t_start = time.time()
    with open(args.output, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()

        with torch.no_grad():
            for i, row in enumerate(rows):
                pred_orig, dist_orig = score_prompt(model, tok, space_ids, row["original_prompt"])
                pred_strip, dist_strip = score_prompt(model, tok, space_ids, row["stripped_prompt"])

                writer.writerow(
                    {
                        "question_id": row["question_id"],
                        "subject": row["subject"],
                        "correct_option": row["correct_option"],
                        "pred_original": pred_orig,
                        "pred_stripped": pred_strip,
                        "dist_original_json": json.dumps(dist_orig.tolist()),
                        "dist_stripped_json": json.dumps(dist_strip.tolist()),
                    }
                )

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    eta = (len(rows) - i - 1) / rate
                    print(f"  {i + 1}/{len(rows)}  ({rate:.1f} q/s, eta {eta:.0f}s)", flush=True)

    print(f"Done. Total time {(time.time() - t_start):.1f}s. Wrote {args.output}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
