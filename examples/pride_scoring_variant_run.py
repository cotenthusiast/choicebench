"""
Phase 4b scoring-mechanism variant: paired comparison of our current
direct_logprob scoring against a reimplementation of the reference
(chujiezheng/LLM-MCQ-Bias) Default scoring method, from the same forward
pass per question.

Not part of the choicebench package — a standalone diagnostic script for the
PriDe reproduction-deviation investigation. Reads the already-persisted
prompts from the original run's direct_logprob CSV (byte-identical prompts,
no re-derivation of the filter/dedup/prompt-building pipeline), so the only
variable between "current" and "reference" scores is the aggregation rule
applied to each question's next-token logits.

Usage:
    python pride_scoring_variant_run.py \
        --input 20260705_204054_dedup_direct_logprob_huggyllama_llama-13b_mmlu_filtered.csv \
        --output scoring_variant_results.csv \
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
    """Mirrors choicebench.methods.library.pride_math.logprob_map_to_label_distribution
    for the fixed 4-letter case: softmax over log-probs == renormalizing the
    underlying probabilities among just these labels."""
    probs = np.exp(logps - logps.max())
    probs = probs / probs.sum()
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    return probs / probs.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="huggyllama/llama-13b")
    ap.add_argument("--revision", default="bf57045473f207bb1de1ed035ace226f4d9f9bba")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap row count")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                     help="Phase 2 dtype-isolation variant (examples/pride_dtype_variant_prediction.md). "
                          "Default float16 matches hf_backend.py's hardcoded dtype exactly.")
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.model}@{args.revision}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"Loading model ({args.dtype}, cuda)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to("cuda")
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Token IDs: space-prefixed dominant form (what hf_backend.py currently
    # reads) vs bare no-space form (the reference implementation's second
    # summed token, from tokenizing ":{L}" with no space). Verified locally
    # before this run — see examples/pride_scoring_variant_prediction.md.
    space_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    nospace_ids = [tok(f":{L}").input_ids[-1] for L in LETTERS]
    print(f"space_ids={dict(zip(LETTERS, space_ids))}", flush=True)
    print(f"nospace_ids={dict(zip(LETTERS, nospace_ids))}", flush=True)
    assert len(set(space_ids)) == 4 and len(set(nospace_ids)) == 4
    assert set(space_ids).isdisjoint(nospace_ids)

    option_indices = torch.tensor(space_ids + nospace_ids, device="cuda")

    with open(args.input, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows from {args.input}", flush=True)

    out_fields = [
        "question_id",
        "subject",
        "correct_option",
        "pred_current",
        "pred_reference",
        "dist_current_json",
        "dist_reference_json",
        "space_probs_json",
        "nospace_probs_json",
    ]

    t_start = time.time()
    with open(args.output, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()

        with torch.no_grad():
            for i, row in enumerate(rows):
                prompt = row["prompt"]
                inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]

                # --- current pipeline: full-vocab log_softmax, extract 4, renormalize ---
                log_probs_full = F.log_softmax(last_logits, dim=-1)
                logps_4 = np.array([log_probs_full[t].item() for t in space_ids])
                dist_current = logprob_map_to_label_distribution(logps_4)
                pred_current = LETTERS[int(np.argmax(dist_current))]

                # --- reference method: restricted 8-way softmax, sum paired forms ---
                probs_8 = F.softmax(last_logits[option_indices], dim=-1).detach().cpu().to(torch.float32).numpy()
                space_probs = probs_8[:4]
                nospace_probs = probs_8[4:]
                dist_reference = space_probs + nospace_probs
                pred_reference = LETTERS[int(np.argmax(dist_reference))]

                writer.writerow(
                    {
                        "question_id": row["question_id"],
                        "subject": row["subject"],
                        "correct_option": row["correct_option"],
                        "pred_current": pred_current,
                        "pred_reference": pred_reference,
                        "dist_current_json": json.dumps(dist_current.tolist()),
                        "dist_reference_json": json.dumps(dist_reference.tolist()),
                        "space_probs_json": json.dumps(space_probs.tolist()),
                        "nospace_probs_json": json.dumps(nospace_probs.tolist()),
                    }
                )

                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    eta = (len(rows) - i - 1) / rate
                    print(f"  {i + 1}/{len(rows)}  ({rate:.1f} q/s, eta {eta / 60:.1f} min)", flush=True)

    print(f"Done. Total time {(time.time() - t_start) / 60:.1f} min. Wrote {args.output}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
