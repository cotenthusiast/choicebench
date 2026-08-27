# Pre-registered prediction: `.strip()` whitespace causal-effect variant

Written and committed before this variant is run. Records a prediction, not a
result — do not edit after the run to make it agree with what happened;
append a "Result" section instead. Third in this series, after
`examples/pride_dtype_variant_prediction.md` (ruled out) and
`examples/pride_scoring_variant_prediction.md` (ruled out).

## What this tests

Phase 4 found 809/13,564 (5.96%) question stems in the persisted run carry
leading and/or trailing whitespace (509 leading-only, 300 trailing-only, 0
both) that the reference implementation's `question.strip()` removes but our
`build_direct_mcq_prompt`/`_build_options_block` never does. Only
*prevalence* was measured before — this variant tests *causal impact*.

## Variant construction

For each of the 809 affected rows, take the exact persisted `prompt` string
(from `runs/20260705_204054_dedup/..._direct_logprob_..._filtered.csv`) and
replace the one literal occurrence of the raw `question_text` with its
`.strip()`'d form — verified to occur exactly once per prompt before
replacing. This reproduces exactly what `create_user_prompt`'s
`question.strip()` would have produced, since the template
(`prompts/pride_repro/direct_mcq.txt`) inserts `{question}` literally with
no other processing. Options are left untouched (zero whitespace cases
there, per Phase 4). Scoring is the current pipeline only (full-vocab
log_softmax, dominant token form) — the scoring-mechanism variant is already
independently ruled out (job 9708113), so mixing it in here would confound
the isolation.

The other 12,755 unaffected rows have byte-identical prompts before and
after, so their predictions are reused directly from job 9708113's
`current_pipeline_fresh` results (`pred_current` column,
`scoring_variant_results.csv`) rather than re-run — this is a paired
comparison, not two independently noisy full runs.

## Reasoning

**Aggregate impact is arithmetically capped regardless of effect size**:
809/13,564 = 5.96% of the population. Even in the extreme case where every
single affected row's prediction flipped from wrong to right (or vice
versa), the maximum possible aggregate accuracy swing is ±5.96pp — smaller
than the 8.35pp gap on its own, and that extreme case is very unlikely.
Realistically I expect only a fraction of the 809 to flip at all.

**Subset-level effect (the real test per Task 1e)**: the whitespace
difference sits at the very start of the prompt (right after `"Question: "`,
or at the question/`"\nOptions:"` boundary for trailing cases), while the
scored position is the *last* token of a prompt with median length 115
tokens (full distribution: examples/pride_reproduction_results, Phase 3
analysis). An extra/missing space this early, this far from the answer
position, changes the token sequence but is a small perturbation relative
to everything the model attends over. I don't expect it to be a dominant
signal for the final-position logits in most cases, but attention-based
models can and do pick up on subtle stylistic irregularities (a double
space or missing space is a rare pattern relative to training data), so I
can't rule out a larger-than-naive effect either.

## Prediction

**Aggregate accuracy delta** (all 13,564 rows, corrected vs original): under
**+0.5pp in magnitude** — capped low by the 5.96% prevalence argument above
regardless of what the subset shows.

**Aggregate RStd delta**: under **0.5** in magnitude, same reasoning.

**Affected-subset (809 rows) accuracy delta**: my best guess is a **small
effect, roughly 0-3pp**, i.e. I expect this mechanism to have some teeth but
not be a dominant one — most of the 809 rows' predictions will not flip,
because the perturbation is early and minor relative to the whole prompt.

**Flip rate within the 809-row subset**: predict **fewer than 10%** (under
~80 rows) change their predicted letter between original and stripped
prompts.

**Overall conclusion I expect to draw**: like the last two variants, I
expect this to be real-but-small — a demonstrable, non-zero effect that
still leaves the +8.35pp/-3.05 RStd gap almost entirely unexplained. If the
affected-subset flip rate or accuracy delta comes back much larger than
predicted (say >20% flip rate, or >10pp subset accuracy swing), that is a
genuinely surprising result given the "minor, early perturbation" reasoning
above, and should be reported as such — it would mean this specific
tokenization region matters far more to the model's final answer than
attention intuition suggests, which would itself be a notable finding
independent of whether it fully explains the gap.

## Result

Run 2026-08-27, Kelvin2 SLURM job 9708414 (k2-gpu-a100mig, gpu114,
gpu:3g.40gb:1), 101.8s wall clock for 809×2 forward passes. Smoke-tested
first (job 9708410, 10 rows). Consistency check: all 809 `pred_original`
values matched job 9708113's `pred_current` for the same question_ids
exactly (0 mismatches) — confirms fidelity before trusting `pred_stripped`.
Script: `examples/pride_whitespace_variant_run.py`. Raw output:
`examples/pride_reproduction_results/ws_variant_results.csv`.

**Aggregate (all 13,564 rows, baseline vs whitespace-corrected)**:

| | accuracy | RStd |
|---|---|---|
| baseline (job 9708113) | 42.8930% | 14.3753 |
| whitespace-corrected | 42.8708% | 14.4244 |
| **Δ** | **-0.022pp** | **+0.049** |

Gap vs paper target: 8.29pp→8.27pp accuracy, -3.02→-2.98 RStd (using this
session's fresh baseline, which differs from the original persisted 42.95/
14.35 by run-to-run fp16 float noise of ~0.06pp, not a new discrepancy).
**Aggregate effect is negligible, as predicted** (<0.5pp/units bound).

**Affected subset (809 rows), original vs stripped prompt**:

| | accuracy | RStd (n≈200/letter, noisy) |
|---|---|---|
| original | 50.56% | 11.94 |
| stripped | 50.19% | 13.80 |
| **Δ** | **-0.37pp** | **+1.86** |

**Flip rate**: 72/809 = **8.90%** of the affected subset changes its
predicted letter between the original and stripped prompt — real and
non-trivial at the individual-question level, right at the edge of (but
within) the predicted <10% bound. Critically, the flips are **not
directional**: 21 originally-right→now-wrong, 18 originally-wrong→now-right,
33 both-wrong-but-different-letter. This is close to a coin flip, not a
systematic push toward or away from correctness.

## Verdict: prediction CONFIRMED, mechanism real but non-systematic

Every number landed inside the predicted bounds, unlike the scoring-variant
session where the prediction was wrong. The whitespace difference does have
real "teeth" at the individual-question level (8.9% flip rate is not
nothing), but the flips are symmetric/undirected rather than systematically
inflating our accuracy or suppressing our RStd — so even setting aside the
5.96% prevalence cap, this mechanism has no *systematic* component that
could explain a multi-point, one-directional gap. It behaves like ordinary
prompt-perturbation noise, not a selection-bias-shaped effect.

**This closes approximately 0% of the +8.35pp/-3.05 gap** (aggregate Δ of
-0.02pp / +0.05, an order of magnitude too small to matter) — third
consecutive ruled-out/negligible candidate.
