# Pre-registered prediction: fp16 -> bf16 dtype variant

Written and committed before this variant is run, per the bisection protocol
for the PriDe/Cyclic-Perm reproduction deviation (see
`examples/pride_reproduction.md` and the diagnostic session tracked
externally). This file records a prediction, not a result — do not edit it
after the run to make it agree with what actually happened; append a
"Result" section instead.

## What changes

Exactly one line, `src/choicebench/backends/hf_backend.py:107`:
`torch.float16` -> `torch.bfloat16` for the `device != "cpu"` branch. Nothing
else changes: same checkpoint
(`huggyllama/llama-13b @ bf57045473f207bb1de1ed035ace226f4d9f9bba`), same
permutation-filtered + deduped population (13,564 rows), same prompt
template (`pride_repro`), same seed (42), same `add_bos_token: false`.

## Baseline being compared against

This run's persisted Default (`direct_logprob`) cell:
`reports/20260705_204054_dedup_metrics.json`:
- accuracy = 0.4295193 (42.95)
- rstd = 14.347638

Paper's target (`examples/pride_reproduction_targets.json`, `default`):
- acc = 34.6, rstd = 17.4

Gap this variant is being tested against: **+8.35pp accuracy, -3.05 RStd**.

## Prior evidence ruling out other single-factor explanations

- Population/prompt-template/parse-rate: ruled out (prior session).
- Original tail-token hypothesis ("we read logit id 29909 instead of 319"):
  **verified false** — `tokenizer.encode("A", add_special_tokens=False)` at
  this checkpoint returns `[319]` (▁A), the same dominant form the
  reference implementation also uses. Not being retested here.
- Truncation (last-1536-tokens): **ruled out** — tokenized every persisted
  prompt in this run with the pinned tokenizer; max length is 1219 tokens,
  p99.9 is 757. Nothing in this population is long enough for a
  last-1536-token truncation policy to ever differ from no truncation.
- `.strip()` whitespace handling: **found material** — 809/13,564 (5.96%)
  question stems carry leading and/or trailing whitespace that our template
  does not strip but the reference `create_user_prompt`
  (`question.strip()`) does. 509 leading-only, 300 trailing-only,
  concentrated in a handful of MMLU subjects (nutrition, moral_disputes,
  marketing, world_religions, formal_logic — known raw-text artifacts in
  `cais/mmlu`). Not touched by this dtype variant.

Neither of the two live, unaddressed differences (whitespace, and a
separately-flagged scoring-rule question raised during Phase 4 — see the
Phase 4/5 status report) is changed by this variant. This variant isolates
**only** numerical precision (fp16 vs bf16 forward-pass rounding).

## Prediction

**Accuracy**: expect a small change, on the order of **-0.5 to +1.0pp**
relative to the fp16 run (42.95 -> roughly 42.0-44.0). Rationale: 779/13,564
rows (5.7%) in this run have a top1-vs-top2 probability margin under 0.01 —
close enough that fp16-vs-bf16 rounding in the final-position logits could
flip the argmax on a subset of these. Bf16 has fewer mantissa bits than fp16
(7 vs 10) but a wider exponent range; for logits already well-separated it
should not matter, so only the near-tie rows are at risk. I do not expect
this to systematically push accuracy in one direction — flips among
near-ties should be roughly symmetric — so the magnitude should stay under
~1.5pp either way, not resolve a majority of the 8.35pp gap.

**RStd**: expect **no material change** (stays roughly 13-16, i.e. still far
from the paper's 17.4). RStd measures per-letter recall spread; that
depends on how the four `A/B/C/D` position-recalls vary relative to each
other, which is driven by the model's positional/token-ID bias structure,
not by forward-pass numerical precision. A dtype change has no mechanism to
alter which letter the model is biased toward.

**Overall conclusion I expect to draw**: dtype is very unlikely to be the
(or even a major) driver of the Default-cell deviation. If the actual result
contradicts this — e.g. accuracy moves by several points, or RStd closes
substantially — that is a stronger and more surprising finding than I
currently expect, and should be reported as such rather than rationalized
after the fact.

## Result

_(Not yet run. Fill in after the variant completes — actual accuracy, rstd,
and per-row diff of which questions flipped, with an honest comparison
against the prediction above.)_
