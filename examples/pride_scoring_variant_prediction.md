# Pre-registered prediction: reference-method scoring reimplementation (Phase 4b)

Written and committed before this variant is run. Records a prediction, not a
result — do not edit after the run to make it agree with what happened;
append a "Result" section instead. Companion to
`examples/pride_dtype_variant_prediction.md` (dtype isolation, separate and
lower-priority variant).

## What this tests

A distinct, source-verified difference between our scoring path and the
reference implementation (`chujiezheng/LLM-MCQ-Bias/code/eval_clm_utils.py`,
`prepare_eval_fn_base`), found while resolving the Phase 4 `.strip()`
question. This is **not** a retest of the killed tail-token hypothesis
("we read logit id 29909 instead of 319") — that claim was about *our* side
and stays dead; we confirmed we do read 319. This is a new claim about
*their* side:

- **Ours** (`hf_backend.py:250-254`): `log_softmax` over the full ~32k
  vocabulary, then extract just the single space-prefixed token per letter
  (A→319, B→350, C→315, D→360), then renormalize those 4 values
  (`logprob_map_to_label_distribution`).
- **Reference**: for each letter, take the *last token* of both `": {L}"`
  and `":{L}"` (space and no-space forms — verified locally these are
  different token IDs: A→319/29909, B→350/29933, C→315/29907, D→360/29928),
  restrict a softmax to just those 8 raw logits (not full vocab), then sum
  the two per-letter probabilities.

Both scores are computed from the *same* forward pass per question (one
model call, two aggregations from the same logits) — a paired comparison,
not two separate noisy runs.

## Baseline being compared against

`reports/20260705_204054_dedup_metrics.json`, `direct_logprob`: accuracy =
0.4295193 (42.95), rstd = 14.347638.
Paper's target (`examples/pride_reproduction_targets.json`, `default`):
acc = 34.6, rstd = 17.4.
Gap: **+8.35pp accuracy, -3.05 RStd** (ours minus theirs).

## Reasoning

The paper's entire thesis is that Default's poor accuracy and high RStd stem
from a per-letter **token-ID bias** baked into the model's prior over
option-ID tokens, independent of content. If the reference method's 8-way
restricted softmax pulls in a second, differently-biased token per letter
(the bare/no-space form, which is a much rarer completion in this exact
"Answer:"-then-letter context than the space-prefixed form), and that second
token's bias is *not* evenly distributed across A/B/C/D, then reproducing
their exact aggregation should reproduce more of their reported bias — i.e.
move RStd *up* (toward 17.4) and accuracy *down* (toward 34.6), relative to
our narrower 4-way extraction.

I do not have a way to check the no-space token's typical bias magnitude
without running this, so the direction below is a reasoned bet, not a
certainty.

## Prediction

**Accuracy**: expect a **decrease**, landing somewhere in the
**37-41%** range (down from 42.95%, i.e. closing roughly a third to two
thirds of the 8.35pp gap). I don't expect it to fully close the gap to
34.6%, since the `.strip()` difference (Phase 4, ~6% of questions) is a
separate, independently-contributing factor not addressed by this variant.

**RStd**: expect an **increase**, landing somewhere in the **15-17** range
(up from 14.35, i.e. closing a meaningful fraction of the -3.05 gap toward
17.4).

**Per-letter mass (Task 1e)**: I expect the no-space/bare token's share of
each letter's *combined* probability to usually be small in the median case
(most probability mass on the space-prefixed form, since that's the
overwhelmingly more natural continuation after "Answer:"), but with a
**non-trivial right tail** — a meaningful minority of questions (my guess:
somewhere in the 10-30% range) where the bare-token mass is large enough to
be competitive, since that's the only way this mechanism could move
aggregate accuracy/RStd by several points rather than being pure noise.

**Overall conclusion I expect to draw**: this scoring-mechanism difference
is a real, non-trivial contributor to the Default-cell gap, but likely not
the sole cause — I expect a partial, not complete, closure of the gap. If
the actual result is a null effect (accuracy/RStd barely move) or, contrary
to my prediction, the gap reverses direction, that is a more surprising
finding than what I currently expect and should be reported as such, not
rationalized after the fact.

## Result

_(Not yet run. Fill in after the variant completes: actual accuracy, rstd,
per-letter no-space mass share distribution, and how much of the 8.35pp/-3.05
gap it closes, with an honest comparison against the prediction above.)_
