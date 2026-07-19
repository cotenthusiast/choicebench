# INVALID — diagnostic evidence only, do not import, do not use

## What this is

Output of the first live attempt at a Group 1 historical repair
(`cbp__gpt-4-1-mini__arc_challenge__text_extraction`, 2026-07-20), preserved
here purely as diagnostic evidence of why the original checkpoint-seed +
resume-via-unmodified-`run_experiment.py` mechanism does not work. **This
data must never be imported, merged, or treated as a valid repair.**

## Why it is invalid

1. **Not actually a fresh call.** `INVALID_full_output.csv` rows for
   `79e8c959bbeb74a0`, `ad6b5d46ae54842c`, `c30e75b011696a95` show
   `latency_seconds == 0.0` and `timestamp_utc` empty — a cache hit, not a
   real API call. Confirm with:
   ```python
   df[df.question_id.isin(['79e8c959bbeb74a0','ad6b5d46ae54842c','c30e75b011696a95'])][['question_id','latency_seconds','timestamp_utc']]
   ```
2. **Even if it had been a fresh call, it would still be contaminated.**
   The prompt for these 3 rows still reads `D. nan` — TSP's unmodified
   `TextExtractionRunner`/`build_text_extraction_prompt` always interpolates
   all 4 option slots; re-running it (fresh or cached) reproduces the exact
   same defect it was meant to fix. This is why the repair harness
   (`../../harness/`) exists instead.

## What actually happened to the real historical file

`two-stage-prompting/runs/20260603_154649/20260603_154649_text_extraction_gpt-4.1-mini_arc_challenge.csv`
was overwritten in place by this run (the checkpoint-resume mechanism's
intended behavior), but because the 3 regenerated rows were a byte-for-byte
cache hit of the pre-existing (already contaminated) values, **the file's
content did not actually change** — every field for all 1000 rows,
including the 3 target IDs, is identical to what was there before this run.
No historical data was newly corrupted; the "repair" was a no-op, not a
regression. The checkpoint file this run wrote to
`two-stage-prompting/checkpoints/20260603_154649/` was deleted by
`run_experiment.py` on successful completion (its own normal behavior) and
so cannot be preserved alongside this CSV snapshot.

## Cost

$0 — the 3 "calls" were cache hits, not real API requests.
