# Live smoke test evidence — harness v2 (2026-07-20)

Cell: `cbp__gpt-4-1-mini__arc_challenge__text_extraction`. Command:

```bash
set -a; source /home/cotenthusiast/Projects/two-stage-prompting/.env; set +a
python -m experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.run_repair \
    --cell-id cbp__gpt-4-1-mini__arc_challenge__text_extraction
```

Result:
```json
{
  "cell_id": "cbp__gpt-4-1-mini__arc_challenge__text_extraction",
  "staged_output": ".../staged_repairs/cbp__gpt-4-1-mini__arc_challenge__text_extraction/cbp__gpt-4-1-mini__arc_challenge__text_extraction_repaired_1000.csv",
  "provenance": ".../provenance/cbp__gpt-4-1-mini__arc_challenge__text_extraction.json",
  "n_keep_rows": 997,
  "n_repaired_rows": 3,
  "total_rows": 1000,
  "cache_namespace": "text_extraction__gpt-4.1-mini",
  "cache_entries_created": 3
}
```

## Checklist, each independently re-verified after the run (not just trusted from the harness's own internal assertion)

**Three genuine API calls occurred.** `assert_cache_grew_by` did not raise
(the harness would have refused to complete otherwise); independently
confirmed 3 new files under `.repair_cache/text_extraction__gpt-4.1-mini/`
(none existed before this run — the namespace was created fresh by this
invocation).

**All prompts are A/B/C-only; no `D. nan` remains.** Re-read directly from
the staged CSV for all 3 repaired rows — confirmed no `D.` line at all, no
`nan` substring anywhere in any of the 3 prompts.

**Latency populated for all 3 (real, distinct, nonzero call durations):**

| question_id | latency_seconds |
|---|---|
| `79e8c959bbeb74a0` | 2.902650 |
| `ad6b5d46ae54842c` | 0.802846 |
| `c30e75b011696a95` | 1.687109 |

**`timestamp_utc` is NOT populated — documented, expected, not a defect.**
Verified: every provider client in `choicebench/clients/*.py`
(openai/gemini/groq/together/anthropic/vllm) hardcodes
`timestamp_utc=None` unconditionally; nothing in `choicebench/clients/base.py`
ever sets it to a real value. This is a genuine, permanent gap in
ChoiceBench itself (present for real calls exactly as much as for cache
hits), not a freshness signal — see `repair_infra.py`'s module docstring.
`latency_seconds` (real, distinct per call, confirmed above) is the
positive evidence actually available, backed by the independent
`assert_cache_grew_by` file-count check as the authoritative freshness
proof.

**The staged 1000-row artifact differs from its source at exactly those
three IDs.** Compared every content column (question_text, choice_a-d,
prompt, raw_text, parsed_choice, correct_option, is_correct) between the
staged output and the historical source
(`two-stage-prompting/runs/20260603_154649/20260603_154649_text_extraction_gpt-4.1-mini_arc_challenge.csv`)
for all 1000 rows: **exactly 3 rows differ, exactly
`{79e8c959bbeb74a0, ad6b5d46ae54842c, c30e75b011696a95}`, and only in the
`prompt` column** (the corrected A/B/C-only prompt replacing the phantom-D
one). `raw_text`/`parsed_choice`/`is_correct` happen to be identical to the
historical (contaminated) values for these 3 rows — gpt-4.1-mini gave the
same correct free-text answer ("slower"/"solid"/"decrease") regardless of
whether it was shown a bogus 4th option, which is a property of this
model's behavior on these 3 questions, not evidence the repair did
nothing — the *prompt it was actually shown* is unambiguously corrected,
which is the thing being repaired.

Noted, not a defect: the 997 "keep" rows show a purely cosmetic
float-vs-int formatting difference in 3 metadata columns
(`sample_index`, `max_tokens`, `seed` — e.g. `0` vs `0.0`), an artifact of
the checkpoint-seed's JSON round-trip (pandas widens an all-int column to
float once any other row in the same column is NaN). No content column
differs for any of the 997 keep rows.

**Row/ID integrity:** 1000 total rows, 1000 unique `question_id` values, 0
duplicates, 0 missing.

**Historical source and immutable freeze untouched.** `two-stage-prompting`
`git status` shows no new tracked changes (same 2 pre-existing untracked
files as before this session); `two-stage-prompting/checkpoints/20260603_154649/`
remains empty (this harness never writes there — confirmed by directory
listing immediately after the run). The freeze
(`model-generalization/paper_data_freeze/`) is read-only to every module in
this harness (`load_frozen_arc_dataset` never writes).

## Cost

3 real gpt-4.1-mini API calls, ~$0.001 (gpt-4.1-mini pricing, ~200 tokens/call).
