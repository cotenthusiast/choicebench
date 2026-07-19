# The missing 2×2 cell: options visible + LLM matcher

Fills the last cell of a 2×2 (Stage-1 visibility × Stage-2 matcher):

| | embedding matcher | LLM matcher |
|---|---|---|
| **options hidden**  | `twostage_semantic_match` (TSP, post-hoc) | `two_prompt` / `two_stage` (TSP/MG/ChoiceBench) |
| **options visible** | `text_extraction` (TSP/MG) | **this experiment** |

## What this is, precisely

Stage 1 is **not** regenerated. This experiment reuses the free-text answers
already produced by the historical `text_extraction` condition (model sees
the question and all options, is told to answer in free text) and adds only
a new Stage 2: a second model call that matches that free text to a lettered
option, using two-stage-prompting's original `option_matching.txt` protocol,
ported verbatim (not `choicebench`'s own version — see
`historical_protocol.py`'s module docstring for the exact, verified
behavioral differences that make the shipped version unsuitable here).

So per question: **0 Stage-1 calls + 1 Stage-2 call**, vs. `text_extraction`'s
1 call and `two_stage`'s 2 calls. The free text going into Stage 2 is
bit-for-bit what `text_extraction` already stored; the Stage-2 mechanism is
bit-for-bit what `two_stage`'s own Stage 2 already does. See
`tests/experiments/test_runner.py::TestIsolationFromSiblingCells` for the
executable proof.

## Files

- `historical_protocol.py` — verbatim port of TSP's `parser.py` (letter
  extraction, text-match fallback, `parse_model_answer`) and
  `build_option_matching_prompt`. Kept separate from
  `choicebench.parsing.parser` / `choicebench.pipeline.prompt_builder`
  because those are NOT behavior-identical (dynamic option rendering that
  drops missing options; stricter token-boundary text matching; letter
  extraction restricted to real option letters). Provenance: TSP
  `two-stage-prompting@b8e784f3eb5d2a727a97eb675140b383a34584fa`.
- `prompts/v1/option_matching.txt` — byte-for-byte copy of TSP's template
  (diffed clean). `direct_mcq.txt`/`free_text.txt` are also copied in,
  unused, only because `choicebench.methods.base.ExperimentRunner.__init__`
  requires all three to exist in one versioned prompt directory — see
  `prompts/v1/README.md`.
- `stage1_sources.py` — registry of the exact historical Stage-1 CSV paths
  per model/benchmark (API: TSP `paper_api_main` run `20260603_154649`;
  local MMLU: TSP `paper_local_main` runs `20260603_213613`/`214857`; local
  ARC: model-generalization run `20260617_162624`, since TSP's own local ARC
  files are stale at 850/1000 rows) plus `load_and_validate_stage1()`, which
  fails closed (raises `Stage1ValidationError`) on: a NaN `choice_d` for any
  of the 3 known 3-option ARC questions without an explicit replacement, a
  NaN `choice_d` for any *other* question (unknown contamination shape), a
  row count != 1000, duplicate `question_id`s, or a null/empty
  `free_text_response`.
- `runner.py` — `VisibleLlmMatcherRunner(ExperimentRunner)`. Takes a
  `stage1_lookup` dict (question_id -> reused Stage-1 row) at construction
  and does Stage 2 only. Not registered in `choicebench.registry` —
  reachable only via the `"module.path:ClassName"` external-method syntax
  documented in the ChoiceBench README, per the instruction to keep this as
  custom experiment code rather than expand the shipped method surface.
- `repaired_stage2.py` + `prompts/v1/option_matching_repaired_3option.txt` —
  narrow, audited-repair-only Stage-2 builder used exclusively for the 3
  known 3-option ARC questions (see next section). Renders A/B/C only —
  never `D. nan`, never an empty D. `historical_protocol.py` itself is
  untouched; its 4-slot output is retained only as provenance evidence and
  is no longer reachable from `runner.py` for these 3 question_ids.
- `arc_question_id_mapping.py` — resolves the 6 ARC question_ids where
  TSP's historical hash disagrees with ChoiceBench's own trusted
  `data/processed/arc_challenge_normalized.csv` hash (see "ARC question_id
  mismatches" below).
- `../../tests/experiments/` — `test_historical_protocol.py` (includes a
  byte-for-byte regression test against a real, already-published
  `two_prompt` Stage-2 prompt), `test_stage1_sources.py` (fail-closed
  behavior), `test_runner.py` (call-count isolation, provenance, the
  variable-option repair, and historical-protocol fidelity for ordinary
  rows), `test_arc_question_id_mapping.py`.

## The 3 contaminated ARC-Challenge questions

`79e8c959bbeb74a0`, `ad6b5d46ae54842c`, `c30e75b011696a95` genuinely have
only 3 real options. TSP's API `text_extraction` Stage-1 prompt-building
code always interpolates all four option slots regardless, so the model was
shown a literal 4th option reading `D. nan` when its free-text Stage-1
answer was elicited for these 3 questions on all 4 API models — a real
prompt defect at elicitation time, not a rendering artifact fixable after
the fact. Reusing that free-text answer as-is would carry the defect into
this cell. `load_and_validate_stage1()` refuses to proceed unless a
corrected replacement is supplied for exactly these 3 IDs.

model-generalization's own local run (`20260617_162624`) already elicited
these 3 rows correctly (its `text_extraction` prompt-builder drops a missing
option instead of interpolating NaN — verified by reading the stored
`prompt` text directly), so the local-model side of this fix already has a
usable corrected source. The API side does not: no corrected Stage-1 free
text exists yet for these 3 IDs × 4 API models.

**How corrected rows will be supplied later** (not implemented in this
phase — no inference has been run): a small CSV with the same schema as
`stage1_sources.REQUIRED_COLUMNS`, one row per (model, contaminated
question_id), produced by re-eliciting Stage 1 for only those 3 questions
per model using a corrected options-visible prompt that omits the missing
4th option (mirroring MG's own `text_extraction.txt` behavior for that
case) rather than TSP's static 4-slot template. That CSV would be passed as
`replacement_rows` to `load_and_validate_stage1()`. For the 2 local models,
`20260617_162624`'s own 3 rows can likely be used directly as that
replacement source instead of new inference — still to be confirmed against
the corrected local ARC set as a whole, not assumed.

**Stage 2 is now repaired too, for these 3 rows only.** `option_matching.txt`'s
static 4-slot template would still render these rows' missing 4th option as
`D. nan` if used unchanged — reintroducing the same phantom-option
contamination one stage later. `runner.py` detects a missing `choice_d` and,
for exactly these 3 audited question_ids, renders through
`repaired_stage2.py`'s dedicated 3-option template instead — real A/B/C
options only. `historical_protocol.build_option_matching_prompt` is
unmodified and still used for every ordinary 4-option row, and its `D. nan`
output for these 3 rows is preserved as provenance/regression evidence in
`test_historical_protocol.py` — it is simply no longer on the production
path for them. A missing `choice_d` on any *other* question_id (i.e. outside
the audited set) makes the runner raise rather than guess.

## ARC question_id mismatches (resolved 3 of 6, 3 blocked)

TSP's own `question_id` (used throughout this experiment, since Stage 1 is
reused as-is) is not the identifier ChoiceBench's Stage-1-freeze/import
workflow expects — that workflow keys everything off ChoiceBench's own
`arc_challenge_normalized.csv` hash. Comparing all 1000 reused ARC rows by
question text + ordered real options + gold answer against that trusted
dataset found 6 mismatches, in two distinct categories
(`arc_question_id_mapping.py`):

**3 resolved (hash-only, content identical)** — exactly the 3 known
3-option questions above. TSP's id-hash formula includes an empty 4th slot
for a missing option; ChoiceBench's omits it entirely from the hash input.
Same question, same 3 real options, same gold answer, different hash
string. Mapped directly:

| TSP `question_id` | ChoiceBench `question_id` |
|---|---|
| `79e8c959bbeb74a0` | `e3d2e85eb821d276` |
| `ad6b5d46ae54842c` | `3d671e2f1721290d` |
| `c30e75b011696a95` | `ed9db36b3d68e1f7` |

**3 blocked — genuine content difference, not a hashing artifact.**
ChoiceBench's trusted ARC dataset has a real 5th option (label E) for these
3 questions that is **absent from every historical TSP row**, including the
already-published `two_prompt`/`text_extraction` cells — traced to
`two-stage-prompting`'s `benchmarks/arc.py`, whose normalizer only ever
extracts labels A–D and has no handling for a 5th option at all. This means
the model was never shown the true full option set for these 3 questions in
*any* historical TSP cell, not just this experiment's reused Stage 1 — a
more severe issue than the phantom-`D. nan` one, and one this experiment
did not create. Per instruction, **not resolved here** —
`translate_tsp_arc_question_id()` raises `MismatchBlockedError` for these:

| TSP `question_id` | ChoiceBench `question_id` | Question | Missing option E |
|---|---|---|---|
| `f87cb129d9aa26c0` | `be30ca6f5bbf0daf` | "How are warm-blooded animals different from cold-blooded animals?" | "Warm-blooded animals are found only in warm climates." |
| `e970a6b50d905595` | `751926ea49e0b65b` | "Sally placed electrodes into a beaker..." | "a hypothesis" |
| `8aec8773c1d6b508` | `97e41313a7c454ea` | "Years ago farmers found that corn plants grew better..." | "water" |

See the top-level report to the user for the explicit stop-and-ask on these
3. The remaining 994/1000 ARC questions already hash identically in both
repos and need no translation.

## Not done in this phase

No YAML run config, no benchmark-grid wiring, no inference call. This is
Stage-2-protocol implementation, the audited 3-option Stage-2 repair,
fail-closed Stage-1 reuse, the ARC question_id mapping, and tests only, per
instruction.
