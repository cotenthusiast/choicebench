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
- `../../tests/experiments/` — `test_historical_protocol.py` (includes a
  byte-for-byte regression test against a real, already-published
  `two_prompt` Stage-2 prompt), `test_stage1_sources.py` (fail-closed
  behavior), `test_runner.py` (call-count isolation, provenance, historical
  quirk preservation).

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

Note Stage 2 is unaffected either way: `option_matching.txt`'s static
`option_a`..`option_d` slots still render the corrected rows' missing 4th
option as `D. nan`, exactly matching `two_prompt`'s own published Stage 2
for these same 3 questions. See `historical_protocol.py`'s module docstring
for why that is intentional, not a bug to fix.

## Not done in this phase

No YAML run config, no benchmark-grid wiring, no inference call. This is
Stage-2-protocol implementation plus fail-closed Stage-1 reuse plus tests
only, per instruction.
