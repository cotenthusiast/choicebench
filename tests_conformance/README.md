# ChoiceBench Conformance Suite

This suite is an executable specification of the ChoiceBench paper protocol.
Expected scientific behavior comes from the frozen experiment specification
(reproduced in condensed form below, from the task brief given at HEAD
`d5b889f82617ac2876f595dd3d0a1db24cd71506` on `paper/eacl-2026-revision`),
**not** from `src/choicebench`'s existing implementation or `tests/`. Where
this plan cites a production file/function, that citation is only "here is
where to call in", never "here is what the answer should be" — the expected
value in every test comes from the spec or from hand computation.

If a test conflicts with production behavior, the test wins: record the
failure, do not edit `src/choicebench` and do not water down the assertion.

## Implementation status

All 18 sections (A-S) below are implemented and passing: 303 tests total
(293 passed, 10 `xfail(strict=True)` documenting genuine spec violations
found while writing the suite -- see each file's own docstrings/xfail
`reason=` strings for the precise finding and its production location).
Run the whole suite with `python -m pytest tests_conformance/ -v`, and the
call-totals report with `python tests_conformance/report_call_totals.py`.

The section-by-section plan below (written before implementation, per the
task's own Phase 1/Phase 2 process) is kept as a record of intent. A few
sections were implemented more strongly than originally planned, or under
different exact test names -- notably:

- **J (PriDe)**: J1/J2 run the REAL production construction path
  (`load_config` → `load_benchmark_selection` → `load_preflight` →
  `instantiate_runner`) against the actual prepared MMLU/ARC validation-split
  data committed in this checkout, not a synthetic calibration pool as
  originally planned (the real data turned out to be available offline).
- **L (stochasticity)**: exercises `scripts/paper/run_stochasticity_repeats.py`
  end-to-end with a `SpyBackend`-returning monkeypatch of its `build_backend`
  import, matching every L1-L21 sub-item; L21 is the one genuine gap found
  (`compute_agreement_rate` has no `{0,1,2,3}`-completeness check).
- **M (cache identity)**: found and documented a SECOND instance of the
  "Batch bypasses client_extra_identity" bug class, distinct from the
  already-fixed response-cache path: `BatchAPIBackend._batch_state_path()`
  still omits `extra_identity`, so two differently-pinned OpenRouter
  deployments can collide on the same batch-state file.
- **N (batch/sync normalization)**: `BatchAPIBackend.generate_batch()` has no
  length/identity check on `fetch_batch_results()`'s return value before
  zipping it onto `uncached_indices` -- a missing/short result list is
  silently truncated, not a loud failure (N3, xfail). The crash-window
  limitation (N5) is also undocumented in the module (xfail).
- **O (resume equivalence)**: found two real gaps -- conflicting duplicate
  "completed" rows are silently treated as completed (O8), and `--no-resume`
  against a pre-existing output appends duplicate rows rather than starting
  fresh (O9). Both are `xfail(strict=True)`.
- Test file boundaries mostly match the plan 1:1; a few structural checks
  that the plan assigned to a specific section (e.g. J10/J11, PriDe's
  logprob-only structural gates) ended up living in
  `test_production_configs.py` alongside the other config-level structural
  assertions they naturally belong with, cross-referenced from the owning
  section's own file.

## How this plan was built

Per the task brief's process: production source/config *interfaces* were
read first (to learn call signatures, config schema, and result-row shapes)
without reading `tests/` for expected behavior. Two structural corrections
to the spec's assumed architecture came out of that reading and are load-bearing
for the whole plan — see "Architecture corrections" below.

## Architecture corrections (read before writing tests against section D/G)

1. **"Baseline derived from cyclic rotation 0" is not `methods/library/shuffled_baseline.py`.**
   That module (`ShuffledBaselineRunner`) is a *different*, registered method
   (`shuffled_baseline`) that makes its own fresh call under a random
   per-question shuffle — it is not used by the frozen paper configs at all
   (`config/paper/mmlu_core_methods.yaml`'s own header explicitly says
   direct_mcq/baseline is *not* a live method entry). The real "zero-inference
   baseline" the spec means is **`choicebench.analysis.derive_baseline_from_cyclic.derive_baseline_from_cyclic(source_df, *, method_name="direct_mcq")`**
   — a plain function, not an `ExperimentRunner`, taking a saved
   `cyclic_permutation` result DataFrame and re-deriving `direct_mcq` from
   rotation 0's own `raw_text`/`transport_status`, zero new calls, no backend
   parameter at all. Section D tests target this function.

2. **"semantic_matching_v1" is not a registered method either.** It is
   **`choicebench.analysis.derive_from_free_text.derive_matched_results(source_df, *, method_name, embed_fn=None)`**
   — same shape as (1): takes a saved `two_stage` result DataFrame (needs a
   `free_text_response` column), re-derives via
   `choicebench.scoring.text_matcher.match_text_to_options`, zero new calls,
   no backend parameter. Section G tests target this function.

3. **The stochasticity orchestrator is not in `src/choicebench`.** It is
   **`scripts/paper/run_stochasticity_repeats.py`** — a paper-specific script
   whose `run(...)` function, `_load_canonical_obs0_rows(...)`,
   `_validate_execution_mode(...)`, and `_load_completed_pairs(...)` are the
   real targets for section L. It reuses obs0 from a `--canonical-obs0-csv`
   (the main accuracy run's own output) and only calls the model fresh for
   repetitions 1..n-1. It is fully importable and unit-testable (no
   `__main__` side effects on import).

4. **PriDe's calibration-count spec target is `PriDeRunner._calibration_n`**
   (set from the `calibration_n` constructor kwarg, itself threaded from
   `methods[].params.calibration_n` in YAML via `instantiate_runner()`),
   *not* `preflight.n`. Both are frozen to the same value (77 MMLU / 15 ARC)
   in the paper configs, but they are different knobs — `preflight.n`
   controls how many rows are *loaded* as the candidate pool (in
   `choicebench.preflight.load_preflight`), `calibration_n` controls how many
   of the *eligible* rows `PriDeRunner._ensure_calibration()` actually
   samples. Section J tests must exercise both and assert they agree, per
   the frozen config's own comment ("silently calibrates on fewer... if
   left unset").

## File structure

```
tests_conformance/
    README.md                       # this file
    conftest.py                     # shared fixtures: diagnostic benchmark, tmp run dirs, prompt dir
    fakes/
        __init__.py
        spy_backend.py               # SpyBackend(BaseBackend): records every call, controllable per-call responses
        fixtures.py                  # the tiny diagnostic benchmark (Section C) as question_row dicts/DataFrame
    test_production_configs.py       # A
    test_frozen_manifests.py         # B
    test_call_graph_cyclic_baseline.py   # D
    test_permutation_semantics.py    # E
    test_two_stage_protocol.py       # F
    test_semantic_matching.py        # G
    test_text_extraction_matcher.py  # H
    test_ihs_protocol.py             # I
    test_pride_protocol.py           # J
    test_tiebreak_protocol.py        # K
    test_stochasticity_protocol.py   # L
    test_cache_identity.py           # M
    test_batch_sync_normalization.py # N
    test_resume_equivalence.py       # O
    test_metrics_oracles.py          # P
    test_provenance.py               # Q
    test_backward_compatibility.py   # R
    test_call_arithmetic.py          # S
```

Target: ~55-65 tests across these 18 files (task brief's 40-70 range),
weighted toward call-graph/lineage/identity integration tests over shallow
unit tests, per the task brief's stated preference.

## Shared infrastructure

**`fakes/spy_backend.py` — `SpyBackend(BaseBackend)`**

Implements the real `choicebench.backends.base.BaseBackend` interface
(`model_name`, `provider` properties; `generate(prompt, **kwargs) -> str`;
`score_options(prompt, options, **kwargs) -> list[float]`;
`supports_logprobs` property; `is_async_capable()`, default `False` so
`ExperimentRunner._call_backend_generate` uses the plain sync `generate()`
path — real production spy behavior, not a bypass). It:

- Assigns each `generate()` call a deterministic marker (`CALL_001`,
  `CALL_002`, ...) unless a `responder` callback/queue is supplied for that
  call, so tests can either (a) count calls generically or (b) script exact
  per-rotation/per-candidate responses (e.g. "always answer the option whose
  *text* is 'Paris'" for E2/E3-style semantic-vs-positional experiments).
- Records `.calls: list[SpyCall]` with `prompt`, `call_index`, and
  `response_text`, so a test can assert exact call count *and* inspect each
  prompt (e.g. to confirm F2's Stage-1 prompt has no options block).
  `score_options()` calls are recorded separately (`.score_calls`) with
  `(prompt, options, returned_scores)`.
- `raise_on_call: set[int] | Callable` support, to script a transport
  failure on a specific call index (I9, D6-style tests) — raises inside
  `generate()`/`score_options()`, exercised through
  `ExperimentRunner._call_backend_generate`'s existing try/except (never a
  test-side monkeypatch of that method).
- A `supports_logprobs=True` constructor flag (default True, like
  `DummyBackend`) so it can serve `cyclic_logprob`/`pride` tests directly.

**`fakes/fixtures.py` — the Section C diagnostic benchmark**

A small, hand-built list of question dicts in the same normalized-record
shape `ExperimentRunner`/`build_option_map` expect (`question_id`,
`subject`, `question_text`, `choices_json`, `correct_index`,
`correct_option`, `n_choices`), built via
`choicebench.pipeline.options.build_choices`-compatible structure so
`_build_options`/`_build_label_to_source_index` work unmodified on it:

- `q_n4_paris` — 4 options, canonical order `["Paris", "London", "Berlin", "Madrid"]`, gold = A ("Paris").
- `q_n4_none_literal` — 4 options, one option text is literally `"None"`, gold placed elsewhere so a naive string-match bug (matching `pd.NA`/`math.isnan` against the literal text "None") would surface.
- `q_n3_primary` — 3 options.
- `q_n5_wide` — 5 options.
- `q_n4_similar_wording` — two options that are near-duplicates ("A rapid decline in population" / "A rapid decrease in population") to stress the text-matching cascade's containment/cosine stages (G/H).

Each fixture ships a hand-computed table (module-level dict, not derived from
`build_rotations`) of: canonical `source_index` per letter, and for each
rotation index `r` in `0..n-1`, the expected displayed `label -> canonical
letter` map — computed independently by hand (rotation `r` displays
canonical option `(i + r) mod n` at slot `i`, matching `build_rotations`'
documented identity-at-rotation-0 contract) so E1/E2/E3 assertions are
checked against an independently derived oracle, not against
`build_rotations`' own output fed back into itself.

## Section-by-section plan

### A. Production config → runtime propagation (`test_production_configs.py`, ~9 tests)

Targets: `choicebench.config.schema.load_config`, `choicebench.cli.run_experiment.build_backend`, `.instantiate_runner`.

- **A1** — `load_config()` on all 13 files in `config/paper/*.yaml` (parametrized); each must return an `ExperimentConfig` without raising.
- **A2** — For every model entry in every text-generating paper config (all except `*_pride.yaml`), assert `generation_kwargs.temperature == 0.0` and `generation_kwargs.max_new_tokens == 1024`.
- **A3** — For every model entry in every paper config, assert `provider_seed is None`.
- **A4** — Call `build_backend(model_config, run_id="t", run_seed=42)` for the OpenRouter-Llama entry of `mmlu_core_methods.yaml` and the Together-Qwen entry, with `CLIENT_REGISTRY["openrouter"]`/`["together"]` monkeypatched (via `pytest.MonkeyPatch` on the imported dict in `cli.run_experiment`, not by editing `registry.py`) to a fake client class recording constructor kwargs and exposing a stub `async generate(request)`. Assert the resulting `APIBackend._make_request(prompt).seed is None` and that `run_experiment`'s own `run_seed=42` argument never appears as that value.
- **A5** — Build a synthetic `ModelConfig(..., provider_seed=7)` (constructed directly, bypassing YAML, since schema only rejects *unknown* fields) through the same `build_backend` path; assert the resulting request's `.seed == 7`. Proves provider_seed support was not deleted, only defaulted off.
- **A6** — `mmlu_core_methods.yaml` + `mmlu_core_methods_cyclic_batch.yaml` + `mmlu_pride.yaml` models lists, unioned by `model_name_or_path`, equal the 6 frozen conditions exactly (set equality against a hardcoded list from the spec). Repeat for the `arc_*` trio.
- **A7** — `grep`-style: no paper config's raw YAML text contains `gemini`/`groq` (case-insensitive) as a provider value; `CLIENT_REGISTRY` still contains `"gemini"`/`"groq"` (generic support intact, confirms exclusion is config-level not registry-level).
- **A8** — Local model entries (`backend: huggingface`) across all paper configs use exactly `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic` and `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic`, no others.
- **A9** — `arc_core_methods_cyclic_batch.yaml`'s Llama entry is direct DeepInfra (`provider: deepinfra`), never `openrouter` — batch-mode Llama uses a different provider entry than sync Llama, per that config's own header; assert this distinction structurally (not equivalence — equivalence is explicitly out of scope, section "LIVE TESTS").

### B. Frozen manifests (`test_frozen_manifests.py`, ~6 tests)

Targets: `data/manifests/*.csv` directly (plain `pandas.read_csv`), plus `choicebench.stats.n_choices_for_row`.

- **B1** — `mmlu_eval_v2.csv`: 1140 unique `question_id`, `subject` has 57 unique values, each subject has exactly 20 rows. (File has 1141 lines incl. header — confirmed present in this checkout.)
- **B2** — Documented as **not independently regenerable offline** in this checkout (the raw MMLU source dataset is not vendored here); B1's committed-manifest invariants are made mandatory per the task brief's own fallback instruction, and this is listed in section D of the final report ("requires later validation") rather than faked.
- **B3** — `arc_challenge_eval_v2.csv`: 1172 unique `question_id`. (1173 lines incl. header — confirmed present.)
- **B4** — Join `arc_challenge_eval_v2.csv`'s question_ids against the normalized ARC dataframe (loaded via `choicebench.benchmarks.arc`'s loader against `data/processed/` if present in this checkout; if the processed artifact isn't present offline, fall back to computing `n_choices` from the manifest's own `choices_json`/equivalent column if the manifest carries full content, else mark this sub-test `xfail(reason="processed ARC artifact not in clean checkout")` rather than fabricating counts) and assert 1165/4/3/5 distribution reported by `arc_pride.yaml`'s own header comment matches independently.
- **B5** — Load `arc_pride.yaml`'s calibration split (`split: validation`) and the frozen eval manifest's IDs; assert the two ID sets are disjoint (set intersection empty). Same for MMLU.

### C. Diagnostic benchmark — implemented as `fakes/fixtures.py`, exercised throughout D-L; no standalone test file.

### D. Cyclic + baseline call graph (`test_call_graph_cyclic_baseline.py`, ~6 tests)

Targets: `choicebench.methods.library.permutation.PermutationRunner`, `choicebench.analysis.derive_baseline_from_cyclic.derive_baseline_from_cyclic`.

- **D1/D2/D3** (one test, N=4): construct `PermutationRunner(backend=SpyBackend(), method_name="cyclic_permutation", ..., seed=42, benchmark_name="diag")`, call `.run_one(q_n4_paris, 0)`. Assert `spy.call_count == 4`. Feed the single-row result (wrapped in a 1-row DataFrame) to `derive_baseline_from_cyclic(df)`; assert `spy.call_count` is still `4` (no new backend interaction possible — the function takes no backend arg, so this is really a signature/API-shape assertion: `derive_baseline_from_cyclic.__code__.co_varnames` / `inspect.signature` has no backend/client parameter). Assert the derived row's `prompt`/`raw_text`/`transport_status` equal rotation 0's own recorded `prompt`/`raw_text`/`transport_status` from the source row (lineage by field equality against the *same* source row, not against a second computation).
- **D4** — repeat D1-D3 shape for `q_n3_primary` (3 calls).
- **D5** — repeat for `q_n5_wide` (5 calls).
- **D6** — call `derive_baseline_from_cyclic(pd.DataFrame([{"not": "cyclic shaped"}]))` (missing `per_rotation_choices_json`) and assert it raises `ValueError` (per the function's own documented contract) — proves it fails loudly on a non-cyclic source rather than degrading silently.

### E. Permutation / source_index semantics (`test_permutation_semantics.py`, ~7 tests, one per E1-E7)

Targets: `choicebench.pipeline.prompt_builder.build_rotations`, `PermutationRunner`, `fakes/fixtures.py`'s independent oracle table.

- **E1** — For `q_n4_paris`, call `build_rotations(canonical_options)`; for each rotation compare `.mapping` against the fixture's independently hand-computed expected map. Assert canonical `source_index` (via `build_label_to_source_index`) is identical across all 4 `Rotation` objects (it's a property of the question, not the rotation).
- **E2** — `SpyBackend` scripted to always answer the *canonical* correct semantic option regardless of which letter it's displayed at (responder inspects the prompt's option-text-to-"Paris" mapping — the spy is allowed to parse its own input prompt for this, since it's test scaffolding, not production code under test). Run all 4 rotations through `PermutationRunner.run_one`; assert `per_rotation_choices_json` decodes to `["A","A","A","A"]` (all rotations agree on canonical letter A) and derived flip rate (hand-computed: 0 flips / 4) is 0.
- **E3** — `SpyBackend` scripted to always answer the same displayed letter "A" regardless of rotation; assert `per_rotation_choices_json` decodes to 4 *different* canonical letters (one per rotation, since "always displayed-A" tracks a different canonical option each rotation) and hand-compute the resulting flip count from that list.
- **E4/E5** — repeat E1's mapping-correctness check for `q_n3_primary` / `q_n5_wide`.
- **E6** — Assert `question_row["correct_option"]`/`correct_index` are unchanged before/after calling `build_rotations` on the same row (pure function, no mutation) — guards the "changing display never mutates gold" invariant directly rather than inferring it.
- **E7** — Compute rotations in list order vs. reversed order (`for r in reversed(rotations)`) through two independent `PermutationRunner` instances with independently-seeded-identical `SpyBackend`s that respond deterministically by *prompt content* (not call order); assert both produce the same final majority-voted canonical letter and the same per-rotation canonical-choice *set* (order of collection must not matter to the semantic result).

### F. Two-stage rotation semantics (`test_two_stage_protocol.py`, ~7 tests)

Targets: `choicebench.methods.library.two_stage.TwoStageRunner.run_one` / `.run_stage2_rotations`.

- **F1/F2/F3/F4** (one test, N=4): `runner.run_one(q_n4_paris, 0)` against a `SpyBackend`; assert exactly 1 call was made whose prompt is the free-text (Stage-1) prompt, then assert that prompt (via `spy.calls[0].prompt`) contains no lettered option text from the canonical options (string-`not in` check per option text). Take the returned `free_text_response` string and pass it *by value* into `runner.run_stage2_rotations(q_n4_paris, free_text_answer=that_exact_string, sample_index=0)` against a fresh `SpyBackend`; assert exactly 4 new calls, and that all 4 Stage-2 prompts contain that same free-text string verbatim (lineage proved by string containment of the *actual* Stage-1 output, not a re-elicited one — since `run_stage2_rotations` takes no Stage-1 artifact object, passing the same Python string *is* the lineage proof: assert the 4 prompts's `free_text=` interpolation site holds `is`-identical text to Stage 1's `raw_text`). Assert Stage-2 rotation `r`'s prompt's option ordering equals the fixture's independent oracle for rotation `r`.
- **F5/F6** — repeat F1-F4's `run_stage2_rotations` call-count assertion for N=3 (3 Stage-2) and N=5 (5 Stage-2).
- **F7** — Instantiate `TwoStageRunner` with `prompt_version="v1_reasoning"` (real `prompts/v1_reasoning/` dir must contain a `free_text` template — verified as a precondition, `pytest.skip` with a clear reason if the paper branch's reasoning prompt bundle doesn't ship a `free_text.txt`, since building a fake one would defeat the "real prompt" assertion) and assert the Stage-1 prompt differs from v1's own Stage-1 prompt for the same question (proves reasoning bundle is actually wired to Stage 1, not just Stage 2) while the call-graph shape (1+N) is identical to F1.

### G. Semantic matching (`test_semantic_matching.py`, ~6 tests)

Targets: `choicebench.analysis.derive_from_free_text.derive_matched_results`, `choicebench.scoring.text_matcher.match_text_to_options`.

- **G1** — Build a synthetic `two_stage`-shaped DataFrame (columns: the usual question/option columns plus `free_text_response="Paris"`); call `derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=lambda texts: <a small hand-built exact-match-triggering fn>)`. Assert the function signature has no backend/client parameter (same `inspect.signature` technique as D3) — zero target-model calls is a structural property here, not something to spy on.
- **G2** — Build the free-text source row twice with option orderings A and B swapped in `choices_json`, `free_text_response` identical; assert `derive_matched_results` reads `free_text_response` byte-identically regardless of the row's option order (it is not re-derived from anything rotation-dependent).
- **G3** — Same source `free_text_response`, three synthetic rows differing only in canonical option *order* (shuffled `choices_json`); assert the *semantic* (source_index-based) predicted option is identical across all three, even though the *letter* differs.
- **G4** — Construct a row with `free_text_response = float("nan")` (as a real `pandas`-loaded CSV round-trip would produce for an empty cell — write a CSV with an empty field and `pd.read_csv` it back, rather than hand-constructing the NaN, to reproduce the exact real failure mode). Document current behavior precisely: the task brief requires this "not silently become literal string 'nan' and produce a plausible answer" — assert that no exact/containment/cosine hit is silently returned. If current code raises `AttributeError` (a NaN float has no `.strip()`) instead of returning `None`/PARSE_MISSING gracefully, the test asserts the ValueError/AttributeError explicitly (`pytest.raises`) and this is flagged as a NON-BLOCKING finding (fails loudly, just not with a clean exception type) rather than silently adjusted to expect a plausible answer.
- **G5** — `derive_matched_results` has no source-identity validation (confirmed by reading the function — it takes any DataFrame with a `free_text_response` column). Write the test asserting the frozen invariant (wrong-source artifact should be rejected) and `pytest.mark.xfail(strict=True, reason=...)` it, per the task brief's explicit instruction to write it and let it fail rather than skip it.

### H. Text extraction + visible LLM matcher (`test_text_extraction_matcher.py`, ~8 tests)

Targets: `TextExtractionRunner.run_one/.run_rotations`, `VisibleLLMMatcherRunner.run_one/.run_matching_rotations`.

- **H1/H2/H3** — `TextExtractionRunner.run_one` for N=4/3/5 against `SpyBackend`; assert 1 call each (single-call method — the *rotations* variant is what fans out to N, tested separately below to keep this a clean per-N count check as asked).
  - Additional sub-assertion folded in here: `TextExtractionRunner.run_rotations(q_n4_paris, 0)` against a fresh `SpyBackend` makes exactly 4 calls (rotations variant), matching H1's "N=4" framing structurally — this satisfies H1-H3 as "text extraction over N options" while also covering the rotations path the rest of section H depends on.
- **H4** — Build `VisibleLLMMatcherRunner`; construct `question_row` with `extracted_text` already populated (as if joined from a prior `text_extraction` run) and call `.run_one`. Assert **exactly 1** new call (the LLM-match call) — zero *text-extraction* calls, per spec's precise framing ("zero NEW text-extraction calls", not "zero calls total"; the matcher's own 1 call is expected and separately accounted for in section S).
- **H5** — Call `.run_matching_rotations(q_n4_paris, per_rotation_extracted_text=[<4 distinct strings tagged rot0..rot3>], sample_index=0)`; assert each of the 4 generated prompts contains its *same-index* tagged string (rot 2's prompt contains "rot2", never "rot1" or "rot3") — proves same-rotation lineage, not just "some text was reused".
- **H6** — Call `.run_matching_rotations` with `per_rotation_extracted_text` of length 3 for a 4-option question. Read `visible_llm_matcher.py`'s `zip(rotations, per_rotation_extracted_text)` — `zip` silently truncates to the shorter length rather than raising. **This is a genuine finding**: assert the *current* (silent-truncation) behavior explicitly via `pytest.raises` wrapped in `pytest.mark.xfail(strict=True, reason="run_matching_rotations silently zips/truncates on length mismatch instead of raising — see visible_llm_matcher.py's zip(rotations, per_rotation_extracted_text)")`, so the suite goes red the day someone "fixes" this without updating the test, and stays documented as a known gap until then.
- **H7** — Both `TextExtractionRunner`/`VisibleLLMMatcherRunner` accept `question_row["extracted_text"]` with no producer-identity check (confirmed by reading the source — no validation code exists). `pytest.mark.xfail(strict=True, ...)` per the same pattern as G5.

### I. IHS (`test_ihs_protocol.py`, ~10 tests)

Targets: `choicebench.methods.library.independent_hypothesis.IndependentHypothesisRunner`.

- **I1/I2/I3** — `run_one` for N=4/3/5; assert `spy.call_count == n_choices` exactly (never N-1 or a padded 4).
- **I4** — Inspect `spy.calls[i].prompt` for each candidate; assert it contains that candidate's option text but none of the *other* options' text (proves isolation — no ordered full list is shown).
- **I5** — `SpyBackend` scripted with distinct valid `<score>NN</score>` per candidate; assert the highest-scored candidate wins (`parsed_choice`).
- **I6** — All 4 candidates return unparseable text (no `<score>` tag); assert `answer_status == FAILURE_STATUS`/`parsed_choice is None` — no fabricated answer, confirmed via `_assemble_result_row`'s `valid_scores` gate.
- **I7** — 3 candidates return valid scores (60, 70, 80), 1 returns unparseable text; assert the unparseable one is excluded from the argmax entirely (never competes as an implicit 0 — i.e. even if the valid scores were `[5, 3, 2]`, the low-scoring valid 5 must still win over the "invisible" invalid one, proving it's excluded rather than defaulting to a losing 0).
- **I8** — Parametrized over `<score>nan</score>`, `<score>inf</score>`, `<score>-inf</score>`, `<score>150</score>` (out of 0-100 range), `<score>abc</score>` (malformed) — assert each is treated as `parse_ok=False` (via the per-option `option_X_score_parse_ok` column), matching `_parse_confidence_score`'s documented `math.isfinite`/range guard.
- **I9** — Candidate 1 succeeds, candidate 2 raises via `SpyBackend`'s `raise_on_call`. Assert (a) the row's top-level `transport_status` reflects **only** candidate 1 (documented base-class behavior: `model_response=responses[0]`) — this is itself worth pinning down explicitly since it means top-level `transport_status` is *not* an aggregate signal — and (b) the per-option column `option_b_model_status` (or equivalent letter) shows the failure, so the aggregate failure is visible somewhere in the row, just not at top-level `transport_status`. This test exists specifically to pin the "aggregate status semantics" nuance the task brief calls out.
- **I10** — Two candidates score an exact tie (both 90.0); assert the winner is whichever `resolve_tie(...)` (imported directly, computed independently in the test) would pick for that `(seed, benchmark, question, method, tied_ids)` tuple — not first-in-list.

### J. PriDe calibration (`test_pride_protocol.py`, ~11 tests)

Targets: `PriDeRunner`, `choicebench.preflight.load_preflight`, `choicebench.config.schema.load_config` on `mmlu_pride.yaml`/`arc_pride.yaml`.

- **J1** — `load_config("config/paper/mmlu_pride.yaml")`; take `config.methods[0].params["calibration_n"]` (=77 per file) and construct a real `PriDeRunner(backend=SpyBackend(supports_logprobs=True), ..., calibration_n=config.methods[0].params["calibration_n"], require_full_calibration=True, calibration_questions=<77+ synthetic modal-k=4 rows>)`. After `.run_many([...])` triggers `_ensure_calibration()`, assert `runner._calibration_n == 77` **and** that the calibration state's `estimation_question_ids` (via the sidecar or `runner._calibration_state.estimation_question_ids`) has length 77 — i.e. assert the *runtime-selected* count, not just the constructor echo.
- **J2** — same for `arc_pride.yaml`, expect 15.
- **J3** — Build a synthetic MMLU-shaped calibration pool of 100 modal-k=4 rows; construct `PriDeRunner(calibration_n=77, ...)`; assert `_pick_calibration_rows`-selected IDs (via the written sidecar JSON's `calibration_question_ids`) has exactly 77 entries, all drawn from the 100-row pool.
- **J4** — Build a synthetic ARC-shaped pool: 12 four-option rows + 5 three-option + 3 five-option (20 total, mirroring `arc_pride.yaml`'s real 295/299/15 ratio in miniature) with `calibration_n=12` (all eligible rows, to make "eligible pool == calibration_n" the discriminating case). Assert eligibility filtering happens **before** sampling: run with `_modal_k=4` and assert all 12 selected rows have `n_choices==4` (none of the 3-/5-option rows ever appear).
- **J5** — The discriminating fixture: 15 four-option rows + 10 three-option rows (25 total), `calibration_n=15`, seeded such that if sampling happened *before* filtering (sample 15 of 25 raw, then filter to 4-option) fewer than 15 four-option rows would survive for at least one seed. Run with real seed 42 and assert exactly 15 eligible rows are used (proves filter-then-sample order empirically, not just by reading the source).
- **J6** — From J3's run, assert every selected calibration row (via the sidecar) has `n_choices == 4` (redundant with J4 at the ARC level; here asserted generically against whatever `_modal_k` was configured).
- **J7** — Run `_pick_calibration_rows` (imported directly) twice with the same pool/seed=42; assert identical selected IDs. Run again with seed=43; assert a different selection (sanity that the seed actually matters).
- **J8** — `PriDeRunner(calibration_n=77, require_full_calibration=True, calibration_questions=<only 10 eligible rows>)`; assert `.run_many([...])` raises `RuntimeError` (per `_ensure_calibration`'s documented guard) rather than silently calibrating on 10.
- **J9** — Build calibration pool and evaluation pool from the same synthetic source with disjoint `question_id` ranges; assert the constructed sets truly don't intersect (a fixture-level assertion, since `PriDeRunner` itself does not check this — the disjointness contract lives in `choicebench.preflight.load_preflight`'s `eval_question_ids` exclusion, tested here via `load_preflight` directly with a `MethodConfig(preflight=PreflightConfig(...))` and an `eval_question_ids` set overlapping the benchmark pool, asserting the overlapping IDs are excluded from the returned selection).
- **J10** — Assert `PriDeRunner(backend=SpyBackend(supports_logprobs=False), ...)` raises `ValueError` at construction (the `if not backend.supports_logprobs: raise ValueError` guard) — structural proof that PriDe requires a logprob-capable (HF-shaped) backend and cannot run against a generate-only API-shaped one, without needing a GPU or real HF weights.
- **J11** — `grep`-equivalent: `inspect.signature(PriDeRunner.__init__)` has no parameter named `alpha`/`alpha_*`, and `config/paper/*pride*.yaml` contain no `alpha` key anywhere in the parsed YAML tree (recursive key-name walk).

### K. Tie-break protocol (`test_tiebreak_protocol.py`, ~8 tests)

Targets: `choicebench.scoring.tiebreak.resolve_tie`, `.majority_vote_with_tiebreak`, exercised both directly and through `PermutationRunner`'s real tie boundary.

- **K1** — `resolve_tie(seed=42, benchmark_id="mmlu", question_id="q1", method_name="cyclic_permutation", tied_canonical_ids=[0,2])` called twice — the model name is not a parameter at all, so this is really "assert the function signature excludes model" plus a same-inputs-same-output determinism check; additionally run `PermutationRunner` with two different `model_label` values (only `model_label`, everything else equal) into an actual induced tie and confirm the voted *canonical* letter is identical.
- **K2** — Same, varying only `backend.provider` on two `SpyBackend` instances.
- **K3** — Induce the same canonical tie under two different rotation orderings (via scripted `SpyBackend` responses keyed by canonical semantic content, not display letter) and confirm the same canonical winner.
- **K4** — Call `majority_vote_with_tiebreak` with the tied-choices list in two different orders (`[A,B,A,B]` vs `[B,A,B,A]` — same multiset); assert identical winner (order-independence of `Counter`-based counting plus `resolve_tie`'s sorted-ids contract).
- **K5** — Two different tied sets (`{0,2}` vs `{1,3}`) for the same `(seed, benchmark, question, method)`; assert each winner is a member of its own tied set (not necessarily related to each other).
- **K6** — Directly assert `independent_hypothesis.py`'s `_argmax_with_tiebreak` calls `resolve_tie` (not `np.argmax`) by inducing an exact score tie and checking the winner matches `resolve_tie`'s independently-computed output for that tuple, rather than "whichever option's confidence text happened to parse first".
- **K7** — Same technique for IHS as K6 (may be the same test as K6 — kept separate only if the assertion differs meaningfully; otherwise merge).
- **K8** — Run the same tie twice in two fresh Python subprocesses (via `subprocess.run([sys.executable, "-c", ...])`, or in-process with `PYTHONHASHSEED` forced to two different values via env + subprocess, since `hash()` randomization is only observable across processes) and assert identical output — proves independence from `hash()`/process-level randomization, consistent with BLAKE2b (not `hash()`) being used.

### L. Stochasticity (`test_stochasticity_protocol.py`, ~14 tests)

Targets: `scripts/paper/run_stochasticity_repeats.run`, `._load_canonical_obs0_rows`, `._validate_execution_mode`, `choicebench.analysis.agreement.compute_agreement_rate`. Import via `importlib` against the repo-root `scripts/paper/run_stochasticity_repeats.py` (add `scripts/paper` to `sys.path` in `conftest.py`, or import as a module by file path — it has no package `__init__.py`, confirm during implementation and use whichever import mechanism the existing `scripts` tests already use, checked in Phase 4).

- **L1** — Build a 1-row canonical obs0 CSV with matching identity columns; call `run(..., n_repetitions=4, ...)` with a `build_backend` monkeypatch that raises if ever called with `model_identity` containing `"rep0"` (obs0 must never reach `build_backend` at all — the real code path for repetition 0 is pure CSV reuse, never a backend call).
- **L2** — Same setup, monkeypatch `build_backend`/`runner_cls` to a `SpyBackend`-backed real `DirectMCQRunner`; assert exactly 3 fresh calls total (reps 1,2,3) for 1 question.
- **L3** — Assert the fresh-repetition prompts (reps 1-3) use canonical (unrotated) option order — compare against `build_direct_mcq_prompt` called directly with canonical options for the same question.
- **L4/L5** — Assert `build_backend` is invoked with 3 distinct `model_identity` strings (`stochasticity_direct_mcq_rep1/2/3`) via a spy wrapper around the monkeypatched `build_backend`; since each identity maps to its own `cache_dir` (per `build_backend`'s own logic, already verified in section A), this is the correct level to assert distinctness at (not re-deriving cache-dir logic here).
- **L6** — Call `run(...)` once (writes reps 0-3 to `output_path`), then call it again with `resume=True` (default) unchanged; assert the second call's return value (`n_written`) is `0` and the output CSV still has exactly 4 rows for that question (no `obs4`, no duplicated 1-3).
- **L7** — Canonical obs0 CSV has `model_name` different from `expected_identity["model_name"]`; assert `_load_canonical_obs0_rows` raises `ValueError`.
- **L8** — Same for `provider` mismatch (not in the literal `_OBS0_IDENTITY_COLUMNS` list as a separate "deployment" field — confirmed the tuple is `("provider", "model_name", "benchmark_name", "prompt_version")` — so "wrong provider/deployment" is exactly the `provider` column; test that column specifically since there is no separate deployment-pinning column at this layer).
- **L9** — `benchmark_name` mismatch → `ValueError`.
- **L10** — This layer has no `method_name` mismatch case in the wrong direction to test structurally (the function already filters `canonical_df` to `method_name` before matching, so a wrong-method canonical file just looks like "missing question_ids" — assert **that** manifestation: a canonical CSV containing only rows for `method_name="two_stage"` used for a `method_name="direct_mcq"` run raises `ValueError` for missing question_ids, which is functionally the same hard-failure guarantee via a different message).
- **L11** — `prompt_version` mismatch, or `temperature`/`max_tokens` mismatch (parametrize both the string and numeric identity columns) → `ValueError`.
- **L12** — Canonical CSV has two rows for the same `question_id` under the target `method_name`; assert `ValueError` (never `keep="first"`).
- **L13/L14** — `method_name="two_stage"`, `n_repetitions=4`; monkeypatch `build_backend` to real `SpyBackend`s; assert 3 fresh Stage-1 + 3 fresh Stage-2 calls total across reps 1-3, and that (via prompt inspection, tagging each rep's Stage-1 output distinctively through the `SpyBackend`'s responder) rep 2's Stage-2 prompt contains rep 2's own Stage-1 output string and not rep 1's or rep 3's.
- **L15** — `run(..., method_name="two_stage", execution_mode="batch")` → assert `ValueError` from `_validate_execution_mode` before any backend is touched.
- **L16** — same for `reasoning_two_stage`.
- **L17/L18** — `_validate_execution_mode("direct_mcq"/"reasoning_mcq", "batch")` does **not** raise (both directions of the sync-only set asserted explicitly, not just "one raises").
- **L19** — Assert the `model_config` object passed through to every `build_backend`/runner call across reps 0-3 is the *same* `ModelConfig` instance (`is` identity) the caller passed to `run()` — proves no per-repetition seed mutation is possible by construction, stronger than reading `provider_seed` off the result rows (which the runner doesn't even record as a column).
- **L20** — `compute_agreement_rate` on a hand-built 4-row `[A,A,A,A]` group → `agreement_rate == 1.0`; `[A,A,A,B]` → `0.0` (only one group, so rate is exactly 0 or 1 — use two groups to get a fractional value, e.g. one unanimous + one split → `0.5`).
- **L21** — **Genuine finding, written to fail**: build a group with only 3 observations (`sample_index`/repetition values 0,1,2 — no 2 missing... actually construct explicitly `question_id` group with rows for repetitions `{0,1,3}` (2 missing) and a second group with a duplicate repetition `{0,1,2,3,3}`; call `compute_agreement_rate`. Current implementation only checks `len(group) >= 2` — it does **not** validate the *set* of repetition indices at all. `pytest.mark.xfail(strict=True, reason="compute_agreement_rate groups by question_id and only checks len>=2; it never validates that the observation-index set is exactly {0,1,2,3}, so a group missing obs2 or carrying a duplicate obs3 is silently treated as a valid comparison group — see agreement.py's groupby loop")`.

### M. Cache / scientific identity (`test_cache_identity.py`, ~7 tests)

Targets: `choicebench.infra.cache._cache_key`, `.client_extra_identity`, `.CachingClientWrapper`.

- **M1** — Two identical `ModelRequest`s (same provider/model_name/payload/temperature/max_tokens/seed) → identical `_cache_key`.
- **M2** — Two `ModelRequest`s identical except one is keyed with `extra_identity={"upstream_provider": "deepinfra", "allow_fallbacks": False}` and the other `None` → different `_cache_key`. Also: two different `upstream_provider` values with everything else equal → different keys.
- **M3** — Same request, `allow_fallbacks` True vs False (via `client_extra_identity` on two fake client objects with `_upstream_provider="deepinfra"` and differing `_allow_fallbacks`) → different keys.
- **M4** — **Regression test for a previously-fixed bug** (per the module's own comments and recent git log: `d5b889f`/earlier commit "include OpenRouter upstream pinning in cache/model identity"). Construct a fake OpenRouter-shaped client (`_upstream_provider`, `_allow_fallbacks` attrs set) and drive the *same* uncached prompt through both `CachingClientWrapper.generate()` (sync path, via a thin asyncio wrapper) and `BatchAPIBackend.generate_batch()` (with a fake batch-capable client stub for `submit_batch`/`poll_batch`/`fetch_batch_results`) against the *same* `ResponseCache` instance; assert both computed the identical `_cache_key` for the identical logical request — i.e. assert equality of keys, not just that each individually "includes" the field, so a future regression that silently reintroduces divergence is caught.
- **M5** — Response cached under identity A (`upstream_provider="deepinfra"`); a request with identity B (`upstream_provider="together"` or `None`) for the byte-identical prompt/model_name must miss the cache (`ResponseCache.get(key_B) is None`).
- **M6** — Sync `APIBackend` and `BatchAPIBackend` constructed with the same `provider`/`model_name`/`temperature`/`max_tokens`/`seed` and no OpenRouter pinning (`client_extra_identity` returns `None` for both) → identical `_cache_key` for the same prompt (transport alone does not redefine scientific identity when extra_identity is absent) — while `BatchAPIBackend`'s separate `_batch_state_path` hashing (operationally-isolated job-resumption key, not the response-cache key) is asserted to be a *different* hash namespace (different function, different key material) so batch state isolation is confirmed without conflating it with M1's identity claim.
- **M7** — A cache record written under the *old* schema (no `namespace` field, or `schema_version` absent) is read back via `ResponseCache.get()`; assert it returns `None` (miss, via `_is_compatible_record`'s schema-version gate) rather than being misinterpreted as a hit for a new routed condition.

### N. Batch/sync result normalization (`test_batch_sync_normalization.py`, ~5 tests)

Targets: `BatchAPIBackend`, `APIBackend`, using fully fake provider clients (stub `submit_batch`/`poll_batch`/`fetch_batch_results`/`generate`) — no real provider SDK involved.

- **N1** — A fake client returns semantically-equal sync (`generate`) and batch (`fetch_batch_results`) responses for the same prompt; run both through their respective backends; assert the resulting `ModelResponse.raw_text`/`.status`/`.is_success()` normalize identically.
- **N2** — Fake `fetch_batch_results` returns results in **reversed** order relative to the submitted `requests` list (each carrying its own identity via the request object itself, since `ModelRequest` has no explicit id field — reassembly is the *client's* job per the module's own docstring, so this test's fake client does its own reordering-by-payload-match internally, and the test asserts `BatchAPIBackend.generate_batch()`'s **caller-facing** order matches the original `prompts` list order it was given, which the real code preserves via `results[local_i] = response` indexed by `uncached_indices`).
- **N3** — Fake `fetch_batch_results` returns fewer results than requests (simulating a missing batch result ID); assert the zip/assignment in `generate_batch` raises rather than silently returning a short list — if it currently does *not* raise (e.g. Python's `zip` truncates silently again, matching the H6 pattern), write the test to the frozen invariant and let it `xfail(strict=True)` per the standard pattern used elsewhere in this plan.
- **N4** — Pre-populate a batch-state file (matching `_batch_state_path`'s real hash for a given uncached request set) with a fake `batch_id`; call `generate_batch` with a fake client whose `submit_batch` raises `AssertionError` if called; assert `generate_batch` completes via `poll_batch`/`fetch_batch_results` alone (resume path used, no duplicate submission).
- **N5** — Not a test: documented in the final report's "cannot be implemented offline" section (the crash window between provider acceptance and local state-file write is explicitly out of scope per the module's own docstring and the task brief's "LIVE TESTS ARE OUT OF SCOPE" section).

### O. Resume equivalence (`test_resume_equivalence.py`, ~8 tests)

Targets: `PermutationRunner`, `TextExtractionRunner`, `TwoStageRunner`, `VisibleLLMMatcherRunner`, `scripts/paper/run_stochasticity_repeats.run`, `choicebench.infra.resumable_csv.check_resume_compatible`.

- **O1** — Run `PermutationRunner.run_one` uninterrupted (N=4, `SpyBackend`) → row A. Simulate "interrupted after 2 rotations" by manually invoking the runner's rotation-building + first-2-calls, discarding, then running the *full* `run_one` fresh (since `PermutationRunner.run_one` has no built-in mid-question checkpointing — it's atomic per question; the realistic resume boundary in this codebase is at the **question** level via `check_existing_result`/output-CSV row presence in `run_experiment.py`, not mid-rotation) → row B. Assert row A == row B on every scientifically meaningful field. This test also documents (in a comment) that true mid-rotation resume is not a supported unit of resumption in this codebase, so O1 is really "re-running an uninterrupted question is idempotent", which is the closest faithful reading of the invariant at the granularity this codebase actually supports.
- **O2/O3/O4** — same idempotence-at-question-granularity pattern for `TextExtractionRunner.run_rotations`, `TwoStageRunner.run_stage2_rotations`, `VisibleLLMMatcherRunner.run_matching_rotations`.
- **O5** — `scripts/paper/run_stochasticity_repeats.run(...)` called with a fake backend that fails (raises) on rep 3 mid-way through multiple questions, leaving reps 0-2 written; call `run(...)` again (resume=True); assert the final file has exactly one row per `(question_id, repetition_index)` in `{0,1,2,3}` per question, no duplicates, rep 3 now present.
- **O6** — `check_resume_compatible(path, exact_columns=[...])` against a file with an extra/missing column → `ValueError`.
- **O7** — `check_resume_compatible(path, required_columns=[...], expected_method_name="direct_mcq")` against a file whose `method_name` column contains `"two_stage"` → `ValueError`.
- **O8** — Build a "conflicting duplicate completed rows" CSV (two rows, same `question_id`, different `is_correct`) and assert whichever consumer reads it (`_load_canonical_obs0_rows`'s ambiguous-duplicate check, reused here as the concrete instance of this general invariant, since `derive_baseline_from_cyclic`/the rotation runners have no separate "duplicate completed row" concept of their own — documented as such) raises rather than collapsing to one row.
- **O9** — `run_stochasticity_repeats.run(..., resume=False)` semantics: reading the source, `resume` only gates `_load_completed_pairs` (empty set if `resume=False`) — it does **not** delete or overwrite the existing output file; `_append_rows_with_schema_union` always appends/concats. Assert this *actual* behavior explicitly (calling with `resume=False` against a pre-populated file appends a second full set of rows rather than starting fresh) — this is a real finding worth flagging (NON-BLOCKING: `--no-resume` semantics may not be "start fresh" as a naive reading of the flag name would suggest) rather than silently assuming the friendlier interpretation.

### P. Metric oracles (`test_metrics_oracles.py`, ~10 tests)

Hand-built tiny `pd.DataFrame`s in every case; expected values computed by the test author independently (by arithmetic / `scipy.stats.beta.ppf` called directly with hand-picked `k,n`, not by calling `choicebench.metrics.accuracy._clopper_pearson`).

- **P1** — 5 rows: 2 correct, 2 wrong, 1 `parsed_choice=NaN`. `Accuracy().compute(df)["accuracy"] == 2/5`.
- **P2** — same df, `accuracy_conditional == 2/4` (denominator excludes the 1 unscorable row).
- **P3** — parse/unscorable rate = `1/5` hand-computed from the same df (not a metric class output — computed directly from `df["parsed_choice"].isna().mean()` as the oracle, then optionally cross-checked if a rate is exposed anywhere; if no production "parse rate" metric exists, this test computes and asserts the rate as a pure `pandas` oracle, documented as such).
- **P4** — `k=2, n=5` → compare `Accuracy`'s CI against `scipy.stats.beta.ppf` called directly in the test with the textbook Clopper-Pearson formula (independent of `_clopper_pearson`'s own code, even though it will incidentally be the same call shape — the point is the test author derives the formula from the definition, not from reading `accuracy.py`).
- **P5** — 4 rotations' canonical predictions `[A, A, B, A]` for a question whose gold is A; hand-compute flip rate = fraction disagreeing with... (define precisely: per `order_sensitivity.py`'s actual metric name/definition, read just before writing this test in Phase 2 — flagged here as needing the metric's exact definition, not assumed).
- **P6** — 4 gold-position groups with recalls `[1.0, 0.5, 0.0, 0.5]` (as percentages: `[100, 50, 0, 50]`); population SD (`ddof=0`) = hand-computed `numpy.std([100,50,0,50])` computed inline in the test (allowed here since `np.std` is a trusted primitive, not the function under test) → assert `RecallRStd`'s `rstd` matches when fed a df engineered to produce exactly those 4 per-letter recalls.
- **P7** — ARC-shaped df with 4-, 3-, and 5-option rows mixed; assert `RecallRStd.compute()`'s label set / recall computation only reflects the 4-option rows' gold letters — construct so a 3-/5-option row's gold letter, if wrongly included, would change the answer, making this a discriminating test rather than a coincidentally-passing one.
- **P8** — Call `RecallRStd._bootstrap_std` (or `.compute()`'s `rstd_std` field) twice on the same df; assert byte-identical output (determinism under the hardcoded seed=42). Separately, patch `_N_BOOTSTRAP` via `monkeypatch.setattr` to a small number (e.g. 50) and assert the resample count used matches (via `numpy.random.default_rng` call-counting or by checking `idx.shape` — whichever is observable without reading bootstrap internals further than already done) — and confirm from the already-read source that resampling is row-level (`rng.integers(0, n, size=(_N_BOOTSTRAP, n))` indexes into per-row arrays) rather than per-position-cell.
- **P9** — Two tiny paired dfs (`results_a`, `results_b`), 4 questions: both correct (2), baseline-only correct (1), method-only correct (1), both wrong (0) — hand-pick `is_correct` columns to realize exactly these counts; call `mcnemar_exact_test(results_a, results_b)` and assert its reported contingency counts equal the hand-built table (not its p-value/significance verdict, which the task brief defers).
- **P10** — Hand-built stochasticity-shaped df (5 questions x 4 observations, 3 unanimous + 2 not) → `compute_agreement_rate(...)["agreement_rate"] == 3/5` exactly.

### Q. Provenance / derived artifacts (`test_provenance.py`, ~6 tests)

Targets: `ExperimentRunner._build_result_row` (the actual schema, already read in full — no separate "provenance module" exists for row-level identity; `choicebench/provenance.py` is model/dataset *content*-identity, a different concept, exercised only incidentally if at all).

- **Q1** — Build a result row via a real runner (`DirectMCQRunner`-equivalent, i.e. any single-call `ExperimentRunner` subclass reachable without extra setup — likely `TextExtractionRunner.run_one`) and assert the row dict contains non-null values for: `model_name`, `provider`, `benchmark_name`, `question_id`, `method_name`, `prompt_version`, `temperature`, `max_tokens`, `parsed_choice`/`answer_status`. No separate display-mapping column exists at this layer for single-call methods (rotation methods carry `per_rotation_choices_json` instead) — assert its *absence* here is expected, not a gap.
- **Q2** — A `derive_baseline_from_cyclic` output row has `derived_from_run_id`/`derived_from_method_name` populated and equal to the source row's own `run_id`/`method_name`.
- **Q3** — Same row's `method_name` is overwritten to `"direct_mcq"` while `derived_from_method_name == "cyclic_permutation"` — proves it doesn't "masquerade" as a genuinely-collected direct_mcq row; a consumer joining on `method_name` alone would still need `derived_from_*` to know it's synthetic, which the row provides.
- **Q4** — A `VisibleLLMMatcherRunner` row carries `reused_extracted_text` equal to the source `text_extraction` row's own extracted text (already covered partly by H4/H5 — this test asserts the specific *column name/presence* contract rather than the call-count).
- **Q5** — A `derive_from_free_text` row built from a source with `free_text_response=None` has `normalized_text is None` (not the string `"None"` or `"nan"`) — direct assertion on `_derive_one_row`'s output dict.
- **Q6** — Construct (by hand, not by driving a runner into this exact state — it may not be reachable through any single real code path) a row with `is_correct=True` and `transport_status=FAILURE_STATUS`/`error_type` set, and check whether any consumer in the codebase (e.g. `Accuracy.compute`) would treat it inconsistently; if no schema-level guard exists to prevent constructing such a row in the first place, `xfail(strict=True)` documenting the absence of a coherence check, per the standard pattern.

### R. Backward compatibility (`test_backward_compatibility.py`, ~5 tests)

- **R1** — `load_config("config/toy_experiment.yaml")` (or `config/experiment_template.yaml`, whichever is a plain pre-revision-style config with a `backend: dummy` model — confirmed to exist in this checkout) parses successfully and can build a `DummyBackend` + `instantiate_runner` for its own `direct_mcq`-or-equivalent method, run one `run_one` call, and produce a well-formed result row — end-to-end through the real offline (`dummy`) path.
- **R2** — Build two `ModelConfig`s identical except `provider_seed` unset (None, new default) vs explicitly `provider_seed=None` (old-style explicit); assert `build_backend` produces byte-identical `APIBackend._make_request(...)` output for both (the new field's mere existence changes nothing when unset).
- **R3** — A non-OpenRouter client (`client_extra_identity` returns `None`) produces the *same* `_cache_key` it would have before `extra_identity` existed as a concept — i.e. `_cache_key(request)` with no `extra_identity` kwarg at all equals `_cache_key(request, extra_identity=None)` — proving the parameter is additive, not identity-changing, for every client that doesn't opt in.
- **R4** — `PriDeRunner(calibration_n=5, require_full_calibration=False (default), calibration_questions=<2 rows>)` degrades gracefully (falls back toward a uniform-influenced state, no exception) — proving the new strict-calibration behavior is opt-in (only triggers `RuntimeError` when `require_full_calibration=True`, matching J8's contrast case).
- **R5** — Meta-test: assert every test file in `tests_conformance/` that touches the filesystem uses a `tmp_path`/`tmp_path_factory` fixture for its I/O root (implemented as a `conftest.py`-level guard/convention documented here, checked by code review during Phase 4/5 rather than by a runtime assertion — no test in this suite writes into `runs/`, `data/`, or any other tracked directory).

### S. Paper-spec call arithmetic (`test_call_arithmetic.py`, ~6 tests)

- **S1** — Parametrized over N=3/4/5, assert via the *actual* spy call counts collected across sections D-L (re-run the minimal shape here for self-containment rather than importing other test modules' fixtures across files) that: cyclic=N, baseline-derived=0, two_stage=1+N (Stage-2-rotations shape) or 2 (plain `run_one`, both shapes asserted separately since they're genuinely different call counts for different questions), semantic=0, text_extraction=N (rotations) or 1 (plain `run_one`), visible_matcher=0 new text-extraction + N matcher calls (rotations) or 1 (plain `run_one`), IHS=N, stochasticity_baseline=3, stochasticity_two_stage=6 (3 Stage1 + 3 Stage2).
- **S2** — Compute expected production totals from `data/manifests/mmlu_eval_v2.csv` (1140q) and `data/manifests/arc_challenge_eval_v2.csv` (1172q) against the 6 frozen conditions and the retained-method list, using S1's formulas. Report the resulting numbers in the Final Report's section F rather than asserting them against any pre-existing documented total (the task brief explicitly forbids copying the old 69,353/72,953 figures) — implemented as a test that computes and `print()`s/attaches the total to a JUnit property or just asserts the computed total is a positive integer matching a value computed twice via two independent code paths in the test itself (e.g. sum-of-per-method vs. sum-of-per-benchmark), catching an arithmetic slip in the test itself rather than validating against an external number.

## Fixtures / fakes not yet fully specified (resolved during Phase 2)

- Exact import mechanism for `scripts/paper/run_stochasticity_repeats.py` (no `__init__.py` under `scripts/`) — check how `tests/scripts/` (if present) already imports sibling scripts before inventing a new mechanism.
- Exact metric name/definition for flip rate (P5) — read `order_sensitivity.py`'s `compute()` body before writing that one test (deferred, not part of the interface-only Phase-1 read).
- Whether `data/processed/` ARC artifacts exist in this checkout for B4; confirmed only that `data/manifests/*.csv` exist so far.

## Review focus (most likely to bite, checked while implementing)

1. **PermutationRunner's tie-break key uses `question_id`, not a synthetic diagnostic-fixture ID collision** — the hand-built fixture question IDs must be realistic strings (not integers) since `resolve_tie` does `str(question_id)`; a fixture using bare ints could mask a str-vs-int mismatch bug. Use string IDs throughout `fakes/fixtures.py`.
2. **`SpyBackend.generate()` must be called synchronously** for sync-path tests (`ExperimentRunner._call_backend_generate` checks `is_async_capable()`) — if `is_async_capable()` defaults True by accident, tests silently route through `asyncio.run(generate_single_async(...))`, which `SpyBackend` doesn't implement, causing confusing `AttributeError`s rather than a clean call-count assertion failure. Default `is_async_capable() -> False` explicitly in `SpyBackend`.
3. **`build_option_map`/`build_choices` require exact keys** (`question_id`, `subject`, `question_text`, `choices_json`, `correct_option`, `n_choices`, ...) — a fixture dict missing one will raise `KeyError` deep inside `_build_result_row` with a confusing traceback pointing at the wrong layer. Validate the fixture shape against a real benchmark-loaded row's keys (e.g. from `data/manifests/`) once during `conftest.py` setup, not per-test.
4. **`derive_baseline_from_cyclic`/`derive_from_free_text` operate on DataFrames, not row dicts** — every section-D/G/Q test building a "source row" must wrap it in `pd.DataFrame([row])`, not pass a bare dict; a bare-dict call will raise on `.columns` access with a message that doesn't obviously point at this mistake.
5. **PriDe's `require_full_calibration` default is `False`** — any J-section test that expects a hard failure (J8) must pass `require_full_calibration=True` explicitly, or it will silently degrade instead of raising, and the test will fail for the wrong reason (assertion on exception type, not a genuine production-behavior finding).

## What Phase 2 will NOT attempt (per task brief's explicit scope)

Real OpenAI/Anthropic/Together/DeepInfra Batch behavior and shapes, OpenRouter→DeepInfra vs direct-DeepInfra deployment equivalence, H100/FP8 loading, real provider nondeterminism, token billing, free-credit behavior. These are listed in the Final Report's "LIVE VALIDATION REQUIRED" section, not stubbed to look tested.
