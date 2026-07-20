# Flip-rate traces: structured protocol diff

Proves that trace retention (plus one explicit, pre-authorized data-hygiene
fix) is the only intended methodological change versus the historical
cyclic-generation-majority protocol. See
`experiments/flip_rate_traces/historical_protocol.py` for full source
citations; this file summarizes the same material as a diff table.

## Historical sources

| Side | Repo @ commit | File |
|---|---|---|
| API (gpt-4.1-mini) | two-stage-prompting @ b8e784f | `src/twoprompt/runners/permutation.py` |
| Local (Qwen2.5-7B-Instruct) | model-generalization @ bb9d3c6 (pre-Eq.1-redesign; NOT current HEAD 24403d8) | `src/modelgen/runners/permutation.py` |

Canonical frozen source CSVs (identity + sha256 in
`model-generalization/paper_data_freeze/manifests/canonical_results_manifest.csv`):

| Cell | Source run | sha256 |
|---|---|---|
| gpt-4.1-mini × arc_challenge | `20260601_183346` | `d7a1d3244294d8a4486581ba377d62187fe7626cf8e000ef0428b69525200ef9` |
| gpt-4.1-mini × mmlu | `20260601_183346` | `dba83af2b8b25db4b689bac61a43e3068f776c028a3aa8e2c608b16180113708` |
| Qwen2.5-7B-Instruct × arc_challenge | `20260529_145812` | `ad7de543bf8bda698c9442f07619e591c4eca7708623f182561ebf82ce08d883` |
| Qwen2.5-7B-Instruct × mmlu | `20260529_145812` | `570016a9c291240c960a4cd2023fa89b662e81e02e8d0fd3804dc1329b9b0843` |

Historical generation settings (read from the canonical CSVs' own columns,
both sides agree): `temperature=0.0, max_tokens=500, seed=42, prompt_version=v1`.

## Diff table

| Component | Historical (cloud) | Historical (local, bb9d3c6) | This runner | Changed? |
|---|---|---|---|---|
| Permutation generation | `dict(zip(keys, values[i:]+values[:i]))`, i=0..N-1 | identical | identical (`historical_protocol.generate_permutations`, imported not reimplemented from scratch — logic verified byte-identical) | No |
| Unpermute (letter→canonical) | text-match scan | identical | identical (`historical_protocol.unpermute_choice`) | No |
| Prompt template | v1, hardcoded 4-line "A./B./C./D." block, never drops missing D | v1, dynamic `{options}` block, drops missing/empty options | choicebench's own v1 `direct_mcq.txt` + `_build_options_block` — dynamic, drops missing options (same as local); byte-identical to cloud template's own text for any question with all 4 real options | **Yes, for cloud side only, and only for the 3 known-contaminated ARC questions** (see below) |
| Missing-option handling | cloud: renders literal `"D. nan"`; local: drops it | (n/a — local already correct) | drops it on both sides; never renders `D. nan`; renders exactly 3 lines / permutes exactly 3 options for `KNOWN_3OPTION_ARC_QUESTION_IDS` | **Yes, for cloud side, explicitly instructed** |
| Parsing (`extract_choice_letter` valid_choices) | cloud: always `("A","B","C","D")` regardless of real option count | local: `options.keys()` (real letters only) | `options.keys()` (matches local; consistent with never allowing a "D" pick on a 3-option question) | Same as local; differs from cloud's historical over-permissive default, which is the same root fix as the D.nan removal |
| Majority vote tie-break | first valid vote, permutation-processing order (`cleaned[0]`) | identical | identical (`historical_protocol.historical_majority_vote`) — **not** choicebench's own current `PermutationRunner._majority_vote`, which uses canonical-letter-order tie-break instead | No (vs. history); intentionally diverges from choicebench's shipped method |
| Per-permutation trace retention | discarded after majority vote; only permutation 0's `raw_text`/`prompt` kept | discarded after majority vote; only permutation 0's kept | every permutation's prompt/raw_text/parse/score/latency/cache/error retained as its own row | **Yes — the intended change** |
| Backend dispatch | N sequential/parallel `generate()` calls | N sequential `generate()` calls | `generate_batch()` (API, never raw `.generate()`) / sequential `generate()` (local) — matches ChoiceBench v0.2.0 Audit Finding 1's correct dispatch pattern | No (methodologically); yes (implementation detail, not a scientific change) |

## Prompt hashes

Computed by `experiments/flip_rate_traces/run_canary.py` canary runs
(`prompt_sha256` field on every trace row) and independently verifiable via
`choicebench.pipeline.prompt_builder.prompt_bundle_identity("v1")`. Canary
runs against real gpt-4.1-mini (see `runs/flip_rate_traces/_canary/`)
confirm: 4 permutations / no `D. nan` for an ordinary ARC question, exactly
3 permutations / no `D. nan` for a `KNOWN_3OPTION_ARC_QUESTION_IDS` member.

## Net methodological claim

The only intended scientific-method change is trace retention. The `D. nan`
removal for the 3 known-contaminated ARC questions is a pre-existing,
already-documented data-hygiene correction (the canonical manifest already
flags the gpt-4.1-mini × arc_challenge cell as
`status=malformed_requires_inference` because of exactly this contamination)
— not a new experimental variable, and explicitly required by this
experiment's own instructions. Every other component is byte-identical or
logic-identical to the historical protocol.
