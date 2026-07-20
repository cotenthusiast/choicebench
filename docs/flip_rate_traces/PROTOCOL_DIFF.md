# Flip-rate traces: structured protocol diff

**v2 correction (supersedes v1).** v1 of this experiment reused ChoiceBench's
own packaged v1 `direct_mcq.txt` template for prompt rendering. Post-hoc
verification against preserved historical data (comparing v1's recomputed
majority votes against the historical aggregate cyclic cells) surfaced an
unexpectedly high disagreement rate on the local/Qwen MMLU cell, which led to
discovering a real, confirmed byte-level bug: ChoiceBench's packaged template
carries an extra trailing newline that neither historical template has
(confirmed via `od -c` against clean checkouts of both source repos). This
went undetected through a full canary pass and broad launch. v1's traces are
preserved as an excluded diagnostic, not deleted — see
`diagnostics_excluded_from_authoritative/flip_traces_trailing_newline_variant/`
in the paper bundle for the full writeup and hashes. v2 (this document, and
`experiments/flip_rate_traces/historical_templates/`) instead loads the two
historical template files verbatim and never touches ChoiceBench's packaged
copy. Fail-closed regression tests now cover every claim in this document
against real historical data: `tests/experiments/test_flip_rate_traces.py`.

Proves that trace retention (plus two explicit, pre-authorized data-hygiene
fixes) is the only intended methodological change versus the historical
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
| Prompt template | v1, hardcoded 4-line "A./B./C./D." block, never drops missing D | v1, dynamic `{options}` block, drops missing/empty options | `historical_templates/api_v1_direct_mcq.txt` / `local_v1_direct_mcq.txt` — verbatim byte copies of each historical repo's own template file (sha256 `bfa10934740cd01c80cf605c318f3107ec9d454a15163ca06b602b8f2fdd0334` / `edf92b7ff58cc627a9c50f61354c3df02a677f30295bda6ab634352c04368245`); `build_api_prompt`/`build_local_prompt` render per-side using each template's own historical structure | **Yes, for cloud side only, and only for the 3 known-contaminated ARC questions** (see below); byte-identical to history for every other question on both sides |
| Missing-option handling | cloud: renders literal `"D. nan"`; local: drops it | (n/a — local already correct) | drops it on both sides; never renders `D. nan`; renders exactly 3 lines / permutes exactly 3 options for `KNOWN_3OPTION_ARC_QUESTION_IDS` | **Yes, for cloud side, explicitly instructed** |
| Parsing (`extract_choice_letter` valid_choices) | cloud: always `("A","B","C","D")` regardless of real option count | local: `options.keys()` (real letters only) | `options.keys()` (matches local; consistent with never allowing a "D" pick on a 3-option question) | Same as local; differs from cloud's historical over-permissive default, which is the same root fix as the D.nan removal |
| Majority vote tie-break | first valid vote, permutation-processing order (`cleaned[0]`) | identical | identical (`historical_protocol.historical_majority_vote`) — **not** choicebench's own current `PermutationRunner._majority_vote`, which uses canonical-letter-order tie-break instead | No (vs. history); intentionally diverges from choicebench's shipped method |
| Per-permutation trace retention | discarded after majority vote; only permutation 0's `raw_text`/`prompt` kept | discarded after majority vote; only permutation 0's kept | every permutation's prompt/raw_text/parse/score/latency/cache/error retained as its own row | **Yes — the intended change** |
| Backend dispatch | N sequential/parallel `generate()` calls | N sequential `generate()` calls | `generate_batch()` (API, never raw `.generate()`) / sequential `generate()` (local) — matches ChoiceBench v0.2.0 Audit Finding 1's correct dispatch pattern | No (methodologically); yes (implementation detail, not a scientific change) |

## Prompt hashes

Verified by `tests/experiments/test_flip_rate_traces.py` against real
historical data, not just template files:

| What | question_id | sha256 |
|---|---|---|
| Historical API template file | — | `bfa10934740cd01c80cf605c318f3107ec9d454a15163ca06b602b8f2fdd0334` |
| Historical local template file | — | `edf92b7ff58cc627a9c50f61354c3df02a677f30295bda6ab634352c04368245` |
| Rendered API ordinary prompt (== historical stored prompt) | `e3335d974df9d047` | `976a506cd3edfe28f393d6156a0f811d0861065726be796325ed34822a3d5a8f` |
| Rendered local ordinary prompt (== historical stored prompt) | `e3335d974df9d047` | `976a506cd3edfe28f393d6156a0f811d0861065726be796325ed34822a3d5a8f` (same text both sides for a full 4-option question — expected, since both templates render identically once no letters are missing) |
| Rendered API repaired 3-option prompt | `79e8c959bbeb74a0` | `b92e5ec247b3eb0b72dbd699b1f1831d9612ff1ff5a3b7198785a1d6b2c9976e` |

Each trace row also carries its own `prompt_sha256` field, independently
recomputable from the row's own `prompt` column.

## Net methodological claim

The only intended scientific-method change is trace retention. Two explicit,
pre-authorized data-hygiene corrections exist on top of that:

1. The `D. nan` removal for the 3 known-contaminated ARC questions — the
   canonical manifest already flags the gpt-4.1-mini × arc_challenge cell as
   `status=malformed_requires_inference` because of exactly this
   contamination, and this experiment's own instructions explicitly require
   not reproducing it.
2. (v2 only) Correcting the trailing-newline bug is not a scientific change
   either — it *removes* an accidental deviation this experiment never
   intended to introduce, bringing prompts back to byte-identical with
   history rather than adding a new difference.

Every other component is byte-identical or logic-identical to the historical
protocol, now covered by fail-closed tests rather than asserted only in prose.
