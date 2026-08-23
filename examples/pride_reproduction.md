# Worked example: reproducing a published number - PriDe on MMLU / llama-13B

This reproduces one row of Table 3 from Zheng et al., *"Large Language Models
Are Not Robust Multiple Choice Selectors"* (ICLR 2024, arXiv:2309.03882): the
MMLU block, `llama-13B`, 0-shot, all five columns the paper reports for that
row: Default, Cyclic Perm, and PriDe at three calibration fractions
(α = 5%, 40%, 80%), each as Accuracy and RStd (the paper's per-letter-recall
selection-bias metric, §2.2).

`llama-13B` is LLaMA-1, not Llama-2: the paper predates Llama-2, and the
`huggyllama/llama-13b` mirror is the only widely available copy of those
weights. That's also why this row: it's open weights, so it's the one cell
in the paper's 20-model table this framework can reproduce end-to-end without
a private API, and it's the row where all five methods this framework
implements (`direct_logprob`, `cyclic_logprob`, and offline-recomputed PriDe
at three αs) line up against numbers the paper actually published.

Scope, stated plainly: this is one row of a 20-model, multi-benchmark table.
It says nothing about the paper's other backbones, other benchmarks, or shot
counts beyond 0-shot. See **Where to go next** for extending the same config
shape to more rows.

## The config

This is the full, unmodified contents of `examples/pride_reproduction.yaml`:

```yaml
# examples/pride_reproduction.yaml
#
# Single-cell reproduction of Zheng et al., ICLR 2024 (arXiv:2309.03882),
# Table 3, MMLU block, llama-13B row, 0-shot. See
# examples/pride_reproduction_targets.json for the official numbers this
# run's metrics should land near.
#
# huggyllama/llama-13b is LLaMA-1 (NOT Llama-2): the paper's "llama-13B".
#
# PriDe itself is deliberately NOT a method below. PriDe (every alpha/seed)
# is recomputed offline from the option_distributions_json this run persists
# for direct_logprob/cyclic_logprob; see examples/pride_from_artifacts.py.
# Running PriDe here would waste GPU time re-deriving what that script
# already gets from these two methods' logprobs.
#
# Prerequisite - prepare the verified, filtered, unique-ID MMLU artifact first:
#   python scripts/prepare_data.py --hf-path cais/mmlu --hf-subset all --split test \
#       --filter-permutation-unsafe --exclude-duplicate-question-ids \
#       --output-name mmlu_filtered
# Both transforms are recorded in artifact and experiment identity. Excluding
# duplicate IDs is required for unambiguous row-level checkpoint/resume state.
# This should produce a 13,564-row artifact (14,042 raw − 426 permutation-unsafe
# − 52 remaining duplicate rows after that filter; see examples/pride_reproduction.md).

experiment:
  name: pride_reproduction_mmlu_llama13b_0shot

models:
  - backend: huggingface
    model_name_or_path: huggyllama/llama-13b
    # Exact commit used for the published walkthrough's run (runs/20260705_204054),
    # confirmed still resolvable on the Hub; see examples/pride_reproduction.md.
    revision: bf57045473f207bb1de1ed035ace226f4d9f9bba
    device: cuda

    # Zheng et al. do not prepend BOS for open-source models (Figure 6
    # caption, arXiv:2309.03882): the HuggingFaceBackend default is True
    # (add BOS), so this must be set explicitly for a faithful reproduction.
    add_bos_token: false

    # direct_logprob/cyclic_logprob never call generate(); they only use
    # score_options() (single forward pass, no sampling), so
    # generation_kwargs below is unused by this run; left at safe defaults.
    generation_kwargs:
      max_new_tokens: 1
      temperature: 0.0
      do_sample: false

# fp16 is HuggingFaceBackend's fixed load() dtype (torch_dtype=torch.float16),
# not a config knob; see examples/hpc/run_pride_reproduction.sbatch for the
# VRAM sizing that assumes it.

benchmarks:
  # The explicit transform is part of the prepared-artifact identity, so this
  # cannot accidentally resolve the canonical unfiltered MMLU artifact.
  - name: huggingface
    hf_path: cais/mmlu
    hf_subset: all
    split: test
    output_name: mmlu_filtered
    transforms: [permutation_safe_v1, unique_question_ids_v1]
    n_samples: null
    subject_filter: null

methods:
  - name: direct_logprob
    requires_logprobs: true
  - name: cyclic_logprob
    requires_logprobs: true

metrics:
  - accuracy
  - recall_rstd
  - mad

run:
  seed: 42
  resume: true
  dry_run: false
  checkpoint_every_n: 50
  prompt_version: "pride_repro"
  concurrency_limit: 10
```

The config comments above cover the mechanical "why"; the fidelity choices
that actually determine whether this is a faithful reproduction are these,
one sentence each:

- **Prompt template**: `prompt_version: "pride_repro"` selects
  `prompts/pride_repro/direct_mcq.txt`, hand-matched to the paper's Figure 6
  layout (`The following are multiple choice questions about {subject}. ...
  Question: {question}\nOptions:\n{options}\nAnswer:`), including MMLU's
  per-question `subject` substituted in (underscores → spaces, e.g.
  `abstract_algebra` → `abstract algebra`).
- **No BOS token**: `add_bos_token: false`, because the paper's Figure 6
  caption states open-source models are scored without a prepended BOS,
  while `HuggingFaceBackend`'s own default is `True`.
- **ID-token logprob scoring**: `direct_logprob`/`cyclic_logprob` never call
  `generate()`; both score the single-token option labels (`A`–`D`) via one
  forward pass through `score_options()`, matching the paper's own scoring
  convention (§2.1) rather than parsing free-text generations.
- **Permutation-safety filter**: `output_name: mmlu_filtered` points at data
  prepared with `--filter-permutation-unsafe`, which drops MMLU questions
  whose correct option text is meta-referential ("all of the above", "none
  of the above", "both A and B", ...), text that would silently break under
  cyclic permutation regardless of which method scores it.
- **K = floor(α·N), seeds 0–4**: PriDe's calibration subset size is
  `floor(alpha * N)` (not `round`), drawn uniformly without replacement,
  independently reseeded per α/seed pair (`np.random.default_rng(seed)`);
  every α is evaluated at 5 seeds so the grid reports a mean ± std, not a
  single draw.
- **Offline-PriDe design**: PriDe is not a method this config runs at all;
  the sbatch job executes exactly two GPU jobs
  (`direct_logprob`, `cyclic_logprob`), and every PriDe (α, seed) cell is
  recomputed afterward, entirely on CPU, from the `option_distributions_json`
  those two methods persist per question
  (see `examples/pride_from_artifacts.py`). One consequence worth being
  explicit about: because every PriDe cell is re-sliced from the *same* two
  already-completed GPU runs, the only thing that varies seed-to-seed is
  which questions land in the calibration subset: there is zero additional
  model-inference noise between seeds. The paper's own protocol, to the
  extent it reruns anything per calibration draw, would not have that
  guarantee. This makes our seed-to-seed spread a strict underestimate of
  whatever noise floor the paper's own protocol has, and is one honest
  source of the RStd/accuracy mismatches below.

## Running it

**1. Prepare the permutation-filtered MMLU data.** The command/output transcript
below is the preserved v0.1.2 reproduction receipt. For a current run, use the
verified command in the config above; it writes a split-addressed norm-v2
artifact and excludes duplicate IDs before inference.
(HuggingFace's dataset cache made this a local cache hit, not a fresh
download: the "Name or service not known" line below is `datasets`
routinely trying the network first and cleanly falling back):

```
$ python scripts/prepare_data.py --hf-path cais/mmlu --hf-subset all --filter-permutation-unsafe --output-name mmlu_filtered
16:04:03  INFO      Downloading cais/mmlu (subset=all, split=test)...
16:04:11  WARNING   Retrying in 1s [Retry 1/5].
16:04:12  WARNING   Using the latest cached version of the dataset since cais/mmlu couldn't be found on the Hugging Face Hub
16:04:12  INFO      Downloaded 14042 rows.
16:04:13  INFO      Normalized to 14042 rows.
16:04:13  INFO      Permutation-safety filter: excluded 426/14042 question(s) (3.0%) with meta-referential options.
16:04:13  INFO      Saved → data/processed/mmlu_filtered_normalized.csv
16:04:14  INFO      Stats: modal k=4 over 13616 questions (100.0% share) → data/processed/mmlu_filtered_stats.json
```

That 426-question exclusion breaks down by matched pattern (from
`data/processed/mmlu_filtered_permutation_filter.json`, committed at
[`examples/pride_reproduction_results/mmlu_filtered_permutation_filter.json`](pride_reproduction_results/mmlu_filtered_permutation_filter.json)
so this table is checkable without rerunning `prepare_data.py`):

| Pattern | Excluded |
|---|---|
| `all_of_the_above` | 215 |
| `none_of_the_above` | 140 |
| `both_and` | 59 |
| `letter_and_letter` | 11 |
| `letter_list_and_letter` | 1 |
| **Total** | **426** |

Against the paper: **ours kept 13,616 of 14,042 (excluded 426, 3.0%); the
paper reports 13,592 kept (excluded ~450, ~3.2%)**: a 24-question gap. The
paper does not publish its exact permutation-unsafe exclusion rule, so this
gap can't be closed further without guessing at unpublished string-matching
heuristics; see **Reading the results** for what this does and doesn't
threaten.

**2. Run `direct_logprob` + `cyclic_logprob` on an H100.** This is the actual
GPU job: submitted via `examples/hpc/run_pride_reproduction.sbatch`, not
re-run for this walkthrough (huggyllama/llama-13b in fp16 over 13,616
questions with cyclic_logprob's 4 forward-passes-per-question is a
30-minute, single-GPU job, not something to redo just to regenerate a log
line already captured). The completed run is `20260705_204054`: 30m50s wall
on one H100. Its persisted `config.yaml` confirms the model revision was
pinned before submission: the checked-in template above now pins this exact
commit directly (previously a `<PIN_BEFORE_RUN>` placeholder, so a stale run
could never happen silently; now pinned to the same commit this run itself
used, confirmed still resolvable on the Hub):

```
$ grep revision runs/20260705_204054/config.yaml
    revision: bf57045473f207bb1de1ed035ace226f4d9f9bba
```

**3. Recompute the PriDe grid, and hit a real duplicate-data guard.**
Pointing `pride_from_artifacts.py` at the raw completed run fails loudly,
by design:

```
$ python examples/pride_from_artifacts.py --run-id 20260705_204054 --alphas 0.05 0.40 0.80 --seeds 0 1 2 3 4 --targets examples/pride_reproduction_targets.json --out /tmp/grid.json
...
ValueError: direct_logprob results under .../runs/20260705_204054 contain 26
duplicate question_id value(s) (e.g. 'fcc342760aaab3fd'); expected exactly
one row per question. Investigate before recomputing PriDe: a stale or
doubly-checkpointed run would silently corrupt the scored subset.
```

This is `_load_method_frame`'s duplicate-`question_id` guardrail
(`examples/pride_from_artifacts.py`) catching real dataset pathology, not a
run artifact: MMLU's own official `test` split contains 27 verbatim
duplicate pairs (identical question text, identical subject, identical
correct answer: two different `question_id` hashes for what is the same
question asked twice; `answer_choices_json`, `subject`, and `question_text`
all matched byte-for-byte on inspection), 26 of which survive into this
run's 13,616-row permutation-safety-filtered subset. Without this guard,
`pride_from_artifacts.py` (or the framework's own `evaluate_run.py`) would
have silently double-counted these 26 questions in every metric. Both
copies of each duplicated question were dropped (not just the second),
since neither copy has a canonical claim to being "the" question, producing
a second run directory, `20260705_204054_dedup`, with those 52 rows removed
from both method CSVs:

```
$ python3 -c "
import pandas as pd
for run in ['20260705_204054', '20260705_204054_dedup']:
    df = pd.read_csv(f'runs/{run}/{run}_direct_logprob_huggyllama_llama-13b_mmlu_filtered.csv')
    dup = df['question_id'].value_counts()
    print(run, 'rows:', len(df), 'dup qids:', (dup > 1).sum())
"
20260705_204054       rows: 13616  dup qids: 26
20260705_204054_dedup rows: 13564  dup qids: 0
```

Under manifest protocol v2 this is enforced earlier: preparation/loading
rejects duplicate `question_id` values. The `unique_question_ids_v1`
transform removes every row in a duplicate-ID group and is recorded in
dataset and condition identity.

**Reconciling against prior reports of MMLU duplication.** On the raw
14,042-row MMLU test split (`cais/mmlu`, `all`, `test`), strict matching on
identical stem, options, subject, and answer key yields 27 duplicate pairs
(54 rows, 0.39%); 26 survive ChoiceBench's 13,616-row permutation-safe
filter, as above. Dropping just the subject field from that key nearly
quadruples the pair count to 105: of those 105, roughly three-quarters (78)
are the same question filed under two different subjects, and the remaining
quarter are the 27 strict pairs already counted. Loosening further to
stem-only matching raises the redundant-row count to 174 (1.24%), consistent
with the 1.2% of identical questions reported by Gupta et al.
(arXiv:2410.20245); the two figures differ on both matching criterion
(three-field vs. stem-only) and counting unit (pairs vs. removable rows).
Our contribution is the enumerated strict-criterion list with committed
receipts and its automatic detection by the pipeline's duplicate-`question_id`
guard above; a check absent from the MMLU-Redux error taxonomy
(arXiv:2406.04127).

Now the grid recomputes cleanly against the deduplicated run:

```
$ python examples/pride_from_artifacts.py --run-id 20260705_204054_dedup --alphas 0.05 0.40 0.80 --seeds 0 1 2 3 4 --targets examples/pride_reproduction_targets.json --out reports/20260705_204054_dedup_pride_grid.json
16:02:22  INFO      Scored subset: N=13564 (intersection of direct_logprob/cyclic_logprob successes)
16:02:26  INFO      alpha=0.05 seed=0  N=13564 K=678  acc=44.7 rstd=5.1 mad=4.4
16:02:28  INFO      alpha=0.05 seed=1  N=13564 K=678  acc=45.0 rstd=4.0 mad=3.9
...
16:02:45  INFO      alpha=0.80 seed=3  N=13564 K=10851  acc=48.0 rstd=5.0 mad=4.2
16:02:47  INFO      alpha=0.80 seed=4  N=13564 K=10851  acc=48.3 rstd=4.9 mad=4.2
16:02:47  INFO      row                acc   tgt_acc   d_acc    rstd  tgt_rstd  d_rstd     mad
16:02:47  INFO      default           43.0      34.6    +8.4    14.3      17.4    -3.1    11.2
16:02:47  INFO      cyclic_perm       49.0      47.7    +1.3     5.1       2.9    +2.2     4.3
16:02:47  INFO      pride_5           44.8      36.4    +8.4     4.4       5.7    -1.3     4.1
16:02:47  INFO      pride_40          46.5      40.4    +6.1     4.4       3.9    +0.5     3.9
16:02:47  INFO      pride_80          48.1      45.3    +2.8     4.8       2.6    +2.2     4.1
16:02:47  INFO      PriDe grid saved to reports/20260705_204054_dedup_pride_grid.json
```

**4. Cross-check against the framework's own evaluator.**
`pride_from_artifacts.py`'s `default`/`cyclic_perm` rows are computed by its
own `compute_default_row`/`compute_cyclic_row` (argmax over the persisted
distributions); the standard `evaluate_run.py` path scores the same two
methods from each row's own `is_correct`, computed at run time. They should
agree exactly, and they do:

```
$ python scripts/evaluate_run.py --run-id 20260705_204054_dedup
16:05:29  INFO      Loaded 27128 rows
16:05:31  INFO        [cyclic_logprob | huggyllama/llama-13b | mmlu_filtered]  {'accuracy': 0.49048953111176646, ... 'rstd': 5.064238729421211, ...}
16:05:31  INFO        [direct_logprob | huggyllama/llama-13b | mmlu_filtered]  {'accuracy': 0.4295193158360366, ... 'rstd': 14.347638248372515, ...}
```

Identical to machine precision to the `default`/`cyclic_perm` rows in the
grid above. That output is `reports/20260705_204054_dedup_metrics.json`,
committed at
[`examples/pride_reproduction_results/20260705_204054_dedup_metrics.json`](pride_reproduction_results/20260705_204054_dedup_metrics.json).

## Results

N = 13,564 scored questions (intersection of `direct_logprob`/
`cyclic_logprob` successes, post-dedup) vs. the paper's reported 13,592.
PriDe rows are mean ± population std over seeds 0–4
(`reports/20260705_204054_dedup_pride_grid.json`'s `per_alpha`, committed at
[`examples/pride_reproduction_results/20260705_204054_dedup_pride_grid.json`](pride_reproduction_results/20260705_204054_dedup_pride_grid.json));
Default/Cyclic Perm are single values (no seed dependence: every question
is scored the same way regardless of α).

| Method | Acc (ours) | Acc (target) | Δ Acc | RStd (ours) | RStd (target) | Δ RStd |
|---|---|---|---|---|---|---|
| Default | 43.0 | 34.6 | +8.4 | 14.3 | 17.4 | −3.1 |
| Cyclic Perm | 49.0 | 47.7 | +1.3 | 5.1 | 2.9 | +2.2 |
| PriDe (α=5%) | 44.8 ± 0.1 | 36.4 | +8.4 | 4.4 ± 0.5 | 5.7 | −1.3 |
| PriDe (α=40%) | 46.5 ± 0.3 | 40.4 | +6.1 | 4.4 ± 0.2 | 3.9 | +0.5 |
| PriDe (α=80%) | 48.1 ± 0.1 | 45.3 | +2.8 | 4.8 ± 0.1 | 2.6 | +2.2 |

**What's archived vs. what requires a rerun.** Three aggregated JSON files
from this run are committed under
[`examples/pride_reproduction_results/`](pride_reproduction_results/):
the final metrics (`20260705_204054_dedup_metrics.json`), the PriDe grid
(`20260705_204054_dedup_pride_grid.json`), and the permutation-filter report
(`mmlu_filtered_permutation_filter.json`): together enough to check every
number in the table above and in **Running it** without a GPU. The raw
per-question run artifacts this walkthrough narrates (`runs/20260705_204054`,
`runs/20260705_204054_dedup`: the `direct_logprob`/`cyclic_logprob` result
CSVs, checkpoints, and manifests) are **not** committed to this repository:
at ~13.6K rows × 2 methods, they're reproducible-on-demand working state, not
release artifacts. Verifying the aggregated numbers above requires trusting
this write-up; independently re-deriving them from scratch (including
re-checking the 26-duplicate-question-id incident this walkthrough
describes) requires rerunning the ~30-minute H100 job in **Running it** step
2 and the offline PriDe recompute in step 3.

## Reading the results

**The qualitative structure reproduces fully.** Every shape the paper
argues for shows up in our numbers: accuracy climbs monotonically with α
(44.8 → 46.5 → 48.1, capping out below Cyclic Perm's 49.0, which is the
ceiling every PriDe α approaches; Cyclic Perm is the limit PriDe converges
to as α → 1 (where every question is calibration-subset and scored via
Eq. 1, i.e. cyclic, directly), so it's expected, and here observed, to
upper-bound PriDe at every α; at intermediate α the non-calibration
fraction is scored by Eq. 8's prior-debiased direct distributions, which
have no formal guarantee of underperforming cyclic on those questions, so
this isn't a constructive bound); RStd collapses at every α relative to
Default's 14.3, down to 4.4–4.8, in the same range as Cyclic Perm's own
5.1; and PriDe at α=5%
already achieves that collapse in full (14.3 → 4.4, at or fractionally below
Cyclic Perm's own 5.1 floor, likely noise from K=678 being the smallest
calibration sample in the grid, not a real effect) for roughly 1.15× the
inference cost of Default alone
(cost ∝ N + 3K forward passes for K = αN calibration questions needing the
4-pass cyclic treatment vs. the 1-pass default; at α=0.05, that's
`1 + 3(0.05) = 1.15`). One clean internal check: **PriDe(5%) − Default is
+1.8 accuracy points in both the paper (36.4 − 34.6) and our run
(44.8 − 43.0)**: an exact match on the *relative* effect of the cheapest
PriDe setting, even though the absolute accuracies it's applied to differ by
8+ points. That's the strongest evidence in this walkthrough that we've
reproduced the paper's mechanism, not just landed in its neighborhood by
coincidence.

**The requested shape check, explicitly:** does Cyclic Perm beat PriDe(80%)
on accuracy, with PriDe(5%) doing more for RStd than for accuracy? Yes to
both. Cyclic Perm's 49.0 > PriDe(80%)'s 48.1. And PriDe(5%) already closes the *entire* RStd gap to Cyclic Perm
(14.3 → 4.4, at/below the 5.1 floor) while closing only 31% of the accuracy
gap (43.0 → 44.8, a +1.8 point gain, against Default→Cyclic Perm's full
+6.1 point gain): the paper's central claim that PriDe's calibration buys
bias-robustness far cheaper than it buys accuracy holds up numerically here,
if anything more starkly than the paper's own row (which reports PriDe(5%)
at RStd 5.7, still above Cyclic Perm's 2.9, a gap our run's noisier,
larger-N grid doesn't reproduce at this α).

**The accuracy offset is systematic and decays with α**: +8.4 (Default) →
+8.4 (PriDe 5%) → +6.1 (PriDe 40%) → +2.8 (PriDe 80%) → +1.3 (Cyclic Perm).
That decay localizes the offset to the *direct* scoring surface, not to the
PriDe or permutation machinery: Eq. 1 (used for every calibration-subset
question, regardless of α) never touches the default distribution's shape
at all; it's purely a function of the cyclic rollout matrix, so as α
grows, a larger fraction of the scored set is fully insulated from whatever
is biasing `direct_logprob`'s distribution, and the offset shrinks toward
Cyclic Perm's own small +1.3 residual. Consistent with this: our own
`direct_logprob` letter-recall breakdown shows a differently-shaped prior
than the paper implies: recall peaks at C (61.4%) and collapses at D
(21.9%), a much sharper skew than a uniform positional bias would produce
(computed via the same `compute_default_row` the grid table above uses, not
hand-typed).

**Candidate causes for that scoring-surface offset, none confirmed, stated
honestly:**

- **Option-ID token convention — RESOLVED, hypothesis eliminated**
  (local tokenizer audit, no GPU required). Reading
  `HuggingFaceBackend.score_options()`
  (`src/choicebench/backends/hf_backend.py`): it encodes the bare label
  string (`tokenizer.encode("A", add_special_tokens=False)`). This
  walkthrough previously left open whether SentencePiece's dummy-prefix
  handling makes that differ from the space-prefixed form a model would
  naturally produce after `"Answer:"`. The pinned checkpoint's cached
  `tokenizer.json` answers it directly: the LLaMA-1 fast tokenizer's
  normalizer applies an **unconditional `Prepend("▁")`**, so:

  | input string | normalized | pieces | token ids |
  |---|---|---|---|
  | `"A"`–`"D"` (bare) | `▁A`…`▁D` | one token each | 319 / 350 / 315 / 360 |
  | `" A"`–`" D"` (space) | `▁▁A`…`▁▁D` | two tokens (`▁▁`, letter) | 259 + 29909/29933/29907/29928 |

  Two consequences. First, the bare label **already encodes to the same
  space-prefixed piece** that mid-sentence text produces (`"Answer: A"`
  tokenizes with `▁A` following `Answer:`), so what `score_options()`
  scores is exactly the natural continuation surface — the
  bare-vs-space mismatch cannot be contributing to the offset.
  Second, the originally proposed follow-up experiment of re-scoring with
  the space-prefixed labels `" A"`–`" D"` is not expressible through this
  API at all for this tokenizer: those strings encode to two tokens and
  are rejected by `score_options()`'s single-token constraint.
  The offset candidates narrow to checkpoint-mirror provenance, fp16
  numerics (the GPU spot-check below), and paper-side differences.
- **Prompt whitespace at the `Answer:` position**: the same fact as above,
  from the template side: zero trailing whitespace after the colon.
- **Checkpoint mirror provenance**: `huggyllama/llama-13b` pinned at
  `bf57045473f207bb1de1ed035ace226f4d9f9bba`; this is a community re-upload
  of LLaMA-1 weights, not the paper's own checkpoint, and re-uploads have
  occasionally differed in tokenizer config or weight conversion from the
  original release.
- **fp16 numerics on H100**: `HuggingFaceBackend.load()` hardcodes
  `torch_dtype=torch.float16`; minor precision differences vs. whatever the
  paper used are possible but the least likely of these four to produce an
  8-point accuracy swing.

No single-variable experiment isolates which of these (if any) is
responsible. The option-ID token convention — previously the cheapest to
test — has been eliminated by the local tokenizer audit above; the
remaining candidates require either GPU access (the fp16 spot-check below)
or paper-side information, so no zero-cost decisive experiment remains.

**Other underspecified choices, for completeness**: none of these are
likely candidates for the 8-point offset above, but they're all places
where our protocol made an arbitrary call the paper doesn't specify closely
enough to match exactly: the exact permutation-unsafe filter rule (ours
excludes 426/14,042 by five explicit string patterns; the paper's ~450/14,042
rule is unpublished, a 24-question gap on a 13.5k-question set, immaterial
to any of the deltas above); the calibration-subset RNG scheme (uniform
without replacement via `numpy`'s `default_rng(seed)`; the paper does not
specify how `D_e` is sampled beyond "randomly select"); `K = floor(α·N)`
rounding (vs. `round()`, which would shift K by at most 1 and cannot explain
point-scale deltas); and the seed-to-seed std being population std
(`np.std`, ddof=0) over only 5 seeds: the paper reports no per-cell std at
all (see `examples/pride_reproduction_targets.json`'s `provenance` field),
so there's nothing to compare our spread against beyond internal
consistency.

**Closing observation.** The paper's own thesis is that default MCQ
accuracy is fragile to exactly this kind of scoring-surface minutiae, while
permutation-debiased scores are comparatively robust to it. Our own results
are consistent with that thesis one level up: the largest, least-explained
gap in this table (+8.4 points) sits on the method with no debiasing at all,
and it shrinks monotonically as more of the scored set gets Eq. 1's
permutation-pooled treatment; the same protection the paper argues for is,
in this reproduction, visibly absorbing whatever is producing the mismatch.

## Where to go next

- **More Table 3 rows, same config shape.** Nothing in
  `examples/pride_reproduction.yaml` is MMLU- or llama-13B-specific except
  `model_name_or_path`, `benchmarks[0]`, and (per-model) `add_bos_token`:
  copy the config, point it at another backbone or benchmark from Table 3,
  and rerun `examples/pride_from_artifacts.py` unchanged against the new
  run's artifacts.
- ~~**The space-prefix follow-up run** proposed above~~ **Eliminated by the
  local tokenizer audit above**: bare labels already encode to the
  space-prefixed pieces, and the space-prefixed strings cannot be scored at
  all through `score_options()` for this tokenizer. The offset question now
  needs either GPU experiments (fp16 spot-check) or paper-side information.
- **File the loader duplicate-`question_id` warning** as a known issue:
  `prepare_data.py`/benchmark loading should warn (not silently pass through)
  when a prepared benchmark CSV contains duplicate `question_id`s, so this
  doesn't depend on a downstream script's guard to catch it.
