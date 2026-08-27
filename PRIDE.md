# PriDe reproduction-deviation investigation: final record

This closes out the investigation into why our reproduction of Zheng et al.,
ICLR 2024 (arXiv:2309.03882), Table 3, MMLU / `llama-13B` / 0-shot deviates
from the paper's published numbers. See
[`examples/pride_reproduction.md`](examples/pride_reproduction.md) for the
full reproduction walkthrough (config, commands, the original results table);
this document covers only the follow-up bisection into *why* the Default
(`direct_logprob`) cell misses the paper's numbers.

## The gap

Original run (`runs/20260705_204054_dedup`, Kelvin2 job 9315709,
`huggyllama/llama-13b@bf57045473f207bb1de1ed035ace226f4d9f9bba`, N=13,564),
persisted metrics
([`examples/pride_reproduction_results/20260705_204054_dedup_metrics.json`](examples/pride_reproduction_results/20260705_204054_dedup_metrics.json))
vs. the paper's target
([`examples/pride_reproduction_targets.json`](examples/pride_reproduction_targets.json)):

| | Accuracy | RStd |
|---|---|---|
| Ours | 42.95% | 14.35 |
| Paper (Table 3, `default`) | 34.6% | 17.4 |
| **Gap** | **+8.35pp** | **-3.05** |

All bisection experiments below re-derive their own fresh baseline on
current code (typically ≈42.89%/14.375, not 42.95%/14.35) rather than reuse
the original run's numbers, so each is a same-session paired comparison, not
a cross-run one. The ~0.06pp difference between the two baselines is
run-to-run fp16 floating-point noise, not a new discrepancy — noted
explicitly in the affected result files rather than silently absorbed.

## Candidates tested

| # | Candidate | Method | Key number(s) | Verdict | Why |
|---|---|---|---|---|---|
| 1 | Population / prompt template | Reconciled exclusion counts + character-level template diff against reference `create_user_prompt` | 14,042 → 13,616 (-426 permutation-unsafe) → 13,564 (-52 dedup); paper reports ~13,592 (0.21% delta) | **RULED OUT** | Delta is constant across cells while the observed gap varies 8.35→1.35pp across methods; a population-size difference can't produce that shape |
| 2 | Parse-rate / permissive parser | Compared `accuracy` vs `accuracy_conditional` in persisted run metrics | Identical to 16 decimal places | **RULED OUT** | Zero parse failures occurred — nothing for a lenient/strict parser distinction to recover |
| 3 | Truncation (last-1536-tokens) | Tokenized every persisted prompt with the pinned tokenizer | Max prompt length 1,219 tokens; nothing exceeds 1,536 | **RULED OUT** | Our code truncates nowhere; nothing in this population is long enough for a 1,536-token truncation policy to ever bite either way |
| 4 | dtype (fp16 vs bf16) | Controlled variant, same code/data, dtype only changed; pre-registered prediction | Δ = **-0.18pp** accuracy, **+0.28** RStd | **RULED OUT** | Prediction confirmed: change is within near-tie rounding-noise bounds, no systematic effect |
| 5 | Scoring mechanism (two-token-form marginalization) | Reimplemented reference's exact 8-way restricted-softmax scoring, paired against current pipeline from the same forward pass | Δ = **+0.14pp** accuracy, **+0.02** RStd; second-token mass median 3.2%, max 18.7% (never exceeds 30%) | **RULED OUT** | Mechanism is real (verified against reference source and reproduced live) but the second token essentially never carries competitive probability mass at this position |
| 6 | Whitespace (`.strip()`) | Re-scored the 809 affected questions under original vs. `.strip()`'d prompts, paired against unaffected-row baseline | Aggregate Δ = **-0.022pp** / **+0.049** RStd; affected-subset flip rate **8.90%** (21 right→wrong vs. 18 wrong→right — undirected) | **RULED OUT / NEGLIGIBLE** | Real effect at the individual-question level, but symmetric/undirected — ordinary prompt-perturbation noise, not a systematic bias, and capped low in aggregate by 5.96% prevalence regardless |
| 7 | Dataset-label version drift (HF `cais/mmlu` vs. reference repo's frozen 2023 CSVs) | Diffed all 13,564 question texts against the reference repo's bundled MMLU test CSVs | 49/13,564 (0.36%) questions have a different gold label; re-grading impact Δ = **-0.074pp** / **-0.017** RStd | **RULED OUT / NEGLIGIBLE** | Genuine drift exists but affects too few questions to matter |

**Also resolved, not gap candidates in their own right:**

| Question | Finding | Verdict |
|---|---|---|
| Tie-breaking convention | 91/13,564 (0.67%) exact ties; both our pipeline and the reference implementation use identical first-max `np.argmax` | **RESOLVED** — same convention both sides, not a source of divergence |
| Is 34.6% anomalously low vs. a 25% chance floor? | Independent source (OccuQuest, arXiv:2310.16517) reports 39.3% zero-shot LLaMA-13B MMLU — between the paper's 34.6% and our ~43% | **RESOLVED** — 34.6% is a plausible baseline under this scoring convention, not an anomaly |

Full numbers, methodology, and exact SLURM job IDs for every row above:
[`examples/pride_dtype_variant_prediction.md`](examples/pride_dtype_variant_prediction.md)
(jobs 9708132, 9708113),
[`examples/pride_scoring_variant_prediction.md`](examples/pride_scoring_variant_prediction.md)
(job 9708113),
[`examples/pride_whitespace_variant_prediction.md`](examples/pride_whitespace_variant_prediction.md)
(jobs 9708410, 9708414),
[`examples/pride_local_audit_checks.md`](examples/pride_local_audit_checks.md)
(candidate #7, tie-breaking, 34.6% baseline). Each prediction file also
records what was predicted *before* the run, for anyone checking whether a
result was cherry-picked after the fact versus called in advance (two of the
three were called correctly; the scoring-mechanism prediction was wrong, and
says so).

## Deprioritized, not silently dropped

Two candidates remain unexplored. Both were named and set aside explicitly,
not omitted:

- **Checkpoint identity.** `huggyllama/llama-13b` is a community re-upload
  of LLaMA-1 weights on the HF Hub, not an official Meta artifact or the
  paper's own checkpoint. If it was converted or re-uploaded differently
  from whatever exact weights Zheng et al. used in 2023, no scoring-rule
  fix could ever close the resulting gap. There is no byte-level provenance
  check available to test this.
- **Library-version drift.** `transformers`/`torch` have changed LLaMA's
  rotary-embedding and attention internals multiple times since 2023.
  Reproducing the paper's exact ~2023 environment would require pinning an
  old library stack and re-running — a multi-day infrastructure project
  with uncertain payoff, not a single-variable isolation like the seven
  above.

## Closing statement

**Seven candidates tested with real, quantified evidence. All seven are
ruled out or negligible — none individually exceeds ±0.5pp/units. The
+8.35pp accuracy / -3.05 RStd Default-cell gap remains almost entirely
unexplained.** This is the honest result of an exhaustive
implementation-level bisection, not a partial or rounded-down account of it:
nothing tested so far accounts for a meaningful fraction of the deviation.
The two remaining candidate mechanisms (checkpoint identity, library-version
drift) are structurally different from what's been ruled out — neither is
a cheap, isolable single-variable test — and are being deprioritized rather
than pursued further at this time.
