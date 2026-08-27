# Local audit checks: dataset-label drift, tie-breaking, and the 34.6% baseline

Three small checks from the PriDe reproduction-deviation investigation that
were computed and reported in-session but, unlike the three variant
experiments (`pride_dtype_variant_prediction.md`,
`pride_scoring_variant_prediction.md`, `pride_whitespace_variant_prediction.md`),
had no committed artifact until this note. None of these needed a
pre-registered prediction — they're direct measurements/audits against
already-completed run data and external sources, not controlled variants
with a predicted outcome to test. Written up here for the same reason the
others were: so a specific number can be checked without re-deriving it from
a chat transcript.

All three use `runs/20260705_204054_dedup/20260705_204054_dedup_direct_logprob_huggyllama_llama-13b_mmlu_filtered.csv`
(not committed, per the `runs/*` gitignore convention — reproducible via
`examples/hpc/run_pride_reproduction.sbatch`) and/or
`examples/pride_reproduction_results/scoring_variant_results.csv` (also not
committed, same convention — reproducible via
`examples/pride_scoring_variant_run.py`, Kelvin2 job 9708113).

## 1. Dataset-label version drift (HF `cais/mmlu` vs the reference repo's frozen 2023 CSVs)

**Method**: downloaded all 57 subject test CSVs from
`chujiezheng/LLM-MCQ-Bias/code/data_mmlu/test/` (their static 2023 snapshot,
format `Question,A,B,C,D,Answer` with no header — confirmed by reading
`code/eval_clm_utils.py`'s `pd.read_csv(..., names=(...))` call), built a
`question_text -> gold_letter` lookup, and matched against all 13,564
questions in our run by exact question-stem text.

```
curl -s "https://raw.githubusercontent.com/chujiezheng/LLM-MCQ-Bias/main/code/data_mmlu/test/<subject>_test.csv"
# repeated for all 57 subjects listed via:
gh api "repos/chujiezheng/LLM-MCQ-Bias/git/trees/main?recursive=1"
```

**Result**: 13,514/13,564 (99.63%) match on both text and gold label; 1
question text not found in their files at all; **49/13,564 (0.36%) have
identical question text but a different gold answer letter**.

Re-grading our existing predictions (job 9708113's `pred_current`, model
predictions unchanged — only which letter counts as "correct" changes) using
their labels for those 49 questions:

| | accuracy | RStd |
|---|---|---|
| our labels (HF `cais/mmlu`) | 42.8930% | 14.3753 |
| their labels (2023 snapshot) | 42.8192% | 14.3586 |
| Δ | -0.0737pp | -0.0168 |

**Verdict: RULED OUT / negligible.** Real, genuine dataset-version drift
exists (49 questions is a hard fact, not noise), but its impact is an order
of magnitude too small to matter — smaller than every other tested
candidate.

## 2. Tie-breaking convention

**Method**: checked `job 9708113`'s current-pipeline distributions
(`dist_current_json` column) for exact float ties between the top-1 and
top-2 option probabilities.

**Result**: 91/13,564 (0.67%) rows have an exact tie. Both our pipeline
(`np.argmax` in `direct_logprob.py`/`cyclic_logprob.py`) and the reference
implementation (`eval_clm_utils.py::prepare_eval_fn_base`:
`sampled = option_ids[np.argmax(probs)]`) resolve ties identically — first
index wins, i.e. alphabetically-earliest letter among the tied options.

**Verdict: RULED OUT as a source of divergence.** Same convention on both
sides; cannot explain any gap between our numbers and theirs.

## 3. The 34.6%-vs-25%-floor question

**Method**: web search for independent zero-shot LLaMA-13B MMLU numbers
beyond the paper being reproduced.

**Result**: OccuQuest (arXiv:2310.16517), Table 4 / Appendix H: "Vanilla
LLaMA-13B" scores **39.3%** zero-shot MMLU accuracy — an independent
source, sitting between the paper's 34.6% (Zheng et al.) and our
42.89-43.03%, in the same plausible band. (LLaMA's own paper, arXiv:2302.13971,
Table 9, reports 46.9% for 13B — but that's 5-shot, a materially different
setting, not directly comparable.) Also surfaced in this search: LLaMA MMLU
numbers are well-documented in the community (EleutherAI
`lm-evaluation-harness` issues #443, #1213) as sensitive to exact
scoring/tokenization implementation — consistent with, not contradicting,
what this whole investigation keeps finding.

**Verdict: RESOLVED.** 34.6% is a plausible zero-shot LLaMA-13B MMLU
baseline under this scoring convention, not an anomaly requiring
explanation in its own right.
