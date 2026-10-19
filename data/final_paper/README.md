# Final paper data (consolidated)

This directory exists because the actual data behind arXiv:2608.11947
("Accuracy and Order Sensitivity Diverge Under Label-Free Strategies") was
scattered across `~/Research`, `~/Backups`, and several `choicebench-*` git
worktrees outside this repo, under names like `_scratch_*` and
`_archive/*` that didn't reflect what was actually authoritative. This
directory is a **copy** of the two pieces that were verified, cell by cell,
against the published paper (fetched from arXiv directly). The originals it
was copied from are untouched and still exist at the paths noted below.

Contents are gitignored (see repo `.gitignore`) — only this README is
tracked. Re-run the verification below yourself before trusting either file
if you're not sure it's still current.

## `accuracy_bundle/` — accuracy, factorial effects, paired tests, etc.

Verbatim copy of
`final_paper_authoritative_run_20260721T140619Z/final_outputs_corrected/`
(originals also mirrored at `~/Backups/final_paper_authoritative_run_20260721T140619Z/`).
Produced by the `analysis/final-paper-authoritative-run` branch (see
the README's "Branches and provenance" section). `SHA256SUMS` in this directory still verifies
against every file (`sha256sum -c SHA256SUMS`).

Use `cell_metrics.csv` (`strict_accuracy` column) for per-model/benchmark/
method accuracy. Verified spot-checks against the arXiv PDF: baseline
accuracy (Gemini MMLU 84.9%, GPT-4.1-mini MMLU 81.8%, Llama-local ARC
58.0%) and `two_stage_v1` accuracy (Gemini MMLU 68.3%, Gemini ARC 86.6%,
GPT-4.1-mini MMLU 80.1%, Llama-local ARC 58.3%) both match the paper's
`two-stage` numbers exactly — so `two_stage_v1` is the paper's "two-stage."
`cyclic_generation_majority` vs `baseline`: improves accuracy in 10/12
model-benchmark pairs, max +14.9pp (Llama-local, arc_challenge), matching
the paper's headline claim exactly.

**⚠️ `flip_rate_metrics.csv` inside this bundle does NOT match the
published paper.** It's kept only because removing it would break the
bundle's `SHA256SUMS` integrity chain. It covers 4 of 12 cells and its
numbers are ~3x the published Table 8 values for the same cells. Do not use
it — see `../flip_rate/` instead. (A second, independent dead end also
exists outside this repo, on the `exp/flip-rate-traces` branch /
`~/Projects/choicebench-flip-rate-traces` worktree — same story: internally
reproducible from its own raw traces, but does not match the published
paper. See the README's "Branches and provenance" section.)

## `flip_rate/` — order-sensitivity / flip rate

`flip_rate_v2.csv` + the script that produced it (`compute_metrics.py`),
copied from `~/Research/choicebench/_archive/_scratch_full_metrics_audit_20260726_v2/`
— despite the `_scratch` name, **this is the file that matches the
published Table 8.** Verified: all 36 cells (6 models × 2 benchmarks ×
{baseline, two_prompt, twostage_semantic_match}) checked byte-for-byte
against the arXiv PDF's Table 8 and matched exactly, including the paper's
explicit note that "the Cyclic condition is not shown as a row: its flip
rate is identical to Baseline by construction" (so `baseline`'s flip rate
doubles as cyclic permutation's), and that semantic matching's flip rate
is exactly 0.0% in every one of the 12 cells.

`rate_A` is the flip-rate percentage (already ×100, e.g. `21.6` not `0.216`).

## If you need to re-verify

Fetch `https://arxiv.org/pdf/2608.11947`, extract text (`pdftotext -layout`),
and diff Table 8's 36 cells against `flip_rate/flip_rate_v2.csv`'s `rate_A`
column, and the baseline/`two_stage_v1` spot-checks above against
`accuracy_bundle/cell_metrics.csv`'s `strict_accuracy` column.
