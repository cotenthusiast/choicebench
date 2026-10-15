# Paper provenance

Several non-trunk branches in this repository produce results reported in
["Accuracy and Order Sensitivity Diverge Under Label-Free Strategies"](https://arxiv.org/abs/2608.11947)
(Hanna & Feng, arXiv:2608.11947). This page maps each such branch to what it
produces and its status, so anyone landing on one of them (or on
`main`/`choicebench`) has the context without needing to reverse-engineer it
from commit history.

**For the actual final numbers, use [`data/final_paper/`](data/final_paper/README.md)**
(gitignored; regenerate/repopulate it locally per that README) rather than
hunting across branches or worktrees — it consolidates the specific files
that were verified cell-by-cell against the published PDF, and documents
which other copies floating around the filesystem are dead ends.

## Current branch

| Branch | Purpose |
|---|---|
| `paper/arr-oct-2026` | Code, configs, prompts and scripts for the **revised** paper (*A, B, C, or Stable? Accuracy and Order Sensitivity Diverge in LLM Multiple-Choice Evaluation*): the full benchmark x method x model grid, reasoning variants, the 2x2 visibility-by-matcher diagnostic, flip-rate and RStd traces, the stochasticity protocol, and the `tests_conformance/` suite that encodes the frozen protocol. Renamed from `paper/eacl-2026-revision`. Generic framework changes land on `choicebench` first and are merged into this branch. |

## Archived pipelines for the original (v1) paper

These branches are kept, and each tip is also pinned by an `archive/*` tag so it survives any later branch cleanup.

| Branch | Tag | Purpose | Status |
|---|---|---|---|
| `analysis/final-paper-authoritative-run` | `archive/final-paper-authoritative-run` | Canonical import/overlay/metrics pipeline backing the original paper's main results tables (accuracy, factorial effects, paired tests). | Load-bearing for the v1 tables. Its own `flip_rate_metrics.csv` output is *not* load-bearing, though — see next row. |
| `exp/flip-rate-traces` | `archive/exp-flip-rate-traces` | Computes an order-sensitivity/flip-rate metric across the API and local model matrix. | Kept for its trace-generation pipeline, but **its output table (`flip_rate_table.csv`) does not match the paper's published Table 8** — verified by direct comparison against the arXiv PDF (values differ by roughly 3x on the 4 cells the two overlap, and it only covers 2 of 6 models). Likewise, `analysis/final-paper-authoritative-run`'s `flip_rate_metrics.csv` doesn't match either (covers 4/12 cells, different numbers again). The file that *does* match Table 8, verified all 36 cells, is `flip_rate_v2.csv` from an unrelated one-off audit — see `data/final_paper/README.md` for the full trail. |
| `worktree-visible-llm-matcher-cell` | `archive/visible-llm-matcher-cell` | Produced the "visible options + LLM matcher" diagnostic condition. | Superseded: reconciled into `methods/library/visible_llm_matcher.py` on the paper branch. |

## Removed branches

| Former branch | What happened |
|---|---|
| `analysis/final-paper-eval` | Earlier iteration of the import/overlay pipeline now developed on `analysis/final-paper-authoritative-run`; superseded. Its tip is preserved at `archive/final-paper-eval-tip` (and an earlier state at `archive/final-paper-eval-superseded`). |
| `feat/external-results-importer` | A standalone result-importer tool, evaluated separately from the paper and never merged. Removed from the repository (last commit `55c7fb7`). |
| `conformance-suite`, `final-audit-fixes`, `wip/batched-hf-fp8` | Fully contained in other branches; nothing lost. The commit the conformance spec was written against is tagged `archive/conformance-spec-baseline`. |

None of the branches above are merged into trunk (`choicebench`) except as noted;
trunk stays framework-first, with no paper-specific detail.
