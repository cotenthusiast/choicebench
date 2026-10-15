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

| Branch | Purpose | Status |
|---|---|---|
| `analysis/final-paper-authoritative-run` | Canonical import/overlay/metrics pipeline backing the paper's main results tables (accuracy, factorial effects, paired tests). | Load-bearing, kept. Its own `flip_rate_metrics.csv` output is *not* load-bearing, though — see next row. |
| `exp/flip-rate-traces` | Computes an order-sensitivity/flip-rate metric across the API and local model matrix. | Kept for its trace-generation pipeline, but **its output table (`flip_rate_table.csv`) does not match the paper's published Table 8** — verified by direct comparison against the arXiv PDF (values differ by roughly 3x on the 4 cells the two overlap, and it only covers 2 of 6 models). Likewise, `analysis/final-paper-authoritative-run`'s `flip_rate_metrics.csv` doesn't match either (covers 4/12 cells, different numbers again). The file that *does* match Table 8, verified all 36 cells, is `flip_rate_v2.csv` from an unrelated one-off audit — see `data/final_paper/README.md` for the full trail. |
| `worktree-visible-llm-matcher-cell` | Produces one of the paper's diagnostic conditions. | Load-bearing, kept |
| `feat/external-results-importer` | Standalone result-importer tool, evaluated separately from the paper. | Intentionally kept unmerged from trunk |
| `analysis/final-paper-eval` | Earlier iteration of the same import/overlay pipeline now developed on `analysis/final-paper-authoritative-run`. Confirmed to have zero commits not already present on `authoritative-run` (exact merge-base/ancestor): not a distinct pipeline. | Superseded; archived at tag `archive/final-paper-eval-superseded` |

None of the branches above are merged into trunk (`choicebench`); trunk
stays framework-first, with no paper-specific detail.
