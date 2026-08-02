# Paper provenance

Several non-trunk branches in this repository produce results reported in a
paper currently under peer review; a link will be added once it is public.
This page maps each such branch to what it produces and its status, so
anyone landing on one of them (or on `main`/`choicebench`) has the context
without needing to reverse-engineer it from commit history.

| Branch | Purpose | Status |
|---|---|---|
| `analysis/final-paper-authoritative-run` | Canonical import/overlay/metrics pipeline backing the paper's main results tables. | Load-bearing, kept |
| `exp/flip-rate-traces` | Computes one of the paper's order-sensitivity metrics, across the API and local model matrix. | Load-bearing, kept |
| `worktree-visible-llm-matcher-cell` | Produces one of the paper's diagnostic conditions. | Load-bearing, kept |
| `feat/external-results-importer` | Standalone result-importer tool, evaluated separately from the paper. | Intentionally kept unmerged from trunk |
| `analysis/final-paper-eval` | Earlier iteration of the same import/overlay pipeline now developed on `analysis/final-paper-authoritative-run`. Confirmed to have zero commits not already present on `authoritative-run` (exact merge-base/ancestor) — not a distinct pipeline. | Superseded; archived at tag `archive/final-paper-eval-superseded` |

None of the branches above are merged into trunk (`choicebench`); trunk
stays framework-first, with no paper-specific detail.
