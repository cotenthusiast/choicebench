# Changelog

Notable changes to ChoiceBench. Newest first.

## v0.1.3 - 2026-08-03

New trusted baseline release, superseding v0.2.0. v0.2.0 was heavily
AI-assisted and never fully validated; this release fixes the issues found
in a post-hoc audit and restores confidence in the v0.1.x line's release
process, without reverting any of v0.2.0's functional changes:

- Aggregate run/evaluation summaries now distinguish a condition whose rows
  are dominated by backend transport failures ("infra_failure") from a
  genuinely completed condition, instead of silently reporting it as
  "completed" with a misleadingly low accuracy.
- Added a regression test guarding sdist entry-point registration when
  built from the repo root.
- Fixed `test_built_wheel_and_sdist_run_outside_repository` to stop relying
  on `--system-site-packages` inheriting dependencies from ambient
  context; it now installs a fully self-contained venv, matching a truly
  clean environment.
- Refreshed the editable install so `importlib.metadata.version("choicebench")`
  correctly reports the installed version.

## v0.2.0 - 2026-07-18

- Added protocol-v2 immutable run manifests, deterministic experiment and
  condition identities, verified resume compatibility, and manifest-driven
  evaluation. Legacy nonempty run directories are rejected unless reset.
- Replaced generic normalized CSV reuse with split-addressed, source-spec
  addressed prepared artifacts carrying semantic SHA-256 content digests and
  verified metadata. Calibration now loads its actual requested split and
  rejects normalized stem/option overlap with evaluation data.
- Results and checkpoints are keyed by canonical condition IDs. Parameterized
  methods and same-display-name models with different configurations cannot
  overwrite or aggregate into one condition.
- Packaged prompt resources with `importlib.resources`, added installed console
  entry points, and made `CHOICEBENCH_HOME`/the current directory the explicit
  writable workspace rather than deriving paths from `site-packages`.
- Added strict numeric and boolean validation across experiment configuration,
  requests, and clients, plus clean-wheel and adversarial provenance tests.
- Final release hardening moved identity to protocol/manifest v2: local model
  trees and resolved Hub commits are bound to model IDs, external method/metric
  modules are fingerprinted, result/cache/PriDe artifacts are integrity-checked,
  and one process lock protects each run ID and reset operation.
- Source-clone scripts are now thin wrappers around the installed CLI modules.
  Prepared artifacts use normalization v2; duplicate question IDs are rejected,
  and the explicit `unique_question_ids_v1` preparation transform supports
  datasets such as MMLU that contain duplicate content-derived IDs.
- Run-local dataset and prompt snapshots are revalidated before evaluation.
  Evaluation verifies metric implementation identity, records parser/scorer
  identity for `--reparse`, accounts for gated and failed conditions, and
  writes a `<run_id>_<evaluation_id>_metrics.json` report whose evaluation ID
  binds the exact validated result contents (per-condition statuses and result
  digests): identical result sets share one ID, materially different results
  get distinct IDs, and an existing report with different contents is never
  silently overwritten.
- Runs with failed conditions remain evaluable: completed conditions get
  metrics, failed conditions are accounted (status plus redacted error, no
  invented metrics), and the report carries `run_status` (`complete`/`partial`)
  with per-status counts. Unfinished (pending) conditions still refuse
  evaluation.
- Rerunning an identical experiment under an existing run ID with `run.resume`
  disabled is refused up front with an actionable message; it no longer
  downgrades completed conditions to `failed`. Resume refusals and lock
  conflicts now exit with a clean error naming the differing identity
  sections instead of a traceback.
- Row-ownership validation fails closed: a result row with a missing/null
  identity value is rejected at write and at read instead of being silently
  dropped from metric samples.
- Credential handling is schema-aware: unambiguous credential-named keys
  (`api_key`, `authorization`, `client_secret`, …) are refused in scientific
  configuration instead of being redacted before hashing, so secret-shaped but
  scientific parameter names (`token`, `secret_strength`, …) keep their
  identity-affecting values. Embedded URL credentials are still sanitized.
- `do_sample` no longer enters API model identity (the API backend never
  consumes it); HuggingFace identities still bind it. `MAD`'s bootstrap uses a
  masked divide (identical values, no spurious warnings). The test suite is
  self-contained: it prepares its own toy data in a temporary
  `CHOICEBENCH_HOME`, so a clean clone passes `pytest` with no manual step.

**Migration from v0.1.2:** generic `data/processed/*_normalized.csv` files and
root-level run CSVs are unverified legacy artifacts. Re-run preparation for each
split (and declared transform), choose a new run ID, and use the manifest-driven
evaluator. Existing v0.1.2 result directories remain readable by the dedicated
historical analysis scripts but cannot be resumed as v0.2 experiments.

**Scoped guarantees:** ChoiceBench prevents concurrent writers to one run ID;
Slurm tasks must use distinct IDs. Remote APIs can still be nondeterministic and
cannot always expose immutable server-side model revisions. Hub acquisition is
pinned when the Hub provides a commit; exact selected rows are also archived in
the run directory.

## v0.1.2 — 2026-07-06

- Committed the PriDe reproduction's evidence files under
  `examples/pride_reproduction_results/` (the PriDe grid JSON, the
  cross-check metrics JSON, and the permutation-filter sidecar) so every
  number in `examples/pride_reproduction.md` is checkable without rerunning
  the pipeline or having the gitignored `reports/`/`data/processed/`
  originals on hand. Referenced from the three places in the walkthrough
  that cite those artifacts.
- Added `examples/pride_reproduction.md`, the full walkthrough of the completed
  MMLU / llama-13B / 0-shot single-cell reproduction of Zheng et al. (ICLR
  2024) Table 3: real commands and output for data prep, the GPU run, the
  PriDe grid recompute, and a framework cross-check; a results table against
  the paper's published numbers; and an honest accounting of where our
  numbers match, where they diverge, and why (scoring-surface candidates,
  underspecified protocol choices). Surfaced a real dataset issue along the
  way: MMLU's official `test` split contains 27 verbatim duplicate pairs (26
  post-filter), caught by `pride_from_artifacts.py`'s duplicate-`question_id`
  guard rather than silently double-counted; filed as a known issue that the
  benchmark loader should warn on this at load time (not fixed here).
- Reconciled the walkthrough's duplicate-question finding against prior
  reports: strict stem+choices+subject+answer matching on the raw 14,042-row
  split gives 27 pairs (0.39%), while loosening to stem-only matching gives
  174 redundant rows (1.24%), consistent with the 1.2% reported by Gupta et
  al. (arXiv:2410.20245) — the gap is both matching criterion and counting
  unit, not a discrepancy; also notes the strict-criterion guard as a check
  absent from the MMLU-Redux taxonomy (arXiv:2406.04127).
- Added `examples/pride_from_artifacts.py`, the offline PriDe recompute for
  the reproduction wired in `examples/pride_reproduction.yaml`: reads the
  completed `direct_logprob` + `cyclic_logprob` result CSVs for a run and
  reconstructs the full PriDe grid (every `--alphas` x `--seeds` cell) purely
  from their persisted `option_distributions_json` — zero additional GPU
  inference. Per (alpha, seed): calibration questions (K = floor(alpha * N),
  seeded, uniform without replacement) are answered via Eq. 1 and averaged
  into a global prior via Eq. 7; every other question is answered via Eq. 8
  against that prior. Reuses `pride_math`'s equation functions and
  `choicebench.metrics`' `accuracy`/`recall_rstd`/`mad` directly — no
  duplicated math. Questions present in only one CSV, or that failed in
  either, are dropped from the scored subset (logged), not imputed. Writes a
  JSON grid (per-cell metrics + N/K accounting, per-alpha mean/std over
  seeds, and a comparison block against
  `examples/pride_reproduction_targets.json`'s official numbers) and prints a
  compact aligned summary table.
- Added the PriDe (Zheng et al., ICLR 2024, arXiv:2309.03882) single-cell
  reproduction plumbing for MMLU / llama-13B (LLaMA-1) / 0-shot:
  `--filter-permutation-unsafe` on `scripts/prepare_data.py` (excludes
  meta-referential MMLU options like "A and B", "none of the above" —
  `choicebench.permutation_filter`), a new `prompts/pride_repro/` template
  version matching the paper's exact Figure 6 layout with per-question
  `subject` support in `build_direct_mcq_prompt()`, a new `add_bos_token`
  model config field (`HuggingFaceBackend` defaults to adding BOS; the paper
  does not for open-source models), `examples/pride_reproduction.yaml` +
  `examples/hpc/run_pride_reproduction.sbatch` (runs `direct_logprob` +
  `cyclic_logprob` only — PriDe itself is recomputed offline from their
  persisted distributions in a separate script), and
  `examples/pride_reproduction_targets.json` with the paper's official
  Table 3 numbers for this cell.
- Added `shuffled_baseline`, a demo evaluation method (single-call, seeded
  per-question option shuffle) used as the subject of a new extensibility
  walkthrough, `examples/method_comparison.md`.
- Added `recall_rstd`, the paper's selection-bias metric (Zheng et al., ICLR
  2024, §2.2) — standard deviation of per-letter recall, for reproducing
  Table 3-style RStd numbers.
- Added `direct_logprob`, a logprob-based method reproducing the paper's
  "Default" baseline (§2.1): argmax over `score_options()` logprobs instead
  of generate-and-parse. Requires a logprob-capable backend
  (HuggingFace/Dummy).

## v0.1.1 — 2026-07-04

Second release. Everything below, through the vLLM-removal entry, landed
since v0.1.0 (2026-07-01) and ships together here — a variable-option-count
rework, a statistical validity audit, and a distribution-readiness pass
prompted by a blind "cold clone" of the repo onto a fresh HPC/Slurm
environment following only the README.

### 2026-07-04 — Cold-clone distribution-readiness fixes

A blind cold-clone of the repo onto a fresh HPC/Slurm environment, following
only the README, surfaced four real bugs and two documentation gaps.

#### Fixed

- **PriDe calibration contamination.** `source: benchmark` preflight sampling
  reloaded the identical normalized CSV the eval sample was drawn from; the
  split-label check alone didn't guarantee disjoint rows. `load_preflight()`
  now excludes eval `question_id`s from the calibration pool before sampling.
- **Models reloaded from disk on every (benchmark, method) pair.**
  `build_backend()` — and `HuggingFaceBackend.load()`, which loads the
  tokenizer and weights — ran fresh per pair instead of once per run. A
  `backend_cache` keyed on model-config identity, threaded through
  `_async_main → run_models_concurrently → _run_model_isolated → _run_model`,
  now builds/loads each model exactly once per run and reuses it for every
  subsequent benchmark/method.
- **PriDe's modal-k gate exclusion was indistinguishable from a real crash.**
  `ModalKGateError` — raised when a benchmark's modal-k coverage is below
  `pride.modal_k_threshold`, an intentional design exclusion — was caught by
  the same broad exception handler as actual bugs, landing in the run's
  `failures` list and triggering `sys.exit(1)`. Combined with `set -e` in the
  SLURM template, this silently aborted the pipeline before
  `evaluate_run.py` ever ran, even on a fully successful experiment. Gated
  exclusions now route to a separate `gated` tally (with the gate-report
  JSON sidecar written on that path too) and no longer affect the exit code.
- **`examples/hpc/setup_hpc.sh` installed `.[dev]` instead of `.[hf]`,** so a
  new HPC user following the README got a fully working venv that still
  raised `ImportError: torch and transformers are required` on their first
  real model run.
- **`BASH_SOURCE`-based path resolution breaks under real `sbatch`
  execution.** SLURM copies a submitted script to a spool directory before
  running it, so `${BASH_SOURCE[0]}` inside the job resolves to the spool
  path, not the script's real location — confirmed with a diagnostic job.
  `examples/hpc/run_choicebench.sbatch`, `scripts/slurm/submit_job.sh`, and
  `scripts/slurm/submit_array.sh` now resolve `REPO_ROOT` via
  `$SLURM_SUBMIT_DIR` instead.

#### Documented

- README Troubleshooting: `module: command not found` inside a Slurm job
  even though `module load` works when run interactively — `sbatch`
  inherits the submitting shell's environment, so a job submitted from a
  one-shot/non-interactive shell that never sourced the cluster's profile
  scripts inherits the absence of `module` too.
- README Config Reference: a concrete `preflight:` block example
  (`source` / `split` / `n`) under the `pride` method — previously only
  described in prose, with the actual syntax discoverable only by reading
  `preflight.py` directly.
- README "Real Run Results" now shows the actual full-grid run (5
  benchmarks × 2 models × 3 methods, n=10, seed 42) instead of a TODO
  placeholder, including the real `n_evaluated`/`n_total` the modal-k gate
  reported for MMLU-Pro and TruthfulQA.

#### Verified

- Full test suite passing (607 tests) after every change above.
- A real end-to-end run on a SLURM cluster (5 benchmarks, 2 models, 3
  methods) completing in ~13 minutes post-fix, versus timing out entirely
  pre-fix.

### 2026-07-03 — Failure isolation, permutation tie-break, HPC docs

#### Fixed

- **One model's unhandled exception could crash a whole concurrent run.**
  `run_models_concurrently` used `asyncio.gather()` with the default
  `return_exceptions=False`, so one API model's exception destroyed sibling
  models' still-running requests mid-flight and lost their progress since
  the last checkpoint. Each model job is now wrapped in
  `_run_model_isolated()`; a failure is logged and reported, not fatal to
  its siblings.
- **Cyclic-permutation tie-breaks favored rotation order over the
  documented canonical order.** `_majority_vote`'s docstring promised ties
  break by canonical option ordering; the implementation actually returned
  whichever vote arrived first by rotation index — reintroducing the exact
  positional correlation cyclic permutation exists to cancel, on
  high-option benchmarks (MMLU-Pro, up to 10 options) where N-way ties are
  more likely. Now breaks ties by canonical letter order regardless of
  which rotation produced the vote first.

#### Documented

- Clarified that `CHOICEBENCH_*` env vars configure `setup_hpc.sh`/
  `env_hpc.sh`, while `#SBATCH` resource directives must be edited directly
  (`sbatch` parses them as literal text, not shell-substituted env vars).
- Added a Troubleshooting section: CUDA OOM on the HF backend, a bad/missing
  API key surfacing as `ProviderConfigurationError`, and telling a stale
  checkpoint from a fresh run via `--reset-run`.
- Reordered "Why ChoiceBench?" to lead with framework-level standout
  features (extensibility, async concurrency, single-command runs, the
  modal-k gate, research-grade stats) ahead of the lm-eval/Inspect/paper-
  repo comparisons.
- Dropped the MMLU/ARC-Challenge-only download size/time note (asymmetric
  across the 5 bundled benchmarks; extending it would mean guessing at
  unverified figures for the other 3).

### 2026-07-03 — Remove vLLM logprob scoring entirely

Follow-up to the "Changed (behavior decision)" entry below, which gated vLLM
out of config-driven logprob runs but left the underlying capability reachable
via direct Python construction. That escape hatch is now closed.

#### Removed

- **`VLLMClient.score_options_async()` deleted.** `VLLMClient` is now
  generate-only, mirroring `TogetherAIClient` exactly — no provider client
  implements logprob scoring; `pride`/`cyclic_logprob` require
  `backend: huggingface` (or `dummy`, for tests), full stop.
- **Cascading dead code removed**, since it was only reachable through the
  now-deleted vLLM score_options path:
  - `BaseBackend.supports_score_options()`, `APIBackend`'s overrides of it,
    `supports_logprobs`, and the `score_options()` ThreadPoolExecutor bridge.
  - `PriDeRunner._score_question_async()`,
    `PriDeRunner._cyclic_rollout_prob_matrix_async()`, and the vLLM branch of
    `PriDeRunner.run_many_async()` (now inherits `ExperimentRunner`'s base
    implementation).
  - `CyclicLogprobRunner.run_many_async()` (no replacement — this runner has
    no async path now, since no logprob-capable backend is ever
    async-capable).
  - `tests/runners/test_pride_async.py` (deleted entirely) and the
    vLLM-async test classes in `test_vllm_client.py` / `test_cyclic_logprob.py`.

#### Verified

- Full test suite passing after the removal (599 tests).

### 2026-07-03 — Statistical validity audit

Prompted by a fresh-eyes review of the metrics/config layer. Full detail in
commit `65d2b2a`.

#### Fixed

- **`accuracy` had no confidence interval.** Added a 95% Clopper-Pearson
  (exact binomial) CI to both `accuracy` and `accuracy_conditional`
  (`*_ci_low` / `*_ci_high`). Without this, "method A beats method B" wasn't
  statistically defensible. Cross-validated against `statsmodels`'
  independent implementation (200 random cases, zero mismatches). New core
  dependency: `scipy`.
- **`mad`'s denominator was inconsistent.** The gold-answer percentage was
  computed over *all* rows while the predicted-answer percentage was
  computed over *scored-only* rows — so MAD was silently inflated by a
  benchmark/method's parse-failure rate. Both percentages (and the
  bootstrap std) now use the same scored-subset denominator.

#### Added

- **`order_sensitivity` metric** (`order_rstd`, `order_flip_rate`). MAD only
  measures *marginal* answer-letter skew vs. the gold distribution — it
  doesn't tell you whether moving the same content to a different position
  changes the model's answer. `order_sensitivity` does, using
  `cyclic_logprob`'s per-rotation logprob data. Currently NaN for every
  other method, because none of them persist per-rotation data to disk
  (`cyclic_permutation` computes it internally for its majority vote but
  discards it before writing the row — a natural follow-up if this metric
  should cover more methods).

#### Changed (behavior decision)

- **vLLM is no longer accepted as a logprob backend for config-driven
  `pride`/`cyclic_logprob` runs.** vLLM's `score_options` only sees the
  top-20 generation logprobs (anything else floored to -100), a degraded
  approximation vs. HuggingFace's true full-vocabulary logit. Config
  validation (`load_config()`) now rejects `pride`/`cyclic_logprob` +
  `api`+`provider: vllm`. The underlying capability
  (`APIBackend.supports_logprobs` / `VLLMClient.score_options_async`) is
  unchanged for direct Python use outside `load_config()` if you want the
  degraded path anyway — see README, "Logprob methods on vLLM vs
  HuggingFace."

#### Cleaned up

- **`model_status` was overloaded** — transport success for `direct_mcq`/
  `two_stage`, answer-produced for `cyclic_permutation`/`cyclic_logprob`/
  `pride`. Split into two uniform columns: `transport_status` (did the
  backend call return; `None` for logprob-only methods that never call
  `generate()`) and `answer_status` (was a final answer produced — always
  set, uniform across all 5 methods).
- **`PriDeRunner`'s sync/async paths were near-duplicated** (`run_one` vs.
  `_score_question_async`, ~90 lines each, differing only in how the
  logprob call is made). Factored into a shared `_build_debiased_row()`
  helper.

#### Verified

- Full test suite: 612 → 621 passing.
- End-to-end DummyBackend run (toy + real-scale ARC-Challenge, 1,172
  questions) confirming the new metrics compute correctly and quickly
  (~1.2s) at real scale, not just in unit tests.
- Cold-clone-to-benchmark-prep pass through the README (fresh venv,
  `pip install -e .`, toy run, all 5 `prepare_data.py` commands) — all
  clean, no errors.

### 2026-07-02 — Variable-length choices, PriDe modal-k gate, safe run reset

#### Added

- **Variable-length choice schema.** Benchmarks move off the fixed A-D
  format onto an order-derived, variable-length representation (labels
  A..N are always re-derived from choice order, never trusted past J from
  source data). MMLU-Pro keeps up to 10 options and TruthfulQA keeps all
  `mc1` choices instead of truncating to 4. A stats sidecar
  (`src/choicebench/stats.py`) persists modal-k and the choice-count
  distribution next to each normalized CSV.
- **PriDe modal-k compatibility gate** (`src/choicebench/pride_gate.py`).
  Restricts a PriDe run to the modal-k subset and refuses to run when
  fewer than `pride.modal_k_threshold` (default 0.95) of questions share
  it, naming the benchmark, modal k, proportion, and threshold. Writes a
  gate-report sidecar with `n_evaluated`/`n_total` accounting.
- **`--reset-run` flag.** `safe_reset_run_dir()` resolves symlinks before a
  containment check and refuses to delete anything outside the runs root;
  wired in behind an explicit flag. Default behavior (reuse an existing
  run-id in place, resume checkpoints) is unchanged.

#### Documented

- README: new Method Compatibility & Known Limitations section explaining
  the modal-k gate, and that lowering the threshold does not enlarge the
  scored subset — it stays modal-k-only, just shrinks `n_evaluated` vs
  `n_total`.

### 2026-07-01 — HPC examples, MMLU/TruthfulQA normalization fixes

#### Fixed

- **MMLU choice normalization crashed on non-string `choices`.** HF/Pandas
  can hand back `choices` as a native array rather than a string;
  `ast.literal_eval(choices)` assumed the latter unconditionally. A new
  `_parse_choices()` handles both, and validates exactly 4 choices.
- **TruthfulQA's HF path moved.** The dataset relocated from `truthful_qa`
  to the namespaced `truthfulqa/truthful_qa` on HuggingFace; the old path
  no longer loads at all. Updated the `@benchmark` registration, docs, and
  test fixtures — no alias for the old path, since it fails before ever
  reaching our registry lookup.

#### Added

- Optional HPC/Slurm examples (`examples/hpc/setup_hpc.sh`, `env_hpc.sh`,
  `run_choicebench.sbatch`).

#### Cleaned up

- CI: fixed GitHub Actions test-workflow triggers.
- Release docs and registry consistency polish (`.env.example`,
  `CONTRIBUTING.md`, `README.md`, config templates, `io/readers.py`).
