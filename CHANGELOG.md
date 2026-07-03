# Changelog

Notable changes to ChoiceBench. Newest first.

## 2026-07-03 — Remove vLLM logprob scoring entirely

Follow-up to the "Changed (behavior decision)" entry below, which gated vLLM
out of config-driven logprob runs but left the underlying capability reachable
via direct Python construction. That escape hatch is now closed.

### Removed

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

### Verified

- Full test suite passing after the removal (599 tests).

## 2026-07-03 — Statistical validity audit

Prompted by a fresh-eyes review of the metrics/config layer. Full detail in
commit `65d2b2a`.

### Fixed

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

### Added

- **`order_sensitivity` metric** (`order_rstd`, `order_flip_rate`). MAD only
  measures *marginal* answer-letter skew vs. the gold distribution — it
  doesn't tell you whether moving the same content to a different position
  changes the model's answer. `order_sensitivity` does, using
  `cyclic_logprob`'s per-rotation logprob data. Currently NaN for every
  other method, because none of them persist per-rotation data to disk
  (`cyclic_permutation` computes it internally for its majority vote but
  discards it before writing the row — a natural follow-up if this metric
  should cover more methods).

### Changed (behavior decision)

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

### Cleaned up

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

### Verified

- Full test suite: 612 → 621 passing.
- End-to-end DummyBackend run (toy + real-scale ARC-Challenge, 1,172
  questions) confirming the new metrics compute correctly and quickly
  (~1.2s) at real scale, not just in unit tests.
- Cold-clone-to-benchmark-prep pass through the README (fresh venv,
  `pip install -e .`, toy run, all 5 `prepare_data.py` commands) — all
  clean, no errors.
