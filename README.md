# choicebench

ChoiceBench is a lightweight framework for MCQ evaluation-method research on LLMs, with built-in support for answer-order bias analysis and mitigation methods.

[![Tests](https://github.com/cotenthusiast/choicebench/actions/workflows/test.yml/badge.svg)](https://github.com/cotenthusiast/choicebench/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Some non-trunk branches in this repository produce results reported in a paper currently under peer review — see [PAPER.md](PAPER.md) for which branches and their status.

MCQ evaluation is a well-studied LLM benchmark task, but the scaffolding is always the same: load a benchmark, call a model repeatedly, parse its response, score against the gold answer, save results, and compute metrics. This framework handles all of that so you can focus on the experimental condition — what varies between your runs. In five minutes you can run the toy experiment end-to-end. In an afternoon you can add a new debiasing method or metric and run it against MMLU.

---

## Why ChoiceBench?

Most MCQ methodology research ships as bespoke per-paper code. 
Researchers studying answer-order bias, debiasing methods, or MCQ 
evaluation procedures typically reimplement the same experiment 
infrastructure from scratch, paper after paper. ChoiceBench is the 
reusable framework built for exactly this problem.

**Standout features:**

- **Zero-friction modular extensibility.** Drop in a new evaluation method, register a custom dataset, or add a new prompting intervention by writing one clean Python class — methods, metrics, benchmarks, backends, and clients all register in a few steps, or plug in from an external package with zero changes to this repo.
- **Configurable async concurrency.** Models run through a bounded async pipeline — each model's in-flight request count is capped by its own configurable `concurrency_limit`, so you can dial concurrency per provider or rate-limit without touching the run logic. Per-model failure isolation means one broken model or plugin can't take down the rest of a run.
- **Single-command audits.** One `run_experiment.py` invocation runs the full grid — any combination of benchmarks, models, and methods from a single YAML config — instead of juggling per-condition launcher scripts. Every run is checkpointed and safely resumable.
- **Dynamic variable-option handling.** Parsing and bias calculations scale natively past the standard four-choice (A–D) format up to ten choices (A–J), without manual dataset filtering or breaking on modern high-difficulty benchmarks.
- **Native support for five canonical benchmarks.** MMLU, ARC-Challenge, MMLU-Pro, HellaSwag, and TruthfulQA are pre-registered and one `prepare_data.py` command away — no custom normalizer to write for the datasets most MCQ research already uses.
- **A framework-level gate against invalid global calibration.** Any method that needs a single fixed option-count to compute a valid global statistic (PriDe's positional-bias prior is the bundled example) is protected by the modal-k compatibility gate, which blocks the run on a mismatched-option benchmark and reports exactly which questions were excluded — instead of silently producing a meaningless average.
- **Statistics built for research, not a leaderboard number.** Exact 95% confidence intervals on accuracy, and a marginal-skew metric (MAD) kept deliberately separate from a causal order-bias metric (order-sensitivity).

**vs. lm-evaluation-harness:** lm-eval is designed for model 
benchmarking, answering "how does this model perform across 
benchmarks?" ChoiceBench is designed for a different question: 
"how do different evaluation methods compare on the same 
questions?" If you want a leaderboard or capability report, lm-eval 
is the right tool. If you want to compare cyclic permutation against 
direct prompting on MMLU, ChoiceBench is.

**vs. Inspect AI:** Inspect is a strong general-purpose evaluation 
framework, well suited for frontier model auditing, agents, and 
safety evals. ChoiceBench is narrower by design. The experiment 
grid (benchmarks x models x methods) is the native unit. Adding a 
new evaluation method means implementing `run_one()` and the 
framework handles inference, checkpointing, parsing, scoring, and 
comparison reporting. If your research question is the evaluation 
method itself rather than the model, ChoiceBench is built around 
that question.

**vs. paper repos:** Much of the MCQ methodology literature relies 
on per-paper experiment code that works but is difficult to build 
on. ChoiceBench aims to be the shared foundation that makes results 
easier to reproduce and extend.

---

## Real Run Results

A single `run_experiment.py` invocation across all 5 bundled benchmarks (MMLU,
ARC-Challenge, MMLU-Pro, HellaSwag, TruthfulQA), `direct_mcq` + `cyclic_logprob`
+ `pride`, Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, 10 questions per
benchmark, seed 42. Cells are accuracy with the 95% CI in brackets.

| Benchmark | direct_mcq · Llama | direct_mcq · Qwen | cyclic_logprob · Llama | cyclic_logprob · Qwen | pride · Llama | pride · Qwen |
|---|---|---|---|---|---|---|
| MMLU | 0.50 [0.19–0.81] | 0.50 [0.19–0.81] | 0.60 [0.26–0.88] | 0.50 [0.19–0.81] | 0.80 [0.44–0.97] | 0.60 [0.26–0.88] |
| ARC-Challenge | 0.60 [0.26–0.88] | 0.90 [0.55–1.00] | 0.90 [0.55–1.00] | 1.00 [0.69–1.00] | 0.90 [0.55–1.00] | 0.90 [0.55–1.00] |
| HellaSwag | 0.30 [0.07–0.65] | 0.70 [0.35–0.93] | 0.50 [0.19–0.81] | 0.70 [0.35–0.93] | 0.20 [0.03–0.56] | 0.70 [0.35–0.93] |
| MMLU-Pro | 0.50 [0.19–0.81] | 0.20 [0.03–0.56] | 0.30 [0.07–0.65] | 0.50 [0.19–0.81] | gated | gated |
| TruthfulQA | 0.40 [0.12–0.74] | 0.50 [0.19–0.81] | 0.40 [0.12–0.74] | 0.70 [0.35–0.93] | gated | gated |

`pride` is gated on MMLU-Pro and TruthfulQA by the modal-k compatibility gate,
exactly as designed: MMLU-Pro's modal option count is k=10, but only 7/10
questions in this sample share it (70% coverage, below the 95% threshold), and
TruthfulQA's modal k=4 matches 0/10 questions (0% coverage) — both benchmarks
mix option counts, and PriDe's global positional-bias prior only means
anything over a single fixed count. The gate refuses the run on those cells
rather than average over an invalid prior; see [Method Compatibility & Known
Limitations](#pride-requires-a-fixed-label-set-size-the-modal-k-gate) below.

This is a 10-questions-per-benchmark demo run, not a statistically powered
benchmark claim — that's why the confidence intervals above are wide. It
exists to show the full grid running end-to-end, not to rank models or
methods.

---

## Reproduces published results

Unlike the demo grid above, this is a full-scale, single-cell reproduction of
Table 3 from Zheng et al., *"Large Language Models Are Not Robust Multiple
Choice Selectors"* (ICLR 2024, arXiv:2309.03882): MMLU, `llama-13B`
(LLaMA-1), 0-shot, N=13,564 questions, Default / Cyclic Perm / PriDe at three
calibration fractions.

| Method | Acc (ours) | Acc (paper) | Δ Acc | RStd (ours) | RStd (paper) | Δ RStd |
|---|---|---|---|---|---|---|
| Default | 43.0 | 34.6 | +8.4 | 14.3 | 17.4 | −3.1 |
| Cyclic Perm | 49.0 | 47.7 | +1.3 | 5.1 | 2.9 | +2.2 |
| PriDe (α=5%) | 44.8 ± 0.1 | 36.4 | +8.4 | 4.4 ± 0.5 | 5.7 | −1.3 |
| PriDe (α=40%) | 46.5 ± 0.3 | 40.4 | +6.1 | 4.4 ± 0.2 | 3.9 | +0.5 |
| PriDe (α=80%) | 48.1 ± 0.1 | 45.3 | +2.8 | 4.8 ± 0.1 | 2.6 | +2.2 |

The paper's qualitative story reproduces fully — monotone accuracy gains
with α, RStd collapse at every α, Cyclic Perm as the accuracy ceiling — but
absolute accuracy runs a systematic 1.3–8.4 point high, shrinking as more of
the scored set gets permutation-debiased treatment. See
[`examples/pride_reproduction.md`](examples/pride_reproduction.md) for the
full walkthrough: real commands and output, the exact config, and an honest
accounting of that deviation with candidate causes.

---

## Quick Start

### Installation

```bash
git clone https://github.com/cotenthusiast/choicebench && cd choicebench
python -m venv .venv && source .venv/bin/activate
pip install -e .
choicebench-prepare-toy
```

For local HuggingFace inference, install the optional dependencies instead:
```bash
pip install -e ".[hf]"
```

For API backends, copy `.env.example` to `.env` and add your keys:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
GROQ_API_KEY=...
TOGETHER_API_KEY=...
# Optional for local vLLM; defaults to "vllm" if unset.
VLLM_API_KEY=
```

### Run the toy experiment (no GPU, no API key needed)

```bash
choicebench-run --config config/toy_experiment.yaml --run-id toy_experiment --yes
```

This runs the built-in `DummyBackend` (fixed responses, no model) against a 10-question synthetic dataset. It completes in seconds and exercises the full pipeline: benchmark loading → method execution → checkpointing → result CSV writing.

Without `--run-id`, the run is written to a timestamped directory (e.g. `runs/20260628_113755/`) instead; pass `--run-id toy_experiment` so the evaluate step below can find it by name.

### Evaluate results

```bash
choicebench-evaluate --run-id toy_experiment
```

Logs a metrics summary and writes a self-identifying JSON report to
`reports/toy_experiment_eval_<hash>_metrics.json`. Ordinary evaluation and
`--reparse` have distinct evaluation IDs and cannot overwrite one another.

Expected output for the toy run is `accuracy: 0.3`: `DummyBackend` always answers
`A`, and 3 of the 10 toy questions have `A` as their correct option.

### What the output looks like

```
runs/toy_experiment/
  manifest.json                                    # immutable protocol/config/input identity
  run_state.json                                   # mutable completion/gating status only
  config.yaml                                      # original human-readable config
  results/cond_<hash>.csv                          # one file per canonical condition
  checkpoints/cond_<hash>.json                     # present only while interrupted
  artifacts/cond_<hash>/                           # method/gate sidecars
  artifacts/datasets/sel_<hash>.csv                # exact selected input rows
  artifacts/prompts/prompt_<hash>/*.txt             # exact prompt templates consumed
```

Human labels remain in each row, but uniqueness comes from `condition_id`, not
from filename sanitization. Every row also carries `experiment_id`, dataset
artifact/selection IDs, model/method IDs, prompt ID, and the exact split.

### Prepare MMLU or ARC-Challenge

The deterministic toy generator is invoked by `choicebench-prepare-toy`. For example, to prepare MMLU or
ARC-Challenge, normalize them once before running experiments:

```bash
choicebench-prepare --hf-path cais/mmlu --hf-subset all --split test
choicebench-prepare --hf-path allenai/ai2_arc --hf-subset ARC-Challenge --split test
```

Every HuggingFace dataset needs a registered `@benchmark` normalizer before
`prepare_data.py` can process it — there is no code-free path (see **Add a
benchmark**). For the benchmarks in the table below this is already done, so
`choicebench-prepare` writes a verified split-addressed artifact using the
registered benchmark name automatically, so `--output-name` is unnecessary
for built-ins. A generic `name: huggingface` config may use `output_name` as a
logical display/address component, but it still requires a registered
normalizer. Always prepare every evaluation and calibration split explicitly.

---

## Supported Benchmarks

| Benchmark | Name in config | Prepare command |
|---|---|---|
| MMLU | `mmlu` | `python scripts/prepare_data.py --hf-path cais/mmlu --hf-subset all` |
| ARC-Challenge | `arc_challenge` | `python scripts/prepare_data.py --hf-path allenai/ai2_arc --hf-subset ARC-Challenge` |
| MMLU-Pro | `mmlu_pro` | `python scripts/prepare_data.py --hf-path TIGER-Lab/MMLU-Pro` |
| HellaSwag | `hellaswag` | `python scripts/prepare_data.py --hf-path Rowan/hellaswag --split validation` |
| TruthfulQA | `truthful_qa` | `python scripts/prepare_data.py --hf-path truthfulqa/truthful_qa --hf-subset multiple_choice --split validation` |

A HuggingFace MCQ dataset can also be referenced by `hf_path` via `name: huggingface` (instead of by its registry name), but there is **no code-free path** — the dataset still needs a registered `@benchmark` normalizer first (see **Add a benchmark** below). Without one, `prepare_data.py` raises `NotImplementedError`. See `config/experiment_template.yaml` for the full schema.

---

## Config Reference

A complete annotated example:

```yaml
experiment:
  name: my_experiment           # free-form label; used in log output and paths

models:                         # one or more; every model runs every benchmark
  - backend: huggingface        # "huggingface" | "api" | "dummy"
    model_name_or_path: Qwen/Qwen2.5-7B-Instruct  # HF hub ID or local path
    revision: null               # optional model branch/tag/commit; identity-bearing
    device: cuda                # "cuda" | "cpu" | "auto" (multi-GPU)
    generation_kwargs:
      max_new_tokens: 512
      temperature: 0.0
      do_sample: false

  - backend: api                # API-backed model; no GPU required
    provider: openai            # "anthropic" | "openai" | "gemini" | "groq" | "together" | "vllm"
    model_name_or_path: gpt-4.1-mini
    generation_kwargs:
      max_new_tokens: 512
      temperature: 0.0

benchmarks:                     # a non-empty list (one entry per benchmark)
  - name: mmlu                  # "mmlu" | "arc_challenge" | "mmlu_pro" | "hellaswag" | "truthful_qa" | "toy" | "huggingface"
    split: test                 # split name (benchmark-specific)
    source_revision: null       # optional HF revision/commit; part of artifact identity
    transforms: []              # e.g. [permutation_safe_v1], must match preparation
    n_samples: 100              # null = full split; positive int = random subsample
    subject_filter: null        # null | list of MMLU subject strings

methods:
  - name: direct_mcq            # built-in; or "module.path:ClassName" for external
  - name: cyclic_permutation
  - name: two_stage
    # Optional: adds a 3rd call if stage 2 is unparseable.
    # params:
    #   fallback_on_parse_failure: true
  # A logprob method like pride can only be listed if EVERY model above is
  # logprob-capable (huggingface/dummy — no api provider is accepted).
  # With the OpenAI api model present, including it would fail config
  # validation — so it's commented out here:
  # - name: pride
  #   requires_logprobs: true   # schema-validated; rejects non-logprob backends
  #   preflight:                # optional: calibrates PriDe's positional-bias prior
  #     source: benchmark        # "benchmark" (reuse this benchmark's data) or a .csv/.jsonl file path
  #     split: validation        # must differ from the benchmark's own `split:` above (enforced — raises if equal)
  #     n: 100                   # number of calibration questions to sample (deterministic, seeded by run.seed)

metrics:
  - accuracy
  - mad
  # - my_package.metrics.my_metric:MyMetricMetric  # external metric

run:
  seed: 42                      # global seed (subsampling and generation)
  resume: true                  # resume from checkpoint if one exists
  dry_run: false                # print plan and exit without running
  checkpoint_every_n: 50        # save progress to disk every N questions
  prompt_version: "v1"          # packaged prompt bundle; file contents are hashed
  prompt_dir: null               # optional external root for custom prompt bundles
  concurrency_limit: 10         # max in-flight requests per API model (rate-limit guard)
```

---

## Extending the Framework — The Plugin System

### Add a new method (3 steps)

1. Copy `src/choicebench/methods/templates/base_method.py` to `src/choicebench/methods/library/my_method.py`
2. Implement `run_one()` — build prompt → call backend → parse and score → return `_build_result_row(...)`
3. Add to `src/choicebench/registry.py`:
   ```python
   from choicebench.methods.library.my_method import MyMethodRunner
   METHOD_REGISTRY["my_method"] = MyMethodRunner
   ```
   Re-export built-in methods from `src/choicebench/methods/library/__init__.py` if you want them available from `choicebench.methods`. Then use `name: my_method` in your YAML config.

For external methods (not in this repo), skip step 3 and use `name: my_package.methods:MyMethodRunner` directly in YAML.

### Add a new metric (2 steps)

1. Copy `src/choicebench/metrics/templates/base_metric_template.py` to `src/choicebench/metrics/my_metric.py`
2. Implement `compute(results_df)` and add to `BUILTIN_METRICS` in `src/choicebench/metrics/__init__.py`

External: list the import path directly in YAML, e.g.
`metrics: ["my_package.metrics:MyMetricMetric"]` — no framework files needed.

### Add a new API client (3 steps)

1. Copy `src/choicebench/clients/templates/base_client_template.py` to `src/choicebench/clients/my_provider_client.py`
2. Implement `_generate_provider_response()` — call the provider SDK, extract `raw_text`, return a `ModelResponse`
3. Register in `src/choicebench/registry.py` and add the API key to `.env`

The retry loop, backoff, semaphore, and request/response validation are all handled by `BaseClient`. You only implement the single raw API call. Within a run, all API models for a (benchmark, method) execute concurrently via `asyncio.gather()`, and the questions in each batch run concurrently bounded by that model's `concurrency_limit` semaphore; only benchmarks and methods iterate serially. The `concurrency_limit` knob is what caps in-flight requests and prevents provider rate-limit (429) errors.

### Add a new backend (4 steps)

1. Copy `src/choicebench/backends/templates/base_backend_template.py` to `src/choicebench/backends/my_backend.py`
2. Implement `generate()` (required) and optionally `score_options()` + `supports_logprobs = True`
3. Add a branch to `build_backend()` in `scripts/run_experiment.py`
4. Add the backend key to `_VALID_BACKENDS` in `src/choicebench/config/schema.py`. If it implements `score_options()`, set `supports_logprobs = True` on the class and register it in `_NONAPI_BACKEND_CLASSES` (same file). Logprob capability is decided by a single function, `model_supports_logprobs()`, which reads each backend's own `supports_logprobs` — both config validation and the pre-run gate consult it, so there is no second list to keep in sync.

### Add a new benchmark (4 steps)

1. Create `src/choicebench/benchmarks/my_bench.py`
2. Implement `build_normalized_dataframe(df)` — convert the raw HuggingFace
   DataFrame into the normalized schema (`question_id`, `question_text`,
   `choice_a`–`choice_d`, `correct_option`, `subject`, …) — and decorate it:
   ```python
   from choicebench.benchmarks.registry import benchmark

   @benchmark(name="my_bench", hf_path="org/my-bench", hf_subset="default", default_split="test")
   def build_normalized_dataframe(df):
       ...
   ```
3. Import the module in `src/choicebench/benchmarks/__init__.py` so the
   `@benchmark` decorator runs and registers the entry on import.
4. Run `python scripts/prepare_data.py --hf-path org/my-bench --hf-subset default`.
   Because the `hf_path` is registered, the normalized CSV is written to the
   stem `my_bench` automatically (no `--output-name` needed), which is exactly
   what `name: my_bench` in your config loads.

There is no code-free shortcut: every HuggingFace dataset needs a registered
`@benchmark` normalizer (the four steps above). Once registered, you can
reference it either by its registry name (`name: my_bench`) or generically by
`hf_path` (`name: huggingface` with `hf_path`/`output_name` in the config, passing
a matching `--output-name` to `prepare_data.py`).

### External registration via `"module.path:ClassName"` syntax

Any method or metric can be loaded from an external package — just install it in the same virtualenv and use the full import path in YAML. No framework files need to be touched.

### Worked example

`examples/method_comparison.md` walks through adding a new method
(`shuffled_baseline`) end-to-end — implementation, registration, a real
experiment config, and a real run against the toy dataset compared with
`direct_mcq`.

---

## Architecture

```
config/              — YAML experiment configs
scripts/             — entry points (run_experiment.py, evaluate_run.py, prepare_data.py, prepare_toy_data.py)
src/choicebench/
  config/            — schema validation, paths, provider defaults
  registry.py        — METHOD_REGISTRY and CLIENT_REGISTRY (single registration point)
  benchmarks/        — benchmark normalizers (MMLU, ARC-Challenge, MMLU-Pro, HellaSwag, TruthfulQA)
  backends/          — inference backends (HuggingFace, API, Dummy)
  clients/           — API provider clients (Anthropic, OpenAI, Gemini, Groq, Together, vLLM)
  methods/
    base.py          — ExperimentRunner ABC (shared infra: backend calls, result assembly)
    library/         — built-in method implementations
    templates/       — copy-paste templates for new methods
  metrics/
    base.py          — BaseMetric ABC
    library (inline) — built-in metrics (accuracy, MAD)
    templates/       — copy-paste templates for new metrics
  infra/             — checkpointing (CheckpointManager) and caching
  io/                — result readers/writers
  parsing/           — model output parsing
  scoring/           — correctness scoring
```

**Data flow:** config -> verified prepared artifact -> immutable manifest and
condition grid -> backend/method execution -> condition-addressed results ->
manifest-driven evaluation.

### Protocol-v2 identity and provenance

ChoiceBench v0.2 makes every result self-identifying. Preparation maps the
logical benchmark, source/config, requested split, source revision,
normalization version, and declared transforms to a distinct directory:

```
data/processed/<name>/<split>/norm-v2/src_<spec-hash>/
  normalized.csv
  artifact.json
  stats.json
```

`artifact.json` records the exact specification, row count, source metadata
(including the resolved Hub commit where available),
and a semantic SHA-256 digest of the ordered normalized rows. Loading
recomputes that digest. A changed CSV, mismatched metadata file, stale generic
`<name>_normalized.csv`, or wrong split is rejected. Legacy CSVs are never
silently assigned provenance; re-run `choicebench-prepare` for the exact split.

Before inference, `choicebench-run` resolves all datasets and calibration
splits, hashes and archives the exact prompt contents, canonicalizes and credential-sanitizes the complete model
and method configurations, resolves Hugging Face model refs to commits (or
hashes every file in a local model tree), fingerprints configured external
method/metric modules, records the source commit/dirty source-tree digest,
ChoiceBench/Python/dependency versions, expands the declared condition grid,
and writes `manifest.json`. Its timestamp is informational and excluded from
the deterministic experiment identity.

A condition ID covers benchmark + split + prepared artifact + deterministic
sample selection + calibration selection + full model/backend/provider/endpoint
configuration + full method parameters + prompt bundle + seed. Therefore two
same-named methods with different parameters, or two same-display-name models
on different providers/endpoints, cannot share a result or checkpoint.

Resume is accepted only when the newly resolved experiment digest exactly
matches the existing immutable manifest. Changes to code/protocol, seed,
dataset contents, split, sampled rows, prompt contents, method parameters,
model/provider/backend settings, generation settings, or dependencies reject
resume with instructions to choose a new run ID or use `--reset-run`.
Nonempty v0.1 run directories without a manifest are treated as unverifiable
legacy runs and cannot be resumed.

Presentation and operational controls (`experiment.name`, `run.resume`,
`run.dry_run`, checkpoint cadence, API concurrency, and YAML grid ordering) do
not invalidate resume. They remain in the config snapshot but are excluded
from scientific identity. `run_state.json` is deliberately separate and mutable; it records whether
each immutable condition is pending, completed, failed, or gated. Evaluation
reads only result paths declared by the manifest and rejects missing,
unexpected, duplicate, identity-conflicting, or digest-corrupt artifacts.
Failed conditions do not block evaluation: they are accounted in the report
(status and error, no metrics) and the report's `run_status` marks the run as
`partial`; pending conditions refuse evaluation until the run is finished.
The evaluation ID binds the exact validated result contents, so re-evaluating
changed results produces a new report file rather than overwriting the old one.

For papers, report at least the ChoiceBench version, protocol version, and
`experiment_id` from `manifest.json`, and archive the run manifest with the
configuration and results.

Example (abbreviated; recognized credentials are redacted and scientific file
inputs are represented by logical names/content digests, not absolute paths):

```json
{
  "schema_version": "choicebench.manifest.v2",
  "experiment_id": "exp_0123456789abcdef",
  "payload": {
    "protocol_version": "choicebench.protocol.v2",
    "datasets": [{"artifact_id": "ds_...", "split": "test"}],
    "conditions": [{"condition_id": "cond_...", "result_path": "results/cond_....csv"}]
  }
}
```

### Installed workspace policy

Prompt templates are read-only package resources. User data is never written
inside `site-packages`. `CHOICEBENCH_HOME`, when set, is the workspace root;
otherwise the current working directory is used. Its `data/`, `runs/`, and
`reports/` children contain generated artifacts. The supported installed
commands are `choicebench-prepare`, `choicebench-prepare-toy`,
`choicebench-run`, and `choicebench-evaluate`; the existing `scripts/*.py`
entry points remain available from a source clone.

Set `run.prompt_dir` to an external prompt root for a custom bundle. The
logical version and template contents are identity-bearing; the machine path is
not stored. `.env` is loaded from `CHOICEBENCH_HOME` (or the current workspace
when that variable is unset).

ChoiceBench enforces one writer process per run ID with an advisory lock.
Concurrent API requests and models inside that process remain supported. Slurm
array tasks must use distinct run IDs; a second process targeting an active run
fails before touching the manifest, checkpoints, or results. `--reset-run` is
subject to the same lock and refuses symlinked run directories.

Credential handling is schema-aware: unambiguous credential-named keys
(`api_key`, `authorization`, `client_secret`, …) are refused outright in
scientific configuration — rename the parameter if it is ordinary method
configuration — while secret-shaped but scientific names (`token`,
`secret_strength`, …) keep their identity-affecting values. Value-level
sanitization still covers bearer tokens, URL userinfo, and common signed-query
schemes in endpoints and error text; it cannot identify an arbitrary secret
embedded in ordinary prompt/model-output prose. Inspect artifacts before
public release and never place credentials in prompts or scientific parameters.
External method/metric identity binds the configured class's defining module,
distribution version, and owning external package tree. Dynamically loaded code
outside that package tree remains the extension author's responsibility. Remote APIs
may remain nondeterministic and may not expose an immutable server-side model
revision even when the client configuration is fully identified.

**Checkpointing:** progress is saved to
`runs/<run_id>/checkpoints/<condition_id>.json` every `checkpoint_every_n`
questions. Checkpoints embed the experiment, condition, and dataset-selection
identities; corruption or mismatch is a hard error. They are deleted after a
condition completes successfully.

---

## Supported Built-ins

### Methods

| Name | Description | Calls per question |
|---|---|---|
| `direct_mcq` | Single-pass: prompt → parse → score | 1 |
| `cyclic_permutation` | Runs one cyclic permutation per available option, takes majority vote | N options, normally 4 |
| `two_stage` | Stage 1: free-form answer; Stage 2: map to option letter | 2, or 3 if fallback is enabled |
| `pride` | PriDe Eq. 8 logprob debiasing. YAML-driven runs use a uniform prior unless a preflight calibration block is configured (see the `preflight:` example in [Config Reference](#config-reference)). Without preflight, this is logprob argmax only, not calibration-fitted debiasing. Subject to the [modal-k gate](#method-compatibility--known-limitations). | 1 score_options call per eval row; +4K calibration calls per run if K calibration rows are supplied |
| `cyclic_logprob` | Eq. 1 logprob averaging: score every cyclic permutation via `score_options`, average probability mass back to canonical slots, argmax | N options, normally 4 score_options calls |

#### Authoritative answer column per method

Every method writes its final answer to **`parsed_choice`**, and both built-in
metrics (`accuracy`, `mad`) read that column. This is uniform across all methods:

| Method | Authoritative answer column | How it's produced |
|---|---|---|
| `direct_mcq` | `parsed_choice` | parsed from the single `raw_text` generation |
| `cyclic_permutation` | `parsed_choice` | majority vote across rotations |
| `two_stage` | `parsed_choice` | parsed from stage-2 (or fallback) generation |
| `cyclic_logprob` | `parsed_choice` | Eq. 1 argmax over averaged logprobs |
| `pride` | `parsed_choice` | Eq. 8 debiased argmax (also mirrored in `pride_adjusted_choice`) |

`pride` additionally records `pride_adjusted_choice` and `cyclic_logprob` records
`option_distributions_json`, but these are diagnostics — `parsed_choice` is the
single authoritative column for scoring and for any user `groupby`.

Every row also carries two separate status columns, each with a single,
uniform meaning across all five methods:

| Column | Meaning | `"success"` means |
|---|---|---|
| `transport_status` | Did the backend call return? | The backend call returned — even if the output was unparseable, so `parsed_choice` may still be `None`. `None` for methods that never call `generate()` (`cyclic_logprob`, `pride` — they only call `score_options`), since there is no transport event to report. |
| `answer_status` | Was a final answer produced? | `parsed_choice` ended up set, by whatever method-specific process produces it (single-call parse, majority vote, or logprob argmax/debiasing). Always non-`None`. |

These used to be folded into a single overloaded `model_status` column whose
meaning silently changed by method (transport success for `direct_mcq`/
`two_stage`, answer-produced for the other three) — `transport_status` and
`answer_status` replace it so a filter behaves identically regardless of
method. For a cross-method answer-presence filter, `answer_status == "success"`
is now equivalent to `parsed_choice.notna()` (or `parse_status == "parse_ok"`).

#### Logprob methods (pride, cyclic_logprob)

`pride` and `cyclic_logprob` read per-letter log-probabilities via
`score_options`. Only **HuggingFace** (and `dummy`, for tests) implements it —
it reads the true full-vocabulary logit for each option label (one forward
pass). Option labels must be **single tokens** in the model's tokenizer; a
multi-token label raises a `ValueError` naming the offending letter and
suggesting the space-prefixed form.

No API provider client (OpenAI, Anthropic, Gemini, Groq, Together, vLLM)
implements `score_options` — they are all generate-only. **Config-driven runs
(`config.yaml` + `load_config()`) reject `pride`/`cyclic_logprob` on any `api`
backend** — use `backend: huggingface` instead.

Partially-degraded rows are flagged: `n_permutations_failed` / `n_permutations_total`
record how many permutations fell back to a uniform distribution (for `pride`,
these are the calibration rollout permutations).

**PriDe calibration sidecar:** a cached prior is reused only when the model-name
slug, calibration benchmark, calibration question ids, `calibration_seed`,
`temperature`, `max_tokens`, and `prompt_version` all match. It does **not**
capture model weights, so two local checkpoints sharing a basename
(`org/model` → `org_model`) collide on the sidecar path — use a distinct
`run_id`/calibration directory for those.

### Metrics

| Name | Description |
|---|---|
| `accuracy` | Fraction of questions answered correctly (`accuracy`/`accuracy_conditional`), each with a 95% Clopper-Pearson confidence interval (`*_ci_low`/`*_ci_high`) |
| `mad` | Mean absolute deviation between the model's letter-selection distribution and the gold answer distribution, both computed over the same scored subset — a marginal answer-letter skew indicator, *not* a measure of causal answer-order bias |
| `order_sensitivity` | Causal order-bias signal from per-rotation data (`order_rstd`, `order_flip_rate`) — currently only populated for `cyclic_logprob`, since it's the only method that persists per-rotation logprobs; NaN for other methods |

### Clients (API backends)

| Provider key | Notes |
|---|---|
| `anthropic` | Anthropic Messages API (Claude models) |
| `openai` | OpenAI API (gpt-4.1-mini, gpt-4.1, etc.) |
| `gemini` | Google Gemini API (gemini-2.5-flash, gemini-2.5-pro) |
| `groq` | Groq API (Llama, Mixtral models) |
| `together` | Together AI (Qwen, Llama, and other open-weight models) |
| `vllm` | Local vLLM OpenAI-compatible server; configure `base_url`, no hosted API key required. Generate-only, like the other API clients — not accepted for `pride`/`cyclic_logprob`, see [Logprob methods](#logprob-methods-pride-cyclic_logprob) |

### Benchmarks

| Name | Source | Notes |
|---|---|---|
| `mmlu` | HuggingFace `cais/mmlu` | 57-subject, 14k questions; run `prepare_data.py` first |
| `arc_challenge` | HuggingFace `allenai/ai2_arc` | 1172-question subset; run `prepare_data.py` first |
| `mmlu_pro` | HuggingFace `TIGER-Lab/MMLU-Pro` | All options preserved (up to 10 per question); no rows dropped. Modal option count is 10 at 83.0% coverage, so PriDe fails the default 0.95 modal-k gate (see [Method Compatibility](#method-compatibility--known-limitations)) |
| `hellaswag` | HuggingFace `Rowan/hellaswag` | Use the `validation` split; test labels are unavailable |
| `truthful_qa` | HuggingFace `truthfulqa/truthful_qa`, subset `multiple_choice` | Uses `mc1_targets`; all raw mc1 choices are now kept (2–13 per question, no longer truncated to 4); no rows dropped. Heterogeneous option counts (modal k=4 at 26.8% coverage), so PriDe fails the default 0.95 modal-k gate |
| `toy` | Bundled synthetic CSV | 10 questions; no setup needed |
| `huggingface` | User-specified HuggingFace dataset | Requires a registered `@benchmark` normalizer module (see **Add a benchmark**) |

### Option Schema

Normalized CSVs use a variable-choice schema: a `choices_json` column (an
ordered list of `{text, source_index}` objects), a `correct_index` into that
list, a derived `correct_option` letter, and `n_choices`. Render labels
(A, B, C, …) are always re-derived from choice order — never trusted from the
source dataset, which is inconsistent past J — while `source_index` preserves
each option's original position for audit. This supports benchmarks with any
number of options (e.g. MMLU-Pro's up to 10, A–J).

Legacy `choice_a`–`choice_d` CSVs still load unchanged: the reader falls back
to those columns when `choices_json` is absent, so previously prepared datasets
do not need to be re-prepared.

`prepare_data.py` also writes a `<name>_stats.json` sidecar recording the modal
option count (k) and its coverage. PriDe consumes this via a modal-k
compatibility gate (see [Method Compatibility](#method-compatibility--known-limitations)
below); direct_mcq and cyclic_permutation handle mixed option counts directly
and are not gated.

---

## Method Compatibility & Known Limitations

### PriDe requires a fixed label-set size (the modal-k gate)

PriDe estimates a single global positional-bias prior and applies it to every
evaluation question (Zheng et al., ICLR 2024, Eq. 8). That calibration is only
meaningful when the evaluation questions share **one** option count — i.e. a
fixed label set A..k. PriDe cannot meaningfully calibrate one global prior
across a benchmark whose questions have highly variable choice counts.

To enforce this, PriDe runs behind a **modal-k compatibility gate**, configured
by `pride.modal_k_threshold` (default `0.95`):

- The benchmark's modal choice count *k* is computed from the exact selected
  dataset rows archived in the run manifest snapshot.
- If at least a `modal_k_threshold` proportion of the loaded questions have
  exactly *k* options, the run proceeds **on the modal-k subset only** — the
  non-modal-k questions are excluded.
- Otherwise PriDe refuses to run and raises a clear error naming the benchmark,
  the modal *k*, the actual modal-k proportion, and the configured threshold.

**What lowering `modal_k_threshold` does — and does *not* — do.** Lowering the
threshold does **not** expand which questions PriDe scores. PriDe always
evaluates only the modal-k subset, whatever the threshold is set to. Lowering
the threshold only changes whether the run is *permitted to proceed at all* on a
more heterogeneous benchmark — at the cost of a smaller `n_evaluated` relative
to `n_total`. To see exactly what fraction you actually scored, consult the
per-condition gate report sidecar,
`runs/<run_id>/artifacts/<condition_id>/modal_k_gate.json`,
and its `n_evaluated` vs `n_total` accounting — the same figures are mirrored
onto each result row as `gate_n_evaluated` / `gate_n_total`.

**Only PriDe is gated.** `direct_mcq` and `cyclic_permutation` are *not* subject
to this gate. They are per-question methods with no global calibration step, so
they handle variable choice counts natively, question by question.

**Benchmarks affected today.** Under the default `0.95` threshold, two of the
bundled benchmarks fail the gate, so PriDe refuses to run on them as-is:

| Benchmark | Modal *k* | Modal-k coverage | PriDe at default 0.95 |
|---|---|---|---|
| MMLU-Pro | 10 | 83.0% | rejected (coverage < 0.95) |
| TruthfulQA | 4 | 26.8% | rejected (coverage < 0.95) |

`mmlu`, `arc_challenge`, `hellaswag`, and `toy` are effectively homogeneous
(≥99% of questions at k=4) and pass the gate. To run PriDe on MMLU-Pro or
TruthfulQA you must lower `pride.modal_k_threshold`, accepting that PriDe will
still only score the modal-k subset (9,981 of 12,032 for MMLU-Pro; 219 of 817
for TruthfulQA) — check the gate report for the exact `n_evaluated`.

---

## Optional SLURM / HPC examples

The local setup in **Quick Start** is still the recommended path for normal
machines. On HPC/Slurm systems, you may need to load a site-specific Python
module, keep virtualenvs and HuggingFace caches on scratch storage, and submit
runs through `sbatch`.

Editable templates are in `examples/hpc/`:

- `setup_hpc.sh` - one-time virtualenv setup and editable install.
- `env_hpc.sh` - reusable environment activation script for batch jobs.
- `run_choicebench.sbatch` - minimal Slurm example using the toy config and no
  GPU by default.

These two mechanisms apply at different layers, and both are real:

- `CHOICEBENCH_BASE` / `CHOICEBENCH_REPO` / `CHOICEBENCH_PYTHON_MODULE` (plus
  `CHOICEBENCH_VENV`, `PYTHON_BIN`) are **environment variables** read by
  `setup_hpc.sh` and `env_hpc.sh` (`${VAR:-default}`) — set them before running
  either script to control paths, the HF cache location, and which `module
  load` (if any) is used. No template editing is needed for these.
- `#SBATCH` resource directives (partition, GPU, time, memory, CPUs) inside
  `run_choicebench.sbatch` are **not** environment-variable-driven — `sbatch`
  parses `#SBATCH` lines as literal text before the script body ever runs, so
  exporting a shell variable cannot change them. To change these, edit the
  `#SBATCH` lines in `run_choicebench.sbatch` directly for your site (e.g.
  uncomment and set `#SBATCH --partition=...` / `#SBATCH --gres=gpu:1`).

`run_choicebench.sbatch` automatically `source`s `env_hpc.sh` itself — you do
not need to source it manually before `sbatch`. It also reads
`CHOICEBENCH_CONFIG` and `CHOICEBENCH_RUN_ID` env vars (falling back to the
toy config / an auto-generated run id) if you want to point it at a real
experiment without editing the file.

Example:

```bash
export CHOICEBENCH_BASE=/path/to/scratch/choicebench
export CHOICEBENCH_REPO=/path/to/choicebench
export CHOICEBENCH_PYTHON_MODULE=python3/3.10.5/gcc-9.3.0  # example only
bash examples/hpc/setup_hpc.sh
sbatch examples/hpc/run_choicebench.sbatch
```

These examples are convenience starting points, not a guarantee that every
HPC environment works unchanged.

There are also compact submission helpers in `scripts/slurm/` for users who
already have a virtualenv and know the resources they want.
`scripts/slurm/submit_job.sh` does **not** read `CHOICEBENCH_*` env vars — its
virtualenv path (`VENV_DIR`) and its `#SBATCH` resources are hardcoded in the
script and must be edited directly for your cluster:

```bash
# Edit these lines in scripts/slurm/submit_job.sh for your site:
#   VENV_DIR="$HOME/venvs/choicebench"   -> your venv path
#   #SBATCH --partition=gpu              -> your GPU partition name
#   #SBATCH --gres=gpu:1                 -> e.g. gpu:a100:2
#   #SBATCH --time=24:00:00              -> your wall-clock budget

CONFIG=config/my_experiment.yaml sbatch scripts/slurm/submit_job.sh
```

`CONFIG` and `RUN_ID` *are* read as env vars by the script body (not
`#SBATCH` directives), so those two can stay as shown without editing the
file. To override resources per-submission instead of editing the file,
`sbatch` command-line flags take precedence over the script's `#SBATCH`
lines:

```bash
CONFIG=config/my_experiment.yaml sbatch --partition=mypartition \
    --gres=gpu:a100:2 --time=12:00:00 scripts/slurm/submit_job.sh
```

---

## Troubleshooting

**CUDA out-of-memory on the HuggingFace backend.** The HF backend loads
weights in `fp16` on CUDA/auto (`fp32` on CPU) onto the device(s) given by
`device:` in your config. If a
model doesn't fit, either switch to a smaller `model_name_or_path`, set
`device: auto` to shard across all visible GPUs, reduce
`generation_kwargs.max_new_tokens`, or move the model to a node/partition with
more GPU memory; v0.2 has no built-in quantization fallback.

**A model works everywhere except this run, and it's a `ProviderConfigurationError`.**
API clients (`src/choicebench/clients/`) raise
`choicebench.clients.types.ProviderConfigurationError` — not retried, unlike
transient errors — when the provider rejects the request as unfixable by
retrying: a missing/empty API key, an invalid key, or a malformed request
(HTTP 400/401/404/422). Check that the corresponding `*_API_KEY` is set in
`.env` (see **Installation**) and actually valid for the `provider` named in
your config.

**Is this a stale/corrupt checkpoint or a fresh run?** Re-running the same
`--run-id` first rebuilds and verifies the immutable experiment manifest.
Only an identical experiment may resume or reuse a digest-verified completed
condition. Scientific changes and corrupt checkpoints/results are hard errors
before mixing can occur. Use a new run ID to preserve the prior experiment, or
`--reset-run` for an intentional destructive replacement; reset is refused
while another process holds the run lock.

**`module: command not found` inside a Slurm job's `.err` log, even though
`module load` works fine when you run it by hand.** `module` is a bash
function defined by your cluster's profile scripts, which only get sourced in
an interactive/login shell. `sbatch` inherits the environment of the shell
that invoked it — so if the job was submitted from a shell that never sourced
those profile scripts (a single `ssh host "sbatch job.sh"` command, CI,
cron, or any other non-interactive automation), `module` was never defined
there, and the identical `module load` line inside the submitted script fails
even though it works when you submit from a live terminal. Either always
submit from a real interactive terminal, or stop depending on `module load`
inside the sbatch script at all: resolve the interpreter to an absolute path
once (e.g. `module load <name>` then `command -v python3`, done interactively
a single time) and reference that absolute path directly in the script
instead.

---

## Roadmap

Planned v0.2 work:

- **Mixed-option PriDe calibration** — a single PriDe run that calibrates across questions with *different* option counts (rather than gating to the modal k, as it does today).
- **Parallel orchestration across benchmark/method jobs** — ChoiceBench runs API models concurrently (`asyncio.gather`) and questions concurrently under each model's `concurrency_limit`; benchmarks and methods still iterate serially.
- **Inspect AI adapter** — run ChoiceBench methods inside [Inspect](https://inspect.ai) workflows.
- **Broader benchmark adapters and stronger script-level integration tests.**

---

## Citation

If you use ChoiceBench in your research, please cite:

```bibtex
@software{choicebench2026,
  author    = {Hanna, Karl},
  title     = {ChoiceBench: A lightweight framework for MCQ evaluation-method research},
  year      = {2026},
  url       = {https://github.com/cotenthusiast/choicebench},
  version   = {0.2.0},
  license   = {MIT}
}
```
