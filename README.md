# choicebench

A reusable framework for running multiple-choice question (MCQ) evaluation experiments on LLMs.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Tests](https://img.shields.io/badge/tests-267%20passed-brightgreen)

MCQ evaluation is a well-studied LLM benchmark task, but the scaffolding is always the same: load a benchmark, call a model repeatedly, parse its response, score against the gold answer, save results, and compute metrics. This framework handles all of that so you can focus on the experimental condition — what varies between your runs. In five minutes you can run the toy experiment end-to-end. In an afternoon you can add a new debiasing method or metric and run it against MMLU.

---

## Quick Start

### Installation

```bash
git clone <repo-url> && cd choicebench
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

For API backends, copy `.env.example` to `.env` and add your keys:
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
GROQ_API_KEY=...
TOGETHER_API_KEY=...
```

### Run the toy experiment (no GPU, no API key needed)

```bash
python scripts/run_experiment.py --config config/toy_experiment.yaml --yes
```

This runs the built-in `DummyBackend` (fixed responses, no model) against a 10-question synthetic dataset. It completes in seconds and exercises the full pipeline: benchmark loading → method execution → checkpointing → result CSV writing.

### Evaluate results

```bash
python scripts/evaluate_run.py --run-id toy_experiment
```

Outputs a metrics table to stdout and writes a report to `reports/`.

### What the output looks like

```
runs/toy_experiment/
  config.yaml                            # copy of the config used
  toy_experiment__direct_mcq__dummy_1.csv    # one CSV per (method, model)
  toy_experiment__direct_mcq__dummy_2.csv
  checkpoints/                           # auto-cleaned on completion
```

Each result CSV has one row per question with columns for the prompt, raw model output, parsed choice, gold answer, and whether the answer was correct.

---

## Config Reference

A complete annotated example:

```yaml
experiment:
  name: my_experiment           # free-form label; used in log output and run ID

models:
  - backend: huggingface        # "huggingface" | "api" | "dummy"
    model_name_or_path: Qwen/Qwen2.5-7B-Instruct  # HF hub ID or local path
    device: cuda                # "cuda" | "cpu" | "auto" (multi-GPU)
    generation_kwargs:
      max_new_tokens: 512
      temperature: 0.0
      do_sample: false
    run:
      seed: 42

  - backend: api                # API-backed model; no GPU required
    provider: openai            # must match a key in CLIENT_REGISTRY
    model_name_or_path: gpt-4.1-mini
    generation_kwargs:
      max_new_tokens: 512
      temperature: 0.0

benchmark:
  name: mmlu                    # "mmlu" | "arc_challenge" | "toy"
  split: test                   # split name (benchmark-specific)
  n_samples: 100                # null = full split; integer = random subsample
  subject_filter: null          # null | list of MMLU subject strings

methods:
  - name: direct_mcq            # built-in; or "module.path:ClassName" for external
  - name: cyclic_permutation
  - name: pride
    requires_logprobs: true     # schema-validated; rejects non-logprob backends

metrics:
  - accuracy
  - mad
  # - my_package.metrics.my_metric:MyMetricMetric  # external metric

run:
  seed: 42
  resume: true                  # resume from checkpoint if one exists
  dry_run: false                # print plan and exit without running
  checkpoint_every_n: 50       # save progress to disk every N questions
  prompt_version: "v1"         # subdirectory under prompts/
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
   Then use `name: my_method` in your YAML config.

For external methods (not in this repo), skip step 3 and use `name: my_package.methods:MyMethodRunner` directly in YAML.

### Add a new metric (2 steps)

1. Copy `src/choicebench/metrics/templates/base_metric_template.py` to `src/choicebench/metrics/my_metric.py`
2. Implement `compute(results_df)` and add to `BUILTIN_METRICS` in `src/choicebench/metrics/__init__.py`

External: use `name: my_package.metrics:MyMetricMetric` in YAML — no framework files needed.

### Add a new API client (3 steps)

1. Copy `src/choicebench/clients/templates/base_client_template.py` to `src/choicebench/clients/my_provider_client.py`
2. Implement `_generate_provider_response()` — call the provider SDK, extract `raw_text`, return a `ModelResponse`
3. Register in `src/choicebench/registry.py` and add the API key to `.env`

The retry loop, backoff, semaphore, and request/response validation are all handled by `BaseClient`. You only implement the single raw API call.

### Add a new backend (3 steps)

1. Copy `src/choicebench/backends/templates/base_backend_template.py` to `src/choicebench/backends/my_backend.py`
2. Implement `generate()` (required) and optionally `score_options()` + `supports_logprobs = True`
3. Add a branch to `build_backend()` in `scripts/run_experiment.py`

### External registration via `"module.path:ClassName"` syntax

Any method or metric can be loaded from an external package — just install it in the same virtualenv and use the full import path in YAML. No framework files need to be touched.

---

## Architecture

```
config/              — YAML experiment configs
scripts/             — entry points (run_experiment.py, evaluate_run.py, prepare_data.py)
src/choicebench/
  config/            — schema validation, paths, provider defaults
  registry.py        — METHOD_REGISTRY and CLIENT_REGISTRY (single registration point)
  benchmarks/        — benchmark loaders (MMLU, ARC-Challenge, toy)
  backends/          — inference backends (HuggingFace, API, Dummy)
  clients/           — API provider clients (OpenAI, Gemini, Groq, Together)
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

**Data flow:** config → `load_benchmark()` → `build_backend()` → `instantiate_runner()` → `run_method()` (with checkpointing) → `write_run_results()` → `evaluate_run.py` → metrics report.

**Checkpointing:** progress is saved to `runs/<run_id>/checkpoints/` every `checkpoint_every_n` questions. If a run is interrupted, re-running the same command resumes from the last checkpoint automatically (when `resume: true`). Checkpoints are deleted on successful completion.

---

## Supported Built-ins

### Methods

| Name | Description | Calls per question |
|---|---|---|
| `direct_mcq` | Single-pass: prompt → parse → score | 1 |
| `cyclic_permutation` | Runs 4 option permutations, takes majority vote | 4 |
| `two_stage` | Stage 1: free-form answer; Stage 2: map to option letter | 2 |
| `pride` | PriDe debiasing via logprob calibration on a held-out set | N (calibration) + 1 |

### Metrics

| Name | Description |
|---|---|
| `accuracy` | Fraction of questions answered correctly |
| `mad` | Mean Absolute Deviation of accuracy across subjects |

### Clients (API backends)

| Provider key | Notes |
|---|---|
| `openai` | OpenAI API (gpt-4.1-mini, gpt-4.1, etc.) |
| `gemini` | Google Gemini API (gemini-2.5-flash, gemini-2.5-pro) |
| `groq` | Groq API (Llama, Mixtral models) |
| `together` | Together AI (Qwen, Llama, and other open-weight models) |

### Benchmarks

| Name | Source | Notes |
|---|---|---|
| `mmlu` | HuggingFace `cais/mmlu` | 57-subject, 14k questions; run `prepare_data.py` first |
| `arc_challenge` | HuggingFace `allenai/ai2_arc` | 1172-question subset; run `prepare_data.py` first |
| `toy` | Bundled synthetic CSV | 10 questions; no setup needed |

---

## SLURM / HPC

Two scripts in `scripts/slurm/`:

- **`submit_job.sh`** — single job, one config, one GPU node. Suitable for all API runs and local models up to ~30B parameters.
- **`submit_array.sh`** — job array, one task per config. Edit the `CONFIGS` array at the top of the file to run multiple experiments in parallel.

Both scripts require setting `REPO_ROOT` and `VENV_DIR` to match your cluster layout and are designed to be committed alongside your experiment configs for reproducibility.

```bash
CONFIG=config/my_experiment.yaml sbatch scripts/slurm/submit_job.sh
```
