# choicebench

ChoiceBench is a lightweight framework for MCQ evaluation-method research on LLMs, with built-in support for answer-order bias analysis and mitigation methods.

[![Tests](https://github.com/cotenthusiast/choicebench/actions/workflows/test.yml/badge.svg)](https://github.com/cotenthusiast/choicebench/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

MCQ evaluation is a well-studied LLM benchmark task, but the scaffolding is always the same: load a benchmark, call a model repeatedly, parse its response, score against the gold answer, save results, and compute metrics. This framework handles all of that so you can focus on the experimental condition — what varies between your runs. In five minutes you can run the toy experiment end-to-end. In an afternoon you can add a new debiasing method or metric and run it against MMLU.

---

## Quick Start

### Installation

```bash
git clone https://github.com/cotenthusiast/choicebench && cd choicebench
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

For local HuggingFace inference, install the optional dependencies instead:
```bash
pip install -e ".[hf]"
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
python scripts/run_experiment.py --config config/toy_experiment.yaml --run-id toy_experiment --yes
```

This runs the built-in `DummyBackend` (fixed responses, no model) against a 10-question synthetic dataset. It completes in seconds and exercises the full pipeline: benchmark loading → method execution → checkpointing → result CSV writing.

Without `--run-id`, the run is written to a timestamped directory (e.g. `runs/20260628_113755/`) instead; pass `--run-id toy_experiment` so the evaluate step below can find it by name.

### Evaluate results

```bash
python scripts/evaluate_run.py --run-id toy_experiment
```

Logs a metrics summary and writes a JSON report to `reports/`.

### What the output looks like

```
runs/toy_experiment/
  config.yaml                                      # copy of the config used
  toy_experiment_direct_mcq_dummy_1_toy.csv        # one CSV per (run, method, model, benchmark)
  toy_experiment_direct_mcq_dummy_2_toy.csv
  checkpoints/                                     # auto-cleaned on completion
```

The CSV filename encodes `<run_id>_<method>_<model>_<benchmark>`. Each result CSV has one row per question with columns for the prompt, raw model output, parsed choice, gold answer, and whether the answer was correct.

### Prepare MMLU or ARC-Challenge

The bundled toy CSV needs no setup. For the built-in HuggingFace benchmarks,
normalize them once before running experiments:

```bash
python scripts/prepare_data.py --hf-path cais/mmlu --hf-subset all --output-name mmlu
python scripts/prepare_data.py --hf-path allenai/ai2_arc --hf-subset ARC-Challenge --output-name arc_challenge
```

---

## Supported Benchmarks

| Benchmark | Name in config | Prepare command |
|---|---|---|
| MMLU | `mmlu` | `python scripts/prepare_data.py --hf-path cais/mmlu --hf-subset all` |
| ARC-Challenge | `arc_challenge` | `python scripts/prepare_data.py --hf-path allenai/ai2_arc --hf-subset ARC-Challenge` |
| MMLU-Pro | `mmlu_pro` | `python scripts/prepare_data.py --hf-path TIGER-Lab/MMLU-Pro` |
| HellaSwag | `hellaswag` | `python scripts/prepare_data.py --hf-path Rowan/hellaswag --split validation` |
| TruthfulQA | `truthful_qa` | `python scripts/prepare_data.py --hf-path truthful_qa --hf-subset multiple_choice --split validation` |

Any HuggingFace MCQ dataset can also be loaded via `name: huggingface` with `hf_path` specified. See `config/experiment_template.yaml` for the full schema.

---

## Config Reference

A complete annotated example:

```yaml
experiment:
  name: my_experiment           # free-form label; used in log output and paths

models:                         # one or more; every model runs every benchmark
  - backend: huggingface        # "huggingface" | "api" | "dummy"
    model_name_or_path: Qwen/Qwen2.5-7B-Instruct  # HF hub ID or local path
    device: cuda                # "cuda" | "cpu" | "auto" (multi-GPU)
    generation_kwargs:
      max_new_tokens: 512
      temperature: 0.0
      do_sample: false

  - backend: api                # API-backed model; no GPU required
    provider: openai            # required for api; key in CLIENT_REGISTRY
    model_name_or_path: gpt-4.1-mini
    generation_kwargs:
      max_new_tokens: 512
      temperature: 0.0

benchmarks:                     # a non-empty list (one entry per benchmark)
  - name: mmlu                  # "mmlu" | "arc_challenge" | "toy" | "huggingface"
    split: test                 # split name (benchmark-specific)
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
  # logprob-capable (huggingface/dummy). With the api model present, including
  # it would fail config validation — so it's commented out here:
  # - name: pride
  #   requires_logprobs: true   # schema-validated; rejects non-logprob backends

metrics:
  - accuracy
  - mad
  # - my_package.metrics.my_metric:MyMetricMetric  # external metric

run:
  seed: 42                      # global seed (subsampling and generation)
  resume: true                  # resume from checkpoint if one exists
  dry_run: false                # print plan and exit without running
  checkpoint_every_n: 50        # save progress to disk every N questions
  prompt_version: "v1"          # subdirectory under prompts/
  concurrency_limit: 10         # API client guard; orchestration is synchronous in v0.1
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

The retry loop, backoff, semaphore, and request/response validation are all handled by `BaseClient`. You only implement the single raw API call. In v0.1, experiment orchestration still iterates benchmark/model/method jobs synchronously; the client concurrency guard is infrastructure for provider calls, not parallel job scheduling.

### Add a new backend (4 steps)

1. Copy `src/choicebench/backends/templates/base_backend_template.py` to `src/choicebench/backends/my_backend.py`
2. Implement `generate()` (required) and optionally `score_options()` + `supports_logprobs = True`
3. Add a branch to `build_backend()` in `scripts/run_experiment.py`
4. Add the backend key to `_VALID_BACKENDS` in `src/choicebench/config/schema.py`; if it implements `score_options()`, also add it to `_LOGPROB_CAPABLE_BACKENDS`

### External registration via `"module.path:ClassName"` syntax

Any method or metric can be loaded from an external package — just install it in the same virtualenv and use the full import path in YAML. No framework files need to be touched.

---

## Architecture

```
config/              — YAML experiment configs
scripts/             — entry points (run_experiment.py, evaluate_run.py, prepare_data.py, prepare_toy_data.py)
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
| `cyclic_permutation` | Runs one cyclic permutation per available option, takes majority vote | N options, normally 4 |
| `two_stage` | Stage 1: free-form answer; Stage 2: map to option letter | 2, or 3 if fallback is enabled |
| `pride` | PriDe Eq. 8 logprob debiasing. Note: YAML-driven runs use a uniform prior in v0.1 unless a preflight calibration block is configured. Without preflight, this is logprob argmax only — not calibration-fitted debiasing from Zheng et al., ICLR 2024. | 1 score_options call per eval row; +4K calibration calls per run if K calibration rows are supplied |

### Metrics

| Name | Description |
|---|---|
| `accuracy` | Fraction of questions answered correctly |
| `mad` | Mean absolute deviation between the model's letter-selection distribution and the gold answer distribution — measures answer-order bias |

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
| `huggingface` | User-specified HuggingFace dataset | Requires a normalizer branch in `scripts/prepare_data.py` |

### Option Schema

v0.1 uses a legacy normalized CSV schema with `choice_a` through `choice_d`
columns. Generate-based methods build an option map from non-empty trailing
choice columns, so they can tolerate rows with fewer than four options. PriDe
is currently stricter: it requires exactly four valid A-D options. A full
list-based variable-option schema, mixed-option-count PriDe calibration, and
mixed-N subject metrics are v0.2 work.

---

## SLURM / HPC

Two scripts in `scripts/slurm/`:

- **`submit_job.sh`** — single job, one config, one GPU node. Suitable for all API runs and local models up to ~30B parameters.
- **`submit_array.sh`** — job array, one task per config. Edit the `CONFIGS` array in the file, then submit with an explicit array range such as `sbatch --array=1-3 scripts/slurm/submit_array.sh`.

Both scripts compute `REPO_ROOT` from their location. Set `VENV_DIR` and adjust the `#SBATCH` partition/GPU resources to match your cluster layout.

```bash
CONFIG=config/my_experiment.yaml sbatch scripts/slurm/submit_job.sh
```

---

## Roadmap

Planned v0.2 work:

- **Variable-option schema** — first-class `choices` / `correct_index` columns replacing the legacy `choice_a`–`choice_d` layout, enabling benchmarks with 2, 3, or 5+ options per row.
- **Mixed-option PriDe and MAD** — calibration and subject-level metrics that handle questions with different option counts in the same run.
- **Config-driven PriDe calibration splits** — specify calibration rows directly in YAML rather than passing them via Python construction.
- **Parallel orchestration** — concurrent execution across independent model/provider jobs (v0.1 iterates synchronously).
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
  version   = {0.1.0},
  license   = {MIT}
}
```
