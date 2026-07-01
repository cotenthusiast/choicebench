# Contributing to choicebench

## Dev environment setup

```bash
git clone https://github.com/cotenthusiast/choicebench && cd choicebench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # installs the package + test dependencies
cp .env.example .env             # then add your API keys
```

## Running the test suite

```bash
pytest tests/
```

The toy experiment provides a fast end-to-end sanity check without any API keys or GPU:

```bash
python scripts/run_experiment.py --config config/toy_experiment.yaml --yes
```

## Naming conventions

### Methods

| What | Convention | Example |
|---|---|---|
| File name | `snake_case.py` in `methods/library/` | `my_debiasing.py` |
| Class name | `CamelCaseRunner` | `MyDebiasingRunner` |
| YAML key | same as file name, without `.py` | `my_debiasing` |

### Metrics

| What | Convention | Example |
|---|---|---|
| File name | `snake_case.py` in `metrics/` | `positional_bias.py` |
| Class name | `CamelCaseMetric` | `PositionalBiasMetric` |
| YAML key | must match the `name` property | `positional_bias` |

### Clients

| What | Convention | Example |
|---|---|---|
| File name | `<provider>_client.py` in `clients/` | `anthropic_client.py` |
| Class name | `ProviderNameClient` | `AnthropicClient` |
| Registry key | lowercase provider string | `"anthropic"` |

## Registration: built-in vs external

**Built-in** (lives in this repo, will be used by others):
- Methods: add to `METHOD_REGISTRY` in `src/choicebench/registry.py` and re-export from `methods/library/__init__.py`
- Metrics: add to `BUILTIN_METRICS` in `src/choicebench/metrics/__init__.py`
- Clients: add to `CLIENT_REGISTRY` in `src/choicebench/registry.py`

**External** (lives in your own package, project-specific):
- Use `"module.path:ClassName"` syntax in the YAML `methods:` or `metrics:` list
- No framework files need to be modified

## Code style

- All public functions and classes must have type hints on all parameters and return values
- Avoid new hardcoded file paths, model names, or magic numbers in `src/` — put them in `config/` or pass them as arguments
- No print statements in library code — use `logging.getLogger(__name__)`
- Tests for new methods go in `tests/runners/`; tests for new metrics go in `tests/metrics/`

## Adding a new benchmark (4 steps)

1. Create `src/choicebench/benchmarks/my_benchmark.py` with a `build_normalized_dataframe()` function. Output schema must match the canonical normalized columns: `question_id`, `subject`, `question_text`, `choice_a`, `choice_b`, `choice_c`, `choice_d`, `correct_option`, `correct_answer_text`.

2. Decorate the normalizer with `@benchmark(...)`:
   ```python
   from choicebench.benchmarks.registry import benchmark

   @benchmark(
       name="my_benchmark",
       hf_path="org/my-benchmark",
       hf_subset="default",
       default_split="test",
   )
   def build_normalized_dataframe(df):
       ...
   ```

3. Import the module in `src/choicebench/benchmarks/__init__.py` so the decorator runs:
   ```python
   from choicebench.benchmarks import my_benchmark
   ```

4. Run `python scripts/prepare_data.py --hf-path org/my-benchmark --hf-subset default`. Registered HuggingFace paths default to the registry name for the normalized CSV stem, so `name: my_benchmark` in YAML will load `data/processed/my_benchmark_normalized.csv`.

Every HuggingFace dataset needs a registered normalizer first. If you use `name: huggingface` in YAML for an unregistered dataset, `prepare_data.py` cannot normalize it.

## Submitting a PR

1. Fork the repo and create a feature branch from the default branch
2. Run `pytest tests/` — all tests must pass
3. Run the toy experiment end-to-end to verify the pipeline is intact:
   ```bash
   python scripts/run_experiment.py --config config/toy_experiment.yaml --run-id toy_experiment --yes
   ```
4. Open a PR against the default branch with a description of what changed and why
5. For new methods or metrics, include a brief description of the algorithm and a reference if applicable
