# ChoiceBench Conformance Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tests_conformance/`, an independent, offline test suite that encodes the frozen ChoiceBench paper protocol as executable assertions against real production interfaces (config loader, runner classes, orchestration scripts) driven by fake/spy backends — not against existing unit tests or implementation details.

**Architecture:** Every test drives the real production entry point (`load_config()`, a `METHOD_REGISTRY` runner, or a `scripts/paper/*.py` `run()` function) with a hand-built fake/spy `BaseBackend` subclass or `DummyBackend` subclass that records every call. Assertions come from `tests_conformance/FROZEN_SPEC.md` (the verbatim frozen spec), never from reading `tests/` first. One shared `conftest.py` + `fakes/` package is reused by every test file. Failures are RECORDED, never used to justify editing production code or the test's own expected value — a red test is a valid, successful outcome of this plan.

**Tech Stack:** pytest, pandas, the repo's own `src/choicebench` package (installed editable), no network/GPU/HF downloads.

**Spec:** `tests_conformance/FROZEN_SPEC.md` (verbatim copy of the frozen conformance spec this plan implements — every task below cites its section letters, e.g. "spec §D").

## Global Constraints

- Do not modify any file under `src/choicebench/`, `config/`, `scripts/`, or `prompts/` at any point in this plan.
- Do not make real network calls, load real HF weights, or use Kelvin2/GPUs — every backend is a fake/spy.
- `run.seed` / experiment seed is always `42` in frozen paper configs; `provider_seed` is always `None` in frozen paper configs (spec §Seeds).
- Frozen generation settings for every text-generating paper condition: `temperature == 0.0`, `generation_kwargs.max_new_tokens == 1024` (spec §Generation). Note the **library default** is `DEFAULT_MAX_NEW_TOKENS = 512` (`src/choicebench/config/schema.py:23`) — the frozen `1024` value only holds because every `config/paper/*.yaml` sets it explicitly; a test must read the actual YAML value, never assume the default.
- Tests never write into `runs/`, `cache/`, `data/`, or any tracked directory — every test uses `tmp_path` / `tmp_path_factory`.
- Tests never read `tests/` (the existing suite) for *expected values* — only, in a later task, for fixture/helper *syntax* conventions (spec Phase 4).
- If a test would only pass by weakening an assertion below what the frozen spec states, do not weaken it — mark the test `xfail(strict=True, reason="spec §X: <what's wrong>")` instead so the suite stays green-means-green while still recording the disagreement (this is a test-authoring device, not "bending the test to the code" — the assertion itself is never changed).

## Review Focus

- **A pandas `NaN` free-text value is not `None`.** `derive_from_free_text.py:38` (`if free_text is not None and str(free_text).strip() != ""`) treats a `NaN` `free_text_response` as truthy non-empty text and stringifies it to the literal `"nan"`, which then gets embedded/matched as if it were a real answer — this is exactly the failure mode spec §G4/§Q5 warn about. Task 6 must construct a genuine `pd.NA`/`float('nan')` row (not a Python `None`) to catch this; a `None`-only test would pass by accident and miss the bug.
- **`ModelRequest`'s own class default for `seed` is `42`** (`src/choicebench/clients/types.py:109`, sourced from `config/providers.py:24`'s `SEED = 42`), not `None`. A naive test that only checks "`provider_seed` field on `ModelConfig` is `None`" would miss a regression where some call site stopped passing `seed=` explicitly and silently fell back to the class default of `42` — which is exactly the experiment-seed-leak bug spec §A4 is written to catch. Task 1's A4 test must inspect the actual `ModelRequest`/call args a spy backend received, not just the config object.
- **PriDe's eligibility-before-sampling order is easy to get backwards** (`pride.py:225` filters `self._calibration_questions` to modal-k-eligible rows *before* `_pick_calibration_rows` samples `calibration_n` from them) — spec §J5 explicitly demands a fixture where sample-first would yield `<15` valid rows, to distinguish "filters first" from "filters after, got lucky." Task 8 must build that adversarial fixture, not just a fixture where both orderings happen to agree.
- **`client_extra_identity()` is duck-typed on attribute value type, not attribute presence** (`infra/cache.py:153-158`, explicit anti-`Mock`-autovivification comment) — a spy/fake client used elsewhere in the suite must not accidentally expose truthy `_upstream_provider`/`_allow_fallbacks` attributes it doesn't intend to, or Task 11's cache-identity tests will silently test the wrong thing. Task 11 must use the *real* `OpenRouterClient`/provider client classes (constructed with fake HTTP underneath, not a bare `Mock`), never a `unittest.mock.Mock` stand-in for the client itself.
- **`run_stochasticity_repeats.py`'s obs0 identity check compares strings/floats loaded from a CSV**, not live Python objects (`_load_canonical_obs0_rows`, `_OBS0_IDENTITY_COLUMNS`/`_OBS0_NUMERIC_IDENTITY_COLUMNS`) — Task 10's wrong-model/wrong-provider/wrong-benchmark hard-failure tests (§L7-L11) must round-trip a real CSV file (write then `pd.read_csv`), not pass an in-memory DataFrame directly, or the test will exercise a different code path (in-memory dtypes) than production ever does.

---

## Shared fixtures/fakes design (built in Task 1, consumed by every later task)

### `tests_conformance/fakes/spy_backend.py`

A single `SpyBackend(BaseBackend)` used everywhere a fake *generate*-only target-model backend is needed:

```python
# tests_conformance/fakes/spy_backend.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from choicebench.backends.base import BaseBackend


@dataclass
class RecordedCall:
    call_index: int
    prompt: str
    call_id: str  # "CALL_001", "CALL_002", ... — assigned in issue order


class SpyBackend(BaseBackend):
    """Fake generate-only backend that records every prompt it receives and
    returns a scripted or rule-based response per call.

    Args:
        responder: given the 1-based call index and the prompt, returns the
            raw_text to hand back. Defaults to a fixed marker text
            ("CALL_001", "CALL_002", ...) so every response is uniquely
            traceable to the exact call that produced it.
        fail_on: a set of 1-based call indices that should raise instead of
            returning text, to simulate a transport failure at that call.
    """

    def __init__(
        self,
        responder: Callable[[int, str], str] | None = None,
        fail_on: frozenset[int] = frozenset(),
        provider_name: str = "spy",
        model_name_value: str = "spy-model",
    ) -> None:
        self._responder = responder or (lambda i, prompt: f"CALL_{i:03d}")
        self._fail_on = fail_on
        self._provider_name = provider_name
        self._model_name_value = model_name_value
        self.calls: list[RecordedCall] = []

    @property
    def model_name(self) -> str:
        return self._model_name_value

    @property
    def provider(self) -> str:
        return self._provider_name

    def generate(self, prompt: str, **kwargs) -> str:
        i = len(self.calls) + 1
        self.calls.append(RecordedCall(call_index=i, prompt=prompt, call_id=f"CALL_{i:03d}"))
        if i in self._fail_on:
            raise RuntimeError(f"SpyBackend: simulated transport failure on call {i}")
        return self._responder(i, prompt)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def prompts(self) -> list[str]:
        return [c.prompt for c in self.calls]
```

Design notes for implementers:
- `SpyBackend` is sync-only (`is_async_capable()` defaults to `False` from `BaseBackend`) so `ExperimentRunner._call_backend_generate` (`methods/base.py:180-198`) routes every call through the plain `self.backend.generate(prompt)` branch — one Python-level call per model call, trivially countable via `len(spy.calls)`.
- A rule-based `responder` (e.g. "return the option letter whose *text* is `'Paris'`, wherever it currently sits") is what makes spec §E2/§E3's "select the same semantic option under every rotation" vs. "select a fixed displayed letter under every rotation" cases possible — the responder receives the actual rendered `prompt` string, so it can `re.search` the prompt for a known option text and answer accordingly.
- For logprob-scoring tests (PriDe, §J), reuse `choicebench.backends.dummy_backend.DummyBackend` directly (already `supports_logprobs=True`, already used in production for this exact purpose per its own docstring) rather than re-implementing `score_options()` — but wrap it in a thin spy subclass that also records `(prompt, options)` per `score_options()` call, since `DummyBackend` itself does not record calls.

### `tests_conformance/fixtures/diagnostic_benchmark.py`

One frozen, hand-built DataFrame-producing helper implementing spec §C, reused by nearly every task:

```python
# tests_conformance/fixtures/diagnostic_benchmark.py
"""The one synthetic benchmark fixture reused across tests_conformance/.

Every row's canonical (unrotated) option order and gold source_index are
fixed and documented here — this is the ground truth every test compares
derived/observed values against, never re-derived from production code.
"""
from __future__ import annotations

import pandas as pd

# question_id -> (question_text, subject, [option_text, ...] in canonical
# source order, gold_source_index)
_ROWS: dict[str, tuple[str, str, list[str], int]] = {
    "diag_4a": (
        "What is the capital of France?", "geography",
        ["Paris", "Berlin", "Madrid", "Rome"], 0,
    ),
    "diag_4b": (
        "What is 2 + 2?", "math",
        ["3", "4", "5", "None"], 1,
    ),
    "diag_3a": (
        "Which is a primary color?", "art",
        ["Green", "Red", "Orange"], 1,
    ),
    "diag_5a": (
        "Which planet is closest to the sun?", "science",
        ["Venus", "Earth", "Mercury", "Mars", "Jupiter"], 2,
    ),
    # deliberately similar/repeated-looking wording (spec §C)
    "diag_4c": (
        "Which country is largest by area?", "geography",
        ["Russia", "Canada", "Russia (Federation)", "China"], 0,
    ),
}

_LETTERS = "ABCDEFGHIJ"


def diagnostic_rows() -> pd.DataFrame:
    """Return the diagnostic benchmark as a normalized questions DataFrame
    matching what choicebench.pipeline.options.build_option_map()/
    build_choices() expect (canonical option order == source order)."""
    records = []
    for qid, (question_text, subject, options, gold_idx) in _ROWS.items():
        letters = list(_LETTERS[: len(options)])
        record = {
            "question_id": qid,
            "question_text": question_text,
            "subject": subject,
            "n_choices": len(options),
            "correct_index": gold_idx,
            "correct_option": letters[gold_idx],
        }
        for letter, text in zip(letters, options):
            record[f"option_{letter.lower()}"] = text
        records.append(record)
    return pd.DataFrame.from_records(records)


def canonical_options_for(question_id: str) -> list[str]:
    return list(_ROWS[question_id][2])


def gold_source_index_for(question_id: str) -> int:
    return _ROWS[question_id][3]
```

Implementer note: **before finalizing this file**, read `src/choicebench/pipeline/options.py` (`build_option_map`, `build_choices`, `build_label_to_source_index`, `correct_option_for_row`) to confirm the exact column-name contract a "normalized question record" must satisfy (e.g. whether it's `option_a`/`option_b`/... columns, or a `choices_json` column) — the sketch above is the most likely shape based on `methods/base.py`'s usage (`question_row["correct_option"]`, `question_row["subject"]`, `question_row["question_text"]`) but must be verified against `pipeline/options.py` directly before writing the real file, and adjusted to match exactly.

### `tests_conformance/conftest.py`

- `paper_config_paths()` fixture: `sorted(Path("config/paper").glob("*.yaml"))` — the 14 files enumerated in Task 2.
- `tmp_runs_dir(tmp_path)` fixture: yields `tmp_path / "runs"`, so nothing touches the real `runs/` directory.
- `diagnostic_df()` fixture: thin wrapper around `fixtures/diagnostic_benchmark.diagnostic_rows()`.
- Adds `sys.path`/rootdir handling only if needed — prefer relying on the repo's existing `pyproject.toml`/`setup.cfg` editable install (confirm via `pip show choicebench` or `python -c "import choicebench"` before assuming path hacks are needed).

---

## Task 1: Shared infrastructure — `conftest.py`, `fakes/`, `fixtures/`, `README.md`

**Files:**
- Create: `tests_conformance/__init__.py` (empty, so pytest can import `tests_conformance.fixtures.*`/`tests_conformance.fakes.*` as a package)
- Create: `tests_conformance/conftest.py`
- Create: `tests_conformance/fakes/__init__.py`
- Create: `tests_conformance/fakes/spy_backend.py`
- Create: `tests_conformance/fixtures/__init__.py`
- Create: `tests_conformance/fixtures/diagnostic_benchmark.py`
- Create: `tests_conformance/README.md`
- (Already created this session): `tests_conformance/FROZEN_SPEC.md`

**Interfaces:**
- Consumes: `choicebench.backends.base.BaseBackend`, `choicebench.backends.dummy_backend.DummyBackend`, `choicebench.pipeline.options.*` (verify exact names first).
- Produces: `SpyBackend`, `RecordedCall`, `diagnostic_rows()`, `canonical_options_for()`, `gold_source_index_for()`, pytest fixtures `paper_config_paths`, `tmp_runs_dir`, `diagnostic_df` — every later task imports these.

- [ ] **Step 1: Read `src/choicebench/pipeline/options.py` in full** to confirm the exact normalized-question-record contract, and adjust `diagnostic_benchmark.py`'s sketch above to match exactly (column names, whether options live in `option_a..` columns or a `choices_json` blob, what `build_label_to_source_index` needs).

- [ ] **Step 2: Write `fakes/spy_backend.py`** using the design above (adjust only if Step 1's findings require it).

- [ ] **Step 3: Write `fixtures/diagnostic_benchmark.py`** per the confirmed contract from Step 1.

- [ ] **Step 4: Write `conftest.py`** with the fixtures listed above.

- [ ] **Step 5: Smoke-test the fixture against the real pipeline.** Write and run a throwaway script (not a committed test) that does:
  ```python
  from choicebench.pipeline.options import build_option_map, build_label_to_source_index
  from tests_conformance.fixtures.diagnostic_benchmark import diagnostic_rows
  df = diagnostic_rows()
  row = df.iloc[0].to_dict()
  print(build_option_map(row))
  print(build_label_to_source_index(row))
  ```
  Run: `cd /home/cotenthusiast/Projects/choicebench && python -c "..."` (inline, or a scratch file in the scratchpad dir). Expected: no exception, and `build_option_map(row)` reproduces `{"A": "Paris", "B": "Berlin", ...}` for `diag_4a`.

- [ ] **Step 6: Write `README.md`** containing (verbatim, per spec):
  > This suite is an executable specification of the ChoiceBench paper protocol. Expected scientific behavior comes from the frozen experiment specification, not from implementation details.

  Plus: a table mapping every spec section letter (A–S) to its test file, a one-line description of `SpyBackend`/`diagnostic_rows()`, and an explicit statement that `FROZEN_SPEC.md` is the authoritative source and must never be edited to make a test pass.

- [ ] **Step 7: Run `pytest tests_conformance/ --collect-only`** (no tests exist yet, but this validates the package imports cleanly). Expected: exits 0, "no tests collected" or similar, zero import errors.

- [ ] **Step 8: Commit.**
  ```bash
  git add tests_conformance/
  git commit -m "test(conformance): add shared spy backend, diagnostic fixture, and suite scaffold"
  ```

---

## Task 2: `test_production_configs.py` — spec §A (A1–A8)

**Files:**
- Create: `tests_conformance/test_production_configs.py`

**Interfaces:**
- Consumes: `choicebench.config.schema.load_config(path: str) -> ExperimentConfig`, `ExperimentConfig.models: list[ModelConfig]`, `ModelConfig.{backend, provider, model_name_or_path, generation_kwargs.{temperature,max_new_tokens}, provider_seed, upstream_provider, allow_fallbacks, device, torch_dtype}`, `choicebench.cli.run_experiment.build_backend(model_config, run_id, run_seed=..., execution_mode=..., model_identity=...)`, `choicebench.registry.CLIENT_REGISTRY`.

- [ ] **Step 1 (A1): parametrized parse test.**
  ```python
  import glob
  from pathlib import Path
  import pytest
  from choicebench.config.schema import load_config

  PAPER_CONFIGS = sorted(Path("config/paper").glob("*.yaml"))

  def test_a1_all_paper_configs_have_at_least_one_file():
      assert len(PAPER_CONFIGS) >= 1, "config/paper/*.yaml is empty — nothing to conform-test"

  @pytest.mark.parametrize("path", PAPER_CONFIGS, ids=lambda p: p.name)
  def test_a1_paper_config_parses(path):
      config = load_config(str(path))
      assert config.models, f"{path} has no models"
      assert config.benchmarks, f"{path} has no benchmarks"
  ```

- [ ] **Step 2 (A2): temperature/max_tokens for every text-generating model.** For each `(path, model)` pair where `model.backend != "dummy"` and the config's methods are not exclusively `pride`/logprob-only (a PriDe-only config's models still generate during calibration rollouts, so include them too — PriDe's `_cyclic_rollout_prob_matrix` scores, not generates, but the frozen spec's "text-generating target-model paths" line covers every model entry in these configs since none of the 6 conditions are PriDe-exclusive-non-generating in the paper matrix). Assert `model.generation_kwargs.temperature == 0.0` and `model.generation_kwargs.max_new_tokens == 1024`.

- [ ] **Step 3 (A3): every model has `provider_seed is None`.** Iterate all `(path, model)` and assert `model.provider_seed is None`.

- [ ] **Step 4 (A4): real backend construction proves no seed leak.** Build a `SpyBackend`-free real check: use `build_backend` from `mmlu_core_methods.yaml` (OpenRouter Llama = `models[2]`, Together Qwen = `models[3]`) with a `run_seed=42`. Since `build_backend` for an `"api"` backend returns a real `APIBackend` wrapping a real `OpenRouterClient`/`TogetherAIClient` (which will try to read an API key via `python-dotenv`/`os.getenv` but does not make a network call at construction time — confirm this by reading `clients/openrouter_client.py`'s `__init__` before writing this step), call `backend._make_request("dummy prompt")` (private but this is exactly the seam the spec asks to inspect — `APIBackend._make_request`, `backends/api_backend.py:53-61`) and assert `request.seed is None` (not `42`, per Review Focus above) and `request.provider == "openrouter"` / `"together"` respectively.
  - If `OpenRouterClient()`/`TogetherAIClient()` construction raises for a missing API key in this environment, catch that specifically and instead construct the client with `os.environ` monkeypatched to a dummy key for the duration of the test (`monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")`) — never skip this test; the whole point is exercising the real construction path.

- [ ] **Step 5 (A5): generic provider_seed still propagates.** Write a synthetic YAML (via `tmp_path`) with one `api`/`dummy`... — no, `provider_seed` is API-backend-only per `ModelConfig` docstring, so use a `backend: api, provider: openai` entry with `provider_seed: 7`. `load_config()` it, then `build_backend(model_config, run_id="t", run_seed=42, model_identity="x")` (monkeypatching the API key env var as needed) and assert the resulting `APIBackend._seed == 7` (or via `_make_request(...).seed == 7`).

- [ ] **Step 6 (A6): nominal six-model-condition coverage.** Read `mmlu_core_methods.yaml`, `mmlu_core_methods_cyclic_batch.yaml`, `mmlu_pride.yaml`, `mmlu_independent_hypothesis.yaml` and their `arc_*` counterparts; assert the *union* of `(provider, model_name_or_path)` / `(backend, model_name_or_path)` pairs across each benchmark's full config set equals exactly the six frozen conditions from spec §Six model conditions (openai gpt-4.1-mini-2025-04-14, anthropic claude-haiku-4-5-20251001, openrouter+upstream_provider=deepinfra meta-llama/llama-3.1-8b-instruct OR direct deepinfra Turbo Llama depending on config, together Qwen/Qwen2.5-7B-Instruct-Turbo, huggingface RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic, huggingface RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic). Read `mmlu_core_methods_cyclic_batch.yaml` first to get the exact direct-DeepInfra provider/model string before writing the assertion (do not guess it).

- [ ] **Step 7 (A7): no Gemini in paper configs.** For every `config/paper/*.yaml`, assert no model's `provider` is `"gemini"` and no `model_name_or_path` contains `"gemini"` (case-insensitive).

- [ ] **Step 8 (A8): local configs use HF + exact RedHatAI IDs.** For every config with a `backend: huggingface` model, assert `model_name_or_path in {"RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic", "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic"}`.

- [ ] **Step 9: run `pytest tests_conformance/test_production_configs.py -v`**, record pass/fail per spec item in the running scratch notes (not committed) for the final report.

- [ ] **Step 10: commit.**
  ```bash
  git add tests_conformance/test_production_configs.py
  git commit -m "test(conformance): add spec sec A config-to-runtime propagation tests"
  ```

---

## Task 3: `test_frozen_manifests.py` — spec §B (B1–B5)

**Files:**
- Create: `tests_conformance/test_frozen_manifests.py`

**Interfaces:**
- Consumes: `data/manifests/mmlu_eval_v2.csv` (1141 lines incl. header, confirmed), `data/manifests/arc_challenge_eval_v2.csv` (1173 lines incl. header, confirmed). Column names not yet confirmed — read the first few lines of each CSV before writing assertions (`question_id`, `subject` for MMLU; option-count-derivable columns for ARC — likely needs joining against the normalized ARC source data, not just the 1/2-column manifest, to get `n_choices` per row; find where the "normalized ARC data" lives, e.g. `data/prepared/` or similar, via `ls data/`).

- [ ] **Step 1: inspect manifest files.** `head -3 data/manifests/mmlu_eval_v2.csv data/manifests/arc_challenge_eval_v2.csv` and `ls data/` to find the normalized/prepared ARC dataset with per-row option counts (needed for B4). Adjust the plan below if the manifest itself already carries `n_choices`.

- [ ] **Step 2 (B1): MMLU manifest invariants.**
  ```python
  import pandas as pd

  def test_b1_mmlu_manifest_frozen_shape():
      df = pd.read_csv("data/manifests/mmlu_eval_v2.csv")
      assert df["question_id"].is_unique
      assert len(df) == 1140
      assert df["subject"].nunique() == 57
      counts = df["subject"].value_counts()
      assert (counts == 20).all(), counts[counts != 20]
  ```
  (Adjust column name if Step 1 found a different one.)

- [ ] **Step 3 (B2): regeneration-match, marked conditionally.** Attempt to reproduce the MMLU manifest via the real benchmark-loading path (`choicebench.benchmarks.mmlu` / `choicebench.benchmarks.registry`) against `hf_path: cais/mmlu` with `seed=42` and `SamplingConfig(strategy="per_group", group_field="subject", n_per_group=20)`. If this requires a live HF dataset download (it will, per `hf_path`), wrap in:
  ```python
  @pytest.mark.skip(reason="spec §B2: requires downloading cais/mmlu from HF Hub — "
                            "not available in a clean offline checkout; B1's committed-"
                            "manifest invariants are the offline-mandatory subset. "
                            "Needs LIVE/online validation — see README's live-validation list.")
  def test_b2_mmlu_manifest_regeneration_matches_seed_42():
      ...
  ```
  Do not fabricate a fake regeneration — per spec, "report source-regeneration as runtime/external-data verification" when the source is unavailable offline.

- [ ] **Step 4 (B3): ARC manifest invariant.**
  ```python
  def test_b3_arc_manifest_frozen_count():
      df = pd.read_csv("data/manifests/arc_challenge_eval_v2.csv")
      assert df["question_id"].is_unique
      assert len(df) == 1172
  ```

- [ ] **Step 5 (B4): ARC option-count distribution.** Using whatever normalized ARC source Step 1 found (or, if none is committed/available offline, mark `skip` with the same "requires external data" reasoning as B2 — do not fabricate), assert exactly 1165 four-option, 4 three-option, 3 five-option rows among the 1172 evaluation IDs.

- [ ] **Step 6 (B5): calibration/evaluation disjointness — synthetic.** Since a *full* real calibration-pool fixture may not be available offline either, build this as: load the MMLU eval manifest's `question_id` set, and (if available) MMLU's validation-split calibration question IDs actually used for PriDe (check whether these are committed anywhere, e.g. under `data/`); if not available, downgrade to a *synthetic* fixture using `diagnostic_rows()` split into two disjoint ID sets and assert `set(eval_ids) & set(calib_ids) == set()` — this still exercises the intended invariant-checking logic even though it isn't validating the real frozen files. Document which case applies once Step 1's `ls data/` results are known.

- [ ] **Step 7: run and record.**
  `pytest tests_conformance/test_frozen_manifests.py -v`

- [ ] **Step 8: commit.**

---

## Task 4: `test_call_graphs.py` — spec §D (D1–D6) + §F (F1–F7)

**Files:**
- Create: `tests_conformance/test_call_graphs.py`

**Interfaces:**
- Consumes: `choicebench.methods.library.permutation.PermutationRunner` (constructor per `methods/base.py:49-63` — `backend, method_name, split_name, prompt_version, prompts_dir, run_id, temperature, max_tokens, seed, benchmark_name`), `PermutationRunner.run_one(question_row: dict, sample_index: int) -> dict` (returns row with `per_rotation_choices_json`), `choicebench.analysis.derive_baseline_from_cyclic.derive_baseline_from_cyclic(source_df, method_name="direct_mcq") -> pd.DataFrame`, `choicebench.methods.library.two_stage.TwoStageRunner.run_one(...)` (1 free-text + 1 matching call) and `.run_stage2_rotations(question_row, free_text_answer, sample_index) -> dict` (N-1 additional matching calls for the non-rotation-0 rotations, or see Step note below), `scripts.paper.run_two_stage_rotations` for the real orchestration shape.
- `prompts_dir=Path("prompts")` (real repo prompts, `prompt_version="v1"`) — these are static template files, safe to use for real (no network).

- [ ] **Step 1: confirm the two-stage "rotation experiment" orchestration shape.** Spec F1 says "Stage 1 calls = 1, Stage 2 calls = 4" for one N=4 question. `TwoStageRunner.run_one()` makes 1 Stage-1 + 1 Stage-2 call; `run_stage2_rotations()` makes N Stage-2 calls reusing a given `free_text_answer` (it does NOT skip rotation 0 — it reruns *every* rotation, since `run_two_stage_rotations.py`'s `run()` calls `runner.run_stage2_rotations(source_row, free_text, sample_index)` once per already-completed `two_stage` row, covering all N rotations independently of `run_one`'s own single Stage-2 call). Read `scripts/paper/run_two_stage_rotations.py` again (already read this session) to confirm: **the real production flip-rate workflow is `run_one()` once (1 Stage1 + 1 Stage2, canonical order) to get accuracy, PLUS a fully separate `run_stage2_rotations()` call (N Stage2 calls, all N rotations, reusing that same free_text) to get the flip-rate trace** — i.e. spec F1's "Stage 2 calls = 4" is `run_stage2_rotations()` alone (its own 4 calls), not `run_one`'s 1 plus 3 more. Write the test against `run_stage2_rotations()` in isolation for F1/F5/F6, and write a *separate* assertion that `run_one()` alone is 1+1, to avoid conflating the two entry points.

- [ ] **Step 2 (D1/D4/D5): cyclic call count for N=3/4/5.**
  ```python
  import pytest
  from pathlib import Path
  from choicebench.methods.library.permutation import PermutationRunner
  from tests_conformance.fakes.spy_backend import SpyBackend
  from tests_conformance.fixtures.diagnostic_benchmark import diagnostic_rows

  @pytest.mark.parametrize("question_id,expected_n", [
      ("diag_4a", 4), ("diag_3a", 3), ("diag_5a", 5),
  ])
  def test_d1_d4_d5_cyclic_call_count(question_id, expected_n, tmp_path):
      row = diagnostic_rows().set_index("question_id").loc[question_id].to_dict()
      row["question_id"] = question_id
      spy = SpyBackend()
      runner = PermutationRunner(
          backend=spy, method_name="cyclic_permutation", split_name="test",
          prompt_version="v1", prompts_dir=Path("prompts"), run_id="t",
          seed=42, benchmark_name="diag",
      )
      result = runner.run_one(row, sample_index=0)
      assert spy.call_count == expected_n
      return result  # reused by D2/D3 below — restructure as a fixture if pytest disallows returning
  ```
  (Restructure the parametrized case into a fixture-returning helper, since pytest test functions shouldn't `return` a value — factor the row-building + run into a shared `_run_cyclic(question_id, spy=None) -> (spy, result_row)` helper at module scope instead.)

- [ ] **Step 3 (D2/D3): baseline derivation adds zero calls and has verified lineage.**
  ```python
  import pandas as pd
  from choicebench.analysis.derive_baseline_from_cyclic import derive_baseline_from_cyclic

  def test_d2_d3_baseline_derivation_zero_calls_and_lineage():
      spy, cyclic_row = _run_cyclic("diag_4a")
      calls_before = spy.call_count
      source_df = pd.DataFrame([cyclic_row])
      baseline_df = derive_baseline_from_cyclic(source_df, method_name="direct_mcq")
      assert spy.call_count == calls_before  # D2: no new calls possible — derive_baseline_from_cyclic takes no backend at all
      baseline_row = baseline_df.iloc[0]
      # D3: baseline's raw_text/prompt IS rotation 0's own (responses[0]/prompts[0] in PermutationRunner.run_one)
      assert baseline_row["raw_text"] == cyclic_row["raw_text"]
      assert baseline_row["prompt"] == cyclic_row["prompt"]
      assert baseline_row["derived_from_run_id"] == cyclic_row["run_id"]
      assert baseline_row["derived_from_method_name"] == cyclic_row["method_name"]
      assert baseline_row["question_id"] == cyclic_row["question_id"]
  ```
  Note `derive_baseline_from_cyclic` takes `source_df` only (no backend parameter at all, by the module's own docstring) — D2's "assert call count remains exactly 4" is structurally guaranteed (there is no way to add a 5th call), so additionally assert the function's signature has no backend/client parameter via `inspect.signature`, making the "impossible by construction" property explicit rather than just implicit in not calling anything.

- [ ] **Step 4 (D6): malformed source data fails loudly.**
  ```python
  import pytest

  def test_d6_baseline_derivation_rejects_non_cyclic_source():
      bad_df = pd.DataFrame([{"question_id": "x", "raw_text": "foo"}])  # no per_rotation_choices_json
      with pytest.raises(ValueError):
          derive_baseline_from_cyclic(bad_df)
  ```

- [ ] **Step 5 (F1/F5/F6): two-stage rotation call counts.**
  ```python
  from choicebench.methods.library.two_stage import TwoStageRunner

  @pytest.mark.parametrize("question_id,expected_n", [
      ("diag_4a", 4), ("diag_3a", 3), ("diag_5a", 5),
  ])
  def test_f1_f5_f6_two_stage_rotation_call_counts(question_id, expected_n):
      row = _diag_row(question_id)
      spy = SpyBackend()
      runner = TwoStageRunner(
          backend=spy, method_name="two_stage", split_name="test",
          prompt_version="v1", prompts_dir=Path("prompts"), run_id="t",
          seed=42, benchmark_name="diag",
      )
      free_text_response = runner._call_backend_generate(
          # or better: run_one() once to get a real free_text_answer, see Step 6
      )
  ```
  Correct construction: call `runner.run_one(row, 0)` first (1 Stage-1 + 1 Stage-2 call = 2 calls on `spy`), extract `result["free_text_response"]`, **reset the spy** (`spy2 = SpyBackend()`, fresh instance, since `run_stage2_rotations` needs its OWN isolated call count per F1's "Stage 2 calls = 4"), then call `runner2 = TwoStageRunner(backend=spy2, ...)` and `runner2.run_stage2_rotations(row, free_text_response, 0)`; assert `spy2.call_count == expected_n` (this is "Stage 2 calls"), and separately assert the *first* spy's call count was exactly 2 (1 Stage-1 + 1 Stage-2, i.e. F1's "Stage 1 calls = 1" is verified by construction — `run_one` issues exactly one free-text call before any stage-2 call, confirmed by reading `TwoStageRunner.run_one`'s source order in `methods/library/two_stage.py:62-89`).

- [ ] **Step 6 (F2): Stage-1 prompt excludes options.**
  ```python
  def test_f2_stage1_prompt_has_no_options():
      row = _diag_row("diag_4a")
      spy = SpyBackend()
      runner = TwoStageRunner(backend=spy, method_name="two_stage", split_name="test",
                               prompt_version="v1", prompts_dir=Path("prompts"), run_id="t",
                               seed=42, benchmark_name="diag")
      runner.run_one(row, 0)
      stage1_prompt = spy.prompts[0]
      for option_text in canonical_options_for("diag_4a"):
          assert option_text not in stage1_prompt
  ```

- [ ] **Step 7 (F3/F4): lineage + rotation ordering for Stage-2.** Use a `responder` on the rotation spy that echoes back which displayed letter/text it saw (e.g. regex-extract the option block from the prompt), and assert the Nth prompt's *rendered option order* matches `build_rotations(canonical_options)[n]`'s own mapping (import `choicebench.pipeline.prompt_builder.build_rotations` directly and compare). For lineage (F3 "prove, don't infer from equal text"): assert every Stage-2 rotation prompt contains the *exact same* `free_text_response` substring passed in (not regenerated), i.e. `assert free_text_response in prompt for prompt in spy2.prompts`.

- [ ] **Step 8 (F7): reasoning_two_stage same call graph, different prompt.** Repeat Step 5's N=4 case with `prompt_version="v1_reasoning"` and `method_name="reasoning_two_stage"` (per `registry.py`, `reasoning_two_stage` maps to the same `TwoStageRunner` class — confirmed this session). Assert call counts are identical to the `v1`/`two_stage` case, and assert the Stage-1 prompt text differs from the `v1` Stage-1 prompt (proving `prompts_dir`/`prompt_version` actually selects a different template — read `prompts/v1_reasoning/` to confirm a `free_text` template exists there before writing this assertion).

- [ ] **Step 9: run and record**, then **Step 10: commit.**

---

## Task 5: `test_permutation_semantics.py` — spec §E (E1–E7)

**Files:**
- Create: `tests_conformance/test_permutation_semantics.py`

**Interfaces:**
- Consumes: `choicebench.pipeline.prompt_builder.build_rotations(options: dict[str, str]) -> list[Rotation]` (`Rotation.mapping`, `Rotation.slot_to_canonical`, per `methods/library/permutation.py` usage), `PermutationRunner` (as Task 4), `choicebench.metrics.order_sensitivity.OrderSensitivity` (for flip-rate cross-check against `per_rotation_choices_json`).

- [ ] **Step 1 (E1/E6): rotation enumeration + source_index stability.** For `diag_4a`, call `build_rotations(canonical_options)`; assert `len(rotations) == 4`; for each rotation, assert `rotation.mapping` is a permutation of the same `(letter, text)` pairs (same set of texts, different letter assignment for k>0); assert a fixed helper `build_label_to_source_index` applied to the *canonical* row is unaffected by which rotation is being displayed (call it once, independent of rotation — E6 "changing only display order must never mutate gold canonical source_index" is definitionally true since `build_label_to_source_index` takes the canonical row, not a rotation — assert this by checking the function's signature never takes a `Rotation`/rotation index argument at all, via `inspect.signature`).

- [ ] **Step 2 (E2): same-semantic-option responder ⇒ flip rate 0.** Build a `SpyBackend` responder that always answers with the displayed letter for the option whose *text* equals `"Paris"` (regex over the prompt to find which letter `"Paris"` is rendered at). Run `PermutationRunner.run_one` on `diag_4a`; assert every entry in `json.loads(result["per_rotation_choices_json"])` is the *same* canonical letter (`"A"`, since `Paris` is `diag_4a`'s option A / gold), and the voted `parsed_choice == "A"`. Then feed this `per_rotation_choices_json` through `OrderSensitivity().compute(...)` on a 1-row DataFrame with `correct_index=0` and assert `order_flip_rate == 0.0`.

- [ ] **Step 3 (E3): fixed-displayed-letter responder ⇒ flips.** Build a responder that always answers literally `"A"` regardless of prompt content (ignoring which text is at A). Run the same runner; since rotation k puts a different canonical option at displayed position A each time, the per-rotation canonical choices should NOT all be equal; assert `len(set(json.loads(result["per_rotation_choices_json"]))) > 1`, and cross-check via `OrderSensitivity` that `order_flip_rate == 1.0` for this single question.

- [ ] **Step 4 (E4/E5): repeat E2+E3 for `diag_3a` (N=3) and `diag_5a` (N=5).** Parametrize Steps 2–3 over `["diag_4a", "diag_3a", "diag_5a"]`.

- [ ] **Step 5 (E7): rotation processing order independence.** Manually construct the rotations list via `build_rotations`, then compute the majority vote via `choicebench.scoring.tiebreak.majority_vote_with_tiebreak` twice — once over `canonical_choices` in natural order, once over a shuffled copy of the *same* list (`random.Random(0).shuffle(...)` on a copy) — and assert both calls return the identical winning letter (the function is a pure `Counter`-based majority + `resolve_tie` over the *set* of tied ids, so order must not matter; confirmed by reading `tiebreak.py` this session — no ordering dependency exists in `Counter.most_common` for the top count itself, but tie-break resolution depends only on the *set* of tied ids, not their order, per `resolve_tie`'s own docstring "Order does not matter").

- [ ] **Step 6: run and record**, then **Step 7: commit.**

---

## Task 6: `test_semantic_matching.py` — spec §G (G1–G5)

**Files:**
- Create: `tests_conformance/test_semantic_matching.py`

**Interfaces:**
- Consumes: `choicebench.analysis.derive_from_free_text.derive_matched_results(source_df, method_name, embed_fn=None) -> pd.DataFrame`, `choicebench.scoring.text_matcher.EmbedFn` (a callable type — read `scoring/text_matcher.py` to find its exact signature before writing a fake embedder), `choicebench.scoring.text_matcher.match_text_to_options`.

- [ ] **Step 1: read `src/choicebench/scoring/text_matcher.py`** to get `EmbedFn`'s exact call signature (likely `Callable[[list[str]], np.ndarray]` or similar) and `match_text_to_options`'s cascade order (exact → containment → cosine), so Steps 2–5 can build a deterministic fake `embed_fn` that never needs a real sentence-transformers model (keeping this suite fast and offline). A fake `embed_fn` can be a one-hot-by-string-identity encoder — it only needs to make the *test's* exact/containment/cosine behavior deterministic, not be semantically meaningful.

- [ ] **Step 2 (G1): zero target-model calls.** `derive_matched_results` takes no backend parameter at all (confirmed this session, module docstring: "NOT an ExperimentRunner subclass... no backend/client dependency at all, by construction"). Assert this structurally via `inspect.signature(derive_matched_results).parameters` containing no `backend`/`client` parameter, plus a behavioral check: build a 1-row `source_df` with a `free_text_response` of `"Paris"` and confirm `derive_matched_results` runs to completion using only a fake `embed_fn` (no network, no `SpyBackend` needed at all — its absence IS the proof).

- [ ] **Step 3 (G2/G3): rotation-invariance of the matched result.** Build one source row per rotation of `diag_4a` (i.e. `per_rotation_choices_json`-shaped input is not what this function reads — it reads a *single* `free_text_response` per row, matched against `build_option_map(row)`, which is always canonical order regardless of what rotation *produced* that free text upstream). Construct two rows with identical `free_text_response="Paris"` but built from `question_row` dicts that differ only in irrelevant trace metadata (e.g. `run_id`), and assert both derive to the same `parsed_choice`. This demonstrates G2/G3: the derivation is a pure function of `(free_text_response, canonical option map)`, never of display order, because `build_option_map` is always canonical.

- [ ] **Step 4 (G4/Q5): the NaN bug — expected to FAIL, per Review Focus.**
  ```python
  import numpy as np
  import pandas as pd
  import pytest

  def test_g4_nan_free_text_must_not_become_literal_nan_string():
      row = _diag_row_dict("diag_4a")
      row["free_text_response"] = np.nan  # genuine pandas/numpy NaN, not None
      row["run_id"] = "t"; row["method_name"] = "two_stage"
      source_df = pd.DataFrame([row])
      result = derive_matched_results(source_df, method_name="semantic_matching_v1",
                                       embed_fn=_fake_embed_fn)
      derived = result.iloc[0]
      # frozen invariant: a missing/NaN free-text answer must remain
      # unscorable (no parsed_choice), never get matched as if "nan" were
      # a real free-text answer.
      assert pd.isna(derived["parsed_choice"]) or derived["parsed_choice"] is None
  ```
  Per `derive_from_free_text.py:38`'s actual logic (`free_text is not None and str(free_text).strip() != ""` — a NaN float passes both checks and gets stringified to `"nan"` then matched), this test is expected to currently FAIL. Do not weaken the assertion. If it fails, leave it failing (a red test IS the correct, successful outcome per the spec's Phase 3 instructions) — do NOT mark it `xfail` either, since spec §G4/§Q5 list this as a MANDATORY test whose failure must be reported, not silently absorbed. (The `xfail` escape hatch in Global Constraints is for tests that are *structurally* impossible to assert cleanly, not for "this genuinely fails" — a plain failing test is the correct artifact here.)

- [ ] **Step 5 (G5): wrong-source rejection, or documented absence.** Inspect whether `derive_matched_results`/`derive_from_free_text.py` validates `row.get("method_name")` against an expected source method, or benchmark/model identity, before deriving. From the code read this session, it does NOT validate source identity at all — it will happily "derive" from a row whose `method_name` is anything. Write the test per the frozen invariant anyway (spec explicitly allows this to fail):
  ```python
  def test_g5_wrong_source_method_should_be_rejected():
      row = _diag_row_dict("diag_4a")
      row["free_text_response"] = "Paris"
      row["method_name"] = "independent_hypothesis"  # WRONG source method for semantic matching
      row["run_id"] = "t"
      source_df = pd.DataFrame([row])
      with pytest.raises(ValueError):
          derive_matched_results(source_df, method_name="semantic_matching_v1",
                                  expected_source_method="two_stage",  # if no such kwarg exists, this itself proves G5's gap
                                  embed_fn=_fake_embed_fn)
  ```
  If `derive_matched_results` has no such parameter at all (confirmed this session — it does not), write the test to call it without that kwarg and assert a `ValueError`/rejection is raised anyway based on `derived_from_method_name` disagreeing with an expected value the caller supplies out-of-band — if no such check exists anywhere reachable, mark the test as a plain failing assertion (`assert False, "no source-method validation exists in derive_from_free_text.py — spec §G5 gap"`) rather than skipping it, so it shows up in the failing-invariants report.

- [ ] **Step 6: run and record**, then **Step 7: commit.**

---

## Task 7: `test_text_extraction_and_matcher.py` — spec §H (H1–H7)

**Files:**
- Create: `tests_conformance/test_text_extraction_and_matcher.py`

**Interfaces:**
- Consumes: `choicebench.methods.library.text_extraction.TextExtractionRunner.run_one/.run_rotations`, `choicebench.methods.library.visible_llm_matcher.VisibleLLMMatcherRunner.run_one/.run_matching_rotations`, `choicebench.scoring.text_matcher.match_text_to_options` (the exact-match cascade — a `SpyBackend` responder that echoes back the literal canonical option text will hit the "exact" branch and needs no real embedder).

- [ ] **Step 1 (H1/H2/H3): call counts for N=3/4/5.** Parametrize over the three diagnostic questions, `TextExtractionRunner.run_one`, assert `spy.call_count == n_choices` (1 per question — wait, spec says "N=4 text extraction: exactly 4 target-model generations" — re-read `TextExtractionRunner.run_one`: it makes exactly ONE call per question, not N. **Re-check**: spec H1-H3 must refer to `run_rotations` (N calls, one per rotation), not `run_one` (1 call) — confirm this reading against `text_extraction.py`'s two methods (`run_one`: 1 call; `run_rotations`: N calls, one per rotation, confirmed by reading the file this session) before writing the test. Write H1/H2/H3 against `run_rotations`, and add one extra (non-mandatory but cheap) assertion that `run_one` alone is exactly 1 call, to make the distinction explicit in the suite rather than silently picking one interpretation.

- [ ] **Step 2 (H4): visible+LLM matcher adds zero new text-extraction calls.** Build a `per_rotation_extracted_text` list (as `run_matching_rotations` expects) directly from Step 1's `run_rotations` output (`json.loads(row["per_rotation_raw_text_json"])`), on a *fresh* `SpyBackend` for the matcher stage. Assert the matcher's own spy records exactly N calls (its own Stage-2 LLM-match calls — this is what spec H4 is protecting: zero NEW *text-extraction* calls, not zero calls overall, since visible_llm_matcher legitimately makes its own N matcher calls). Prove no text-extraction call happened during this phase by never constructing a `TextExtractionRunner` at all in this step — only feeding it the already-produced JSON.

- [ ] **Step 3 (H5): same-rotation lineage.** Use a responder for the matcher's spy that echoes back which extracted-text substring it received; assert `spy.prompts[k]` contains `per_rotation_extracted_text[k]` for every `k` where that entry is not `None` (this proves rotation `k`'s matcher call consumed rotation `k`'s own text-extraction output, not a shuffled/misaligned one).

- [ ] **Step 4 (H6): trace-length mismatch fails loudly.** Call `run_matching_rotations(row, per_rotation_extracted_text=["Paris"], sample_index=0)` for a 4-option `row` (length-1 list vs 4 rotations expected). From reading `visible_llm_matcher.py` this session: `zip(rotations, per_rotation_extracted_text)` — Python's `zip` silently truncates to the shorter length, so this does NOT currently fail loudly; it silently processes only 1 of 4 rotations. Write the test asserting a raised exception (`ValueError`/`AssertionError`) per the frozen invariant; expect it to FAIL against current behavior (record in the failing-invariants report — do not weaken to "at most N calls").

- [ ] **Step 5 (H7): wrong-source artifact not silently accepted.** `VisibleLLMMatcherRunner.run_one` requires `"extracted_text" in question_row` (raises `KeyError` if absent — confirmed this session) but does NOT check that the extracted text actually came from the same model/benchmark/method. Test the KeyError case (this one genuinely passes: `with pytest.raises(KeyError): runner.run_one(row_without_extracted_text, 0)`), and separately write a test asserting rejection of a row carrying an `extracted_text` value alongside mismatched `derived_from_method_name`/`benchmark_name` fields the runner is expected (per spec) to check — expect this second one to FAIL/have no such check, and record it.

- [ ] **Step 6: run and record**, then **Step 7: commit.**

---

## Task 8: `test_ihs_protocol.py` — spec §I (I1–I10)

**Files:**
- Create: `tests_conformance/test_ihs_protocol.py`

**Interfaces:**
- Consumes: `choicebench.methods.library.independent_hypothesis.IndependentHypothesisRunner.run_one`, response format `"<score>NN</score>"` (confirmed via `_SCORE_PATTERN` this session).

- [ ] **Step 1 (I1/I2/I3): call count == n_choices.** Parametrize over `diag_4a`(4)/`diag_3a`(3)/`diag_5a`(5); responder returns `f"<score>{50 + i}</score>"` (monotonically increasing so the last-seen candidate always "wins" without ties, useful for I5 too); assert `spy.call_count == n_choices`.

- [ ] **Step 2 (I4): each call sees only its own candidate, not the option list.** Assert every recorded prompt contains its own option's text (from `canonical_options_for(question_id)`) but not any *other* option's text from the same question (loop over all other option texts and assert `not in prompt`) — this directly encodes "not the ordered full answer list."

- [ ] **Step 3 (I5): highest valid score wins.** Responder returns distinct scores per call index (e.g. `{1: 10, 2: 90, 3: 50, 4: 20}` for a 4-option question, keyed by 1-based call index which corresponds to option order A/B/C/D); assert `result["parsed_choice"]` equals the letter of the highest-scored option (`"B"` in this example) and `result["is_correct"]` matches whether `"B"` is gold for that question.

- [ ] **Step 4 (I6): all-invalid ⇒ no fabricated answer.** Responder returns `"garbage, no score tag"` for every call; assert `result["parsed_choice"] is None`, `result["answer_status"] == "failure"` or `result["score_status"]` reflects unscorable (check `scoring/types.py`'s exact status vocabulary before asserting the literal value — read it if not already known).

- [ ] **Step 5 (I7): partially invalid never becomes a competing zero.** Responder: option A gets `"<score>5</score>"`, option B gets garbage (unparseable), others get `"<score>1</score>"`. If the frozen "invalid → treated as 0.0 placeholder, excluded from argmax" behavior holds (confirmed via `_parse_confidence_score`'s own docstring this session — code already does this correctly), assert the winner is A (score 5, the genuine highest among *valid* candidates), never B (which would win if 0.0 were allowed to compete against option scores below 5 — pick option scores that make this distinguishable, e.g. give the "others" a score of `-1`... but scores must be 0-100 per range check, so instead give "others" `0.5`, making B's fabricated-zero score of `0.0` distinguishable from a genuine `0.5` if the bug existed). Also assert `row["option_b_score_parse_ok"] is False` and `row["option_b_score"] == 0.0` (documented placeholder value, not treated as competitive — verify by checking the winner is NOT B despite B's placeholder 0.0 being lower than the others' 0.5, which is the actually-correct behavior, not a bug to catch here — I7 is a "this should already pass" confirmation test, unlike I6/G4/H6).

- [ ] **Step 6 (I8): NaN/inf/out-of-range/malformed rejected.** Parametrize responses `"<score>nan</score>"`, `"<score>inf</score>"`, `"<score>-inf</score>"`, `"<score>150</score>"` (out of 0-100 range), `"<score>abc</score>"` (regex won't even match `abc` as `-?\d+(\.\d+)?`, so this one falls into the "no match" branch) — assert every one yields `parse_ok=False` for that candidate (inspect via the row's `option_X_score_parse_ok` field, or by making it the *only* option and confirming the whole question becomes unscorable per I6's assertion shape).

- [ ] **Step 7 (I9): transport failure on non-first candidate is visible.** `SpyBackend(fail_on=frozenset({2}))` on `diag_4a` (call 2 = option B). Assert `result["option_b_model_status"] == "failure"` (per `_assemble_result_row`'s `response.status` handling — confirm exact field name, `option_{suffix}_model_status`, from the source read this session) even though `result["answer_status"]` may still be `"success"` overall (since A/C/D presumably succeeded) — the point is the per-option status column makes candidate 2's failure visible in the persisted row, not that the aggregate status must itself flip to failure.

- [ ] **Step 8 (I10): genuine tie uses shared tie-break.** Responder gives two options identical top scores, others lower; compute the expected winner independently via `choicebench.scoring.tiebreak.resolve_tie` called directly with the same `(seed, benchmark_id, question_id, method_name, tied_canonical_ids)` tuple the runner would use, and assert `result["parsed_choice"]` matches that independently-computed winner (never "first tied letter encountered").

- [ ] **Step 9: run and record**, then **Step 10: commit.**

---

## Task 9: `test_pride_protocol.py` — spec §J (J1–J11)

**Files:**
- Create: `tests_conformance/test_pride_protocol.py`

**Interfaces:**
- Consumes: `choicebench.methods.library.pride.PriDeRunner` (constructor kwargs confirmed this session: `calibration_n`, `calibration_seed`, `require_full_calibration`, `calibration_questions`), `choicebench.config.schema.load_config`, `choicebench.backends.dummy_backend.DummyBackend`, `config/paper/mmlu_pride.yaml`, `config/paper/arc_pride.yaml`.

- [ ] **Step 1 (J1/J2): real production YAML → real runtime `calibration_n`.** Load `mmlu_pride.yaml` via `load_config`; extract `config.methods[0].params["calibration_n"]` (`== 77`, confirmed this session's read) — but J1 explicitly wants the *runtime PriDeRunner's* `calibration_n`, "not just YAML.preflight.n": construct a real `PriDeRunner` with `backend=DummyBackend()`, `calibration_n=config.methods[0].params["calibration_n"]`, `calibration_questions=<synthetic 77+ eligible rows>`, and assert `runner._calibration_n == 77` (private attribute — acceptable here since it's exactly the seam spec J1 asks to inspect: "Not just YAML.preflight.n"). Repeat for `arc_pride.yaml` expecting `15`.
  - This must go through **exactly the construction path `run_experiment.py` itself uses** to build a `PriDeRunner` from config — find that call site (grep `PriDeRunner(` in `cli/run_experiment.py`) and mirror its kwarg-passing exactly (including how `preflight_questions`/`calibration_questions` get threaded from the loaded preflight pool), rather than hand-picking kwargs, so this test really proves "YAML → runtime," not "YAML value copy-pasted into a constructor call I invented."

- [ ] **Step 2 (J3): MMLU-shaped pool — filter then select 77.** Build a synthetic pool of, say, 200 rows (all 4-option, i.e. all eligible for MMLU's modal-k=4), pass as `calibration_questions`, `calibration_n=77`, `calibration_seed=42`; call `runner._ensure_calibration()` (or trigger it via `run_many`); assert exactly 77 rows contributed (inspect `runner._calibration_state.estimation_question_ids` — confirmed field name this session — `len(...) == 77`).

- [ ] **Step 3 (J4/J5): ARC-shaped adversarial pool — the critical ordering test.** Build a pool of exactly 16 four-option rows and 40 three-/five-option rows (56 total), so that "sample 15 first, then filter" would very likely yield fewer than 15 eligible rows (expected eligible count after sampling 15 from 56 ≈ 15 × 16/56 ≈ 4.3, definitely `<15`), while "filter first (16 eligible), then sample 15" always succeeds. Construct `PriDeRunner(calibration_n=15, calibration_seed=42, require_full_calibration=True, modal_k=4, calibration_questions=<the 56 rows>)`, trigger calibration, and assert it succeeds with exactly 15 eligible (all 4-option) rows selected — this is the fixture spec J5 explicitly demands.

- [ ] **Step 4 (J6): all ARC calibration rows are 4-choice.** From Step 3's result, assert every selected row (`self._calibration_questions` rows whose `question_id` is in `estimation_question_ids`) has `n_choices == 4` (or equivalent `len(build_option_map(row)) == 4`).

- [ ] **Step 5 (J7): deterministic under seed 42.** Run Step 2's construction twice with the same inputs and `calibration_seed=42`; assert `estimation_question_ids` is byte-identical both times. Also run once with `calibration_seed=43` and assert it's very likely different (not a strict requirement, but a sanity check that seed actually matters — assert the two id-sets differ, accepting the astronomically small chance of collision).

- [ ] **Step 6 (J8): insufficient eligible pool ⇒ hard failure.** `require_full_calibration=True`, `calibration_n=15`, pool of only 5 eligible 4-option rows; assert `RuntimeError` is raised (confirmed exact exception type this session, `pride.py:234`).

- [ ] **Step 7 (J9): calibration/evaluation disjointness.** Build calibration pool with IDs `calib_0..calib_76` and an evaluation set with IDs `eval_0..eval_1139`; assert `set(calibration_ids) & set(evaluation_ids) == set()` — this is a fixture-construction invariant test (the production code has no explicit disjointness *check* to call into, per the spec's own framing "no evaluation question ID overlaps... in a synthetic/full fixture where both are known" — the test's job is to build the fixture correctly and confirm disjointness holds by construction of the frozen manifests, cross-referencing Task 3's `mmlu_eval_v2.csv`/ARC manifest against whatever validation-split source is used for calibration, if available offline; otherwise mark this a synthetic-only confirmation like B5).

- [ ] **Step 8 (J10): full-logit local path, not vLLM/top-k.** Structural test: assert `PriDeRunner.requires_score_options is True` (class attribute, confirmed this session) and that `config.schema.model_supports_logprobs` returns `False` for `backend="api"` and `True` for `backend="huggingface"`/`"dummy"` (confirmed this session's read of `schema.py:615-638`) — i.e. PriDe is structurally gated away from any API/vLLM backend by config validation itself (`_validate_logprob_compatibility`, `schema.py:641-654`). Also construct a synthetic config with `methods: [{name: pride, requires_logprobs: true}]` and `models: [{backend: api, ...}]` and assert `load_config` raises `ConfigError`.

- [ ] **Step 9 (J11): no alpha sweep active.** Grep `src/choicebench/methods/library/pride_math.py` and `pride.py` for any `alpha` parameter; per `mmlu_pride.yaml`'s own header comment ("No alpha parameter exists in this codebase... calibration_n IS the '5%' knob"), assert `PriDeConfig` (from `config/schema.py`) has no `alpha` field (`"alpha" not in {f.name for f in dataclasses.fields(PriDeConfig)}`) and that `PriDeRunner.__init__`'s signature has no `alpha`/`alpha_sweep` parameter (`inspect.signature`).

- [ ] **Step 10: run and record**, then **Step 11: commit.**

---

## Task 10: `test_tiebreak_protocol.py` — spec §K (K1–K8)

**Files:**
- Create: `tests_conformance/test_tiebreak_protocol.py`

**Interfaces:**
- Consumes: `choicebench.scoring.tiebreak.resolve_tie(*, seed, benchmark_id, question_id, method_name, tied_canonical_ids) -> int`, `majority_vote_with_tiebreak`, `PriDeRunner`/`IndependentHypothesisRunner` end-to-end for K6/K7 (real-production-boundary ties, not just the helper).

- [ ] **Step 1 (K1/K2): model/provider independence — direct helper test.** `resolve_tie` takes no `model`/`provider` argument at all (confirmed this session — the function signature is exactly `seed, benchmark_id, question_id, method_name, tied_canonical_ids`), so K1/K2 are true by construction. Assert this structurally (`inspect.signature(resolve_tie).parameters.keys() == {"seed","benchmark_id","question_id","method_name","tied_canonical_ids"}`), plus a behavioral double-check: call it twice with identical args (model/provider are simply never passed in, so there's nothing to vary) and assert equal results — this is a tautology by signature, and the test should say so in a comment, not pretend it's discovering something.

- [ ] **Step 2 (K3/K4): production-boundary test — display order and execution order don't affect the canonical tie.** Using `PermutationRunner` (Task 4/5 style) on `diag_4a` with a responder engineered to force an exact tie between two canonical options across rotations (e.g. 2 rotations vote "A", 2 vote "B"), run it once, then run it again with `build_rotations`' rotation list order reversed by monkeypatching... — simpler: call `majority_vote_with_tiebreak` directly with `choices=["A","B","A","B"]` vs `choices=["B","A","B","A"]` (same multiset, different order) and assert identical winners — this directly encodes K4 ("different execution order"). For K3 ("different displayed option order"), assert the *canonical* letters compared are unaffected by which rotation displayed them, by re-running Task 5's E7-style shuffle test through the tie-break path specifically (reuse Task 5's fixture with a responder engineered for an exact tie instead of E2/E3's no-tie cases).

- [ ] **Step 3 (K5): different tied set can differ, but must stay in-set.** Call `resolve_tie` with `tied_canonical_ids={0,1}` and `{0,2}` (same seed/benchmark/question/method) — no requirement they differ, but assert each call's result `in` its own input set (trivially true from `resolve_tie`'s own `return ids[index]`, so this is a structural/contract confirmation, not a discovery).

- [ ] **Step 4 (K6): PriDe tie uses shared protocol, not `np.argmax`.** Construct a `PriDeRunner` scenario where `apply_debiased_choice_from_defaults` (from `pride_math.py` — read this file's exact tie-handling before writing the assertion) faces an exact debiased-score tie between two letters; call it directly with engineered `lp_map`/calibration state producing equal Eq.8 scores for two letters, and independently compute the expected winner via `resolve_tie` with the same `(seed, benchmark_id, question_id, method_name, tied_ids)`; assert they match. Read `pride_math.py` (not yet read this session) before finalizing this step — confirm `apply_debiased_choice_from_defaults`'s exact tie-handling call chain.

- [ ] **Step 5 (K7): IHS tie uses shared protocol.** Already covered by Task 8 Step 8 (I10) — this test file can either duplicate a thin version or `import` and re-invoke that scenario; per DRY, add a one-line cross-reference comment here and keep the actual assertion in `test_ihs_protocol.py` to avoid duplicate maintenance, OR duplicate a minimal version here since `tests_conformance/README.md`'s per-file spec-section mapping expects K7 to live in this file — prefer a minimal local duplicate (a few lines) over a cross-file import for suite navigability, per the file-layout spec.

- [ ] **Step 6 (K8): no Python `hash()`/`random` module state leakage.** Call `resolve_tie` with the same args in two fresh subprocess-free ways that would differ if it used salted `hash()` or unseeded `random`: (a) call it twice in the same process, (b) set `PYTHONHASHSEED` to two different values via `subprocess.run([sys.executable, "-c", "..."], env=...)` for the same inputs and assert stdout matches — this genuinely exercises "no Python hash() affects it" (BLAKE2b + explicit seed inputs should be immune to `PYTHONHASHSEED`, unlike a naive `hash(tuple(...))`-based tie-break). This is the one test in this file worth spending subprocess-overhead on since it's the one K-item not already implied by reading the source.

- [ ] **Step 7: run and record**, then **Step 8: commit.**

---

## Task 11: `test_stochasticity_protocol.py` — spec §L (L1–L21)

**Files:**
- Create: `tests_conformance/test_stochasticity_protocol.py`

**Interfaces:**
- Consumes: `scripts.paper.run_stochasticity_repeats.run(...)` and its helpers `_load_canonical_obs0_rows`, `_validate_execution_mode`, `_SYNC_ONLY_METHODS` (all read this session), `choicebench.analysis.agreement.compute_agreement_rate`, `choicebench.cli.run_experiment.build_backend`.
- Note: `scripts/paper/` is not a package with an `__init__.py` necessarily — confirm importability (`python -c "from scripts.paper import run_stochasticity_repeats"`) before writing tests; add a `sys.path` shim in this file's own top if needed (acceptable here since it's mirroring how the script is actually invoked, i.e. `python scripts/paper/run_stochasticity_repeats.py` from repo root — tests should run from repo root too, matching `pytest`'s default rootdir behavior).

- [ ] **Step 1: monkeypatch `build_backend` to inject `SpyBackend` per repetition.** `run_stochasticity_repeats.run()` calls the real `build_backend(model_config, run_id, run_seed=..., execution_mode=..., model_identity=...)` per fresh repetition. Monkeypatch `scripts.paper.run_stochasticity_repeats.build_backend` (module-level import, patch via `monkeypatch.setattr`) with a factory that returns a *new* `SpyBackend` per call, recording each one in a list keyed by the `model_identity` argument it was called with — this is what proves L4 ("distinct invocation/cache identities... every fresh repetition gets a DISTINCT model_identity").

- [ ] **Step 2 (L1/L2/L3): obs0 reuse + 3 fresh calls, canonical order.** Write a `canonical_obs0_csv` (via `tmp_path`) with one row for `diag_4a` under `method_name="direct_mcq"`, matching identity columns (`provider`, `model_name`, `benchmark_name`, `prompt_version`, `temperature`, `max_tokens`) equal to what a synthetic `model_config` (built via a tiny in-memory `ModelConfig`, not a full YAML) will present. Write a `questions_csv` with that one row's full content. Call `run(questions_csv, model_config, run_id="t", output_path=tmp_path/"out.csv", method_name="direct_mcq", prompt_version="v1", canonical_obs0_csv=canonical_obs0_csv, n_repetitions=4, run_seed=42, resume=True, execution_mode="batch")` with the Step 1 monkeypatch active. Assert: zero `SpyBackend`s were even constructed for repetition 0 (obs0 is pure CSV reuse — confirmed this session, `run()`'s obs0 branch never calls `build_backend`); exactly 3 `SpyBackend`s were constructed (repetitions 1–3); each made exactly 1 call (`direct_mcq` = `DirectMCQRunner`, 1 call/question); the output CSV has exactly 4 rows (`repetition_index` 0,1,2,3) all for `question_id="diag_4a"`; L3: every prompt across all 4 observations was built from the *canonical* (unrotated) option order — assert via a responder that echoes the option block and compare against `diagnostic_benchmark.canonical_options_for("diag_4a")`'s order for observations 1-3 (obs0 has no live prompt to inspect here since it's reused from CSV — note this limitation in a comment).

- [ ] **Step 3 (L4/L5): distinct cache identity, no cross-repetition cache bleed.** From Step 2's recorded `model_identity` values, assert they're pairwise distinct (`f"stochasticity_{method_name}_rep{repetition_index}"`, confirmed this session) — this alone is what guarantees separate `cache_dir`s per `build_backend`'s own logic (`cli/run_experiment.py`'s cache_dir derivation from `model_identity`), so L5 ("no fresh repetition may resolve from another's cache") follows structurally from L4 holding; assert L4 directly (the model_identity strings), and add a comment explaining why L5 follows rather than re-testing the whole cache layer here (that's Task 12's job).

- [ ] **Step 4 (L6): resume doesn't create obs4 or duplicate 1-3.** Run Step 2's `run()` call twice against the *same* `output_path` (second call reuses the file, `resume=True`, fresh `SpyBackend`s per call since Step 1's monkeypatch always returns new instances but the *completed_pairs* check in `run()` should skip already-done `(question_id, repetition_index)` pairs). Assert the second call's return value (`n_written`) is `0`, and the output CSV still has exactly 4 rows total (no 5th `repetition_index=4` row, no duplicated rows for 0-3).

- [ ] **Step 5 (L7-L11): wrong-identity obs0 hard failures.** Five parametrized cases, each mutating exactly one of `provider`/`model_name`/`benchmark_name`/`prompt_version`/`temperature` (or `max_tokens`) in the *written CSV file* (per Review Focus: write then re-read via `pd.read_csv`, don't pass an in-memory frame) so it disagrees with `expected_identity`; assert `run()` raises `ValueError` (confirmed exact exception type this session, `_load_canonical_obs0_rows` raises `ValueError` for every one of these mismatches).

- [ ] **Step 6 (L12): duplicate obs0 rows hard failure.** Write `canonical_obs0_csv` with two conflicting rows for the same `question_id`+`method_name`; assert `ValueError` (confirmed, `run_stochasticity_repeats.py`'s `ambiguous_needed` check).

- [ ] **Step 7 (L13/L14): two_stage stochasticity — 3+3 fresh calls, no cross-repetition mixing.** Same harness as Step 2 but `method_name="two_stage"`, `execution_mode="sync"`. Since each fresh repetition constructs its own `TwoStageRunner` via `METHOD_REGISTRY["two_stage"]` and calls `runner.run_many_async(pending_df)` (confirmed this session — `run()`'s fresh-repetition branch uses `run_many_async`, which for `TwoStageRunner` does its own internal Stage1-then-Stage2 batching), each repetition's single `SpyBackend` should record exactly 2 calls (1 Stage1 + 1 Stage2, since there's only 1 pending question) — assert total Stage1 calls across all 3 fresh repetitions = 3, total Stage2 = 3 (`spy.call_count == 2` per repetition × 3 repetitions, split via prompt inspection: call 1 of each repetition's spy has no options in it (Stage1), call 2 does (Stage2)). For L14 (no cross-repetition mixing): assert repetition 2's Stage-2 prompt contains repetition 2's own Stage-1 response text (from that same spy's call 1 raw response `"CALL_001"`-style marker) and NOT repetition 1's or 3's marker text — this is exactly what the `CALL_NNN`-marker responder design is for.

- [ ] **Step 8 (L15/L16): sync-only enforcement.** `_validate_execution_mode("two_stage", "batch")` and `_validate_execution_mode("reasoning_two_stage", "batch")` both raise `ValueError` (confirmed this session); `_validate_execution_mode("two_stage", "sync")` does not raise. Call `run()` end-to-end with `method_name="two_stage", execution_mode="batch"` and assert it raises before any backend is even constructed (assert the Step 1 spy-factory was never called).

- [ ] **Step 9 (L17/L18): batch-eligible methods.** `_validate_execution_mode("direct_mcq", "batch")` and `_validate_execution_mode("reasoning_mcq", "batch")` do not raise (confirmed — `_SYNC_ONLY_METHODS` is exactly `{"two_stage", "reasoning_two_stage"}`).

- [ ] **Step 10 (L19): provider_seed None throughout.** Build the synthetic `model_config` with `provider_seed=None` explicitly; from Step 2/7's spy captures (extend `SpyBackend` or wrap `_make_request`... — actually `SpyBackend` doesn't go through `APIBackend`, it's used directly as `backend=spy` via the `runner_cls(backend=spy, ...)` construction in `run()`, so there's no `ModelRequest.seed` to inspect here at all — `SpyBackend.generate(prompt)` never sees a seed). Reframe L19 as a config-level check instead: assert the synthetic `model_config.provider_seed is None` was used for both the obs0 identity dict and every fresh repetition's `build_backend` call (from Step 1's captured call args, assert `model_config` passed to every `build_backend` invocation is the *same object* / has `provider_seed is None`), proving no per-repetition override was introduced.

- [ ] **Step 11 (L20): agreement metric arithmetic.** Direct oracle test (no runner needed): `compute_agreement_rate(pd.DataFrame({"question_id": ["q1"]*4, "parsed_choice": ["A","A","A","A"]}))` → `agreement_rate == 1.0`; same with `["A","A","A","B"]` → `agreement_rate == 0.0`. (Confirmed function signature this session.)

- [ ] **Step 12 (L21): missing/duplicate observation exclusion contract.** `compute_agreement_rate` groups by `question_id` and requires `len(group) >= 2` to count at all — it does NOT specifically require *exactly* `{0,1,2,3}` repetition indices (it has no `repetition_index` awareness at all per the code read this session — it's generic over any `group_by`/`answer_col`). Write the test per the frozen invariant: build a group with only `repetition_index` `{0,1,3}` (3 rows, missing 2) and a group with `{0,1,2,3,3}` (duplicate 3, 5 rows) fed through a wrapper that's supposed to enforce exactly-4-observations before calling `compute_agreement_rate` — if no such wrapper exists in production (confirmed: it doesn't, per this session's read), assert the *intended* behavior (`pytest.raises` or an excluded-count check) and expect this to FAIL, recording the gap ("no repetition-index completeness check exists between the stochasticity CSV and `compute_agreement_rate`").

- [ ] **Step 13: run and record**, then **Step 14: commit.**

---

## Task 12: `test_cache_identity.py` — spec §M (M1–M7)

**Files:**
- Create: `tests_conformance/test_cache_identity.py`

**Interfaces:**
- Consumes: `choicebench.infra.cache._cache_key(request, extra_identity=None)` (private but this IS the seam, confirmed this session), `choicebench.infra.cache.client_extra_identity(client)`, `choicebench.clients.openrouter_client.OpenRouterClient`, `choicebench.backends.batch_api_backend.BatchAPIBackend` (not yet read this session — **read it fully in Step 1 below** before writing M4).

- [ ] **Step 1: read `src/choicebench/backends/batch_api_backend.py` in full** (deferred from this session's research) to confirm exactly where it calls `_cache_key`/`client_extra_identity` (already grepped: `batch_api_backend.py:91` calls `client_extra_identity(self._raw_client)`) and how it threads that into its own cache/state key, so M4 can assert the *real* call site, not an assumed one.

- [ ] **Step 2 (M1): stable key for identical request+routing.** Build a real `ModelRequest` twice with identical fields; `_cache_key(request1) == _cache_key(request2)`.

- [ ] **Step 3 (M2/M3): upstream_provider/allow_fallbacks change the key.** Construct a real `OpenRouterClient(model_name="x", upstream_provider="deepinfra", allow_fallbacks=False)` and a second with `upstream_provider="together"` (or `allow_fallbacks=True`); compute `client_extra_identity(client)` for each and assert they differ; then assert `_cache_key(request, extra_identity=identity_a) != _cache_key(request, extra_identity=identity_b)` for the same `request`. Also assert a *plain* `OpenRouterClient(model_name="x")` (no pinning) yields `client_extra_identity(...) is None`, and its `_cache_key` equals the pre-existing-behavior key (`_cache_key(request, extra_identity=None)`) — this is the documented backward-compatibility guarantee from `cache.py`'s own docstring (R3 overlap, cross-reference in Task 16).

- [ ] **Step 4 (M4): Batch path incorporates the same identity — the previously-fixed bug.** From Step 1's finding, construct a `BatchAPIBackend` with a pinned `OpenRouterClient` the same way `APIBackend`/`CachingClientWrapper` would be, and assert the batch backend's own cache-key computation (whatever internal method Step 1 found calling `_cache_key`) produces the SAME key as `CachingClientWrapper`'s sync path would for byte-identical `(request, client)` — proving both transports agree on identity. This test should PASS given this session's finding that `client_extra_identity` is already shared (commit `56a9d90` per git log) — record it as a *regression guard*, not an expected failure.

- [ ] **Step 5 (M5): different deployment identity ⇒ no stale cache hit.** Using a real `ResponseCache(cache_dir=tmp_path)`, `put()` a response under `_cache_key(request, extra_identity=identity_a)`, then `get(_cache_key(request, extra_identity=identity_b))` with `identity_a != identity_b`; assert `None` (miss).

- [ ] **Step 6 (M6): sync vs Batch transport alone doesn't redefine scientific identity, but state stays isolated.** Assert `_cache_key(request, extra_identity=None)` is transport-agnostic (it's a pure function of `request` + `extra_identity`, with no transport-mode parameter at all — confirmed by its signature) — i.e. the *response* cache key for equal scientific requests is identical whether reached via `APIBackend` or `BatchAPIBackend`. Separately, assert `BatchAPIBackend`'s own *batch state* directory (`RUNS_DIR / run_id / "batch_state" / model_cache_id`, confirmed this session's read of `run_experiment.py:203-213`) is a *different* path than `APIBackend`'s response `cache_dir` (`RUNS_DIR / run_id / "cache" / model_cache_id`) for the same `model_cache_id` — proving operational isolation without claiming a new scientific identity.

- [ ] **Step 7 (M7): historical cache entries without new identity fields aren't misinterpreted.** `ResponseCache.put()` a payload computed under the OLD key formula (`_cache_key(request, extra_identity=None)`, i.e. what every pre-pinning entry would have used); assert `ResponseCache.get()` under the NEW formula's key (`_cache_key(request, extra_identity={"upstream_provider": "deepinfra", "allow_fallbacks": False})`) is a clean MISS (`None`), never a corrupted/misattributed hit — this is the "fail-loud-not-silent-reinterpret... via a clean cache miss instead of a raised error" property `cache.py`'s own docstring describes; assert it behaviorally.

- [ ] **Step 8: run and record**, then **Step 9: commit.**

---

## Task 13: `test_batch_sync_normalization.py` — spec §N (N1–N5)

**Files:**
- Create: `tests_conformance/test_batch_sync_normalization.py`

**Interfaces:**
- Consumes: `BatchAPIBackend` (read fully in Task 12 Step 1 — reuse those findings here), its `poll_batch()`/result-remapping methods (exact names TBD from that read), `choicebench.clients.types.{BATCH_IN_PROGRESS, BATCH_COMPLETED, BATCH_FAILED}`.

- [ ] **Step 1: from Task 12's full read of `batch_api_backend.py`, extract the exact method names** for: submitting a batch, polling status, fetching+remapping results back to per-request `ModelResponse`s, and resuming from a persisted batch-state file. Write them down in a comment block at the top of this test file before writing any test (this task depends on Task 12 Step 1's research; if executed out of order, redo that read here first).

- [ ] **Step 2 (N1): sync vs Batch normalize to the same `ModelResponse` shape.** Construct a fake provider client whose `generate()` (sync path) and whose batch-submit/poll/fetch (Batch path) are both fed the *same* underlying fake raw provider payload for the same logical request; assert the two resulting `ModelResponse` objects have equal `raw_text`, `status`, `finish_reason` (allow `latency_seconds`/`timestamp_utc` to differ, since those are transport-timing artifacts, not scientific content — this distinction should be explicit in the assertion, comparing only the scientific fields).

- [ ] **Step 3 (N2): out-of-order Batch results still map to the right request.** Build a fake batch result set where result IDs come back in reverse/shuffled order relative to submission order; assert `BatchAPIBackend`'s remapping still attaches each result to its originating request by ID, not by position (construct requests with distinguishable payloads, e.g. `"Q1"`, `"Q2"`, `"Q3"`, and results shuffled, and assert `responses[i].raw_text` corresponds to `requests[i]`'s own ID after remapping).

- [ ] **Step 4 (N3): duplicate/missing batch result IDs fail loudly.** Feed a fake result set missing one expected ID, and separately one with a duplicate ID; assert both raise (exact exception type from Step 1's read).

- [ ] **Step 5 (N4): resume from completed batch state reattaches, no duplicate submission.** Write a fake persisted batch-state file (whatever schema Step 1 found) marked as already submitted/completed with a given batch job ID; construct a `BatchAPIBackend` pointed at that state dir with a fake client whose "submit" method raises `AssertionError("must not be called")` if invoked; call the resume/generate path and assert no `AssertionError` was raised (i.e. submission was skipped in favor of reading the existing state) and results were reattached correctly.

- [ ] **Step 6 (N5): documented crash-window limitation, not a fabricated guarantee.** This is a documentation-presence test, not a behavioral one: assert that `batch_api_backend.py`'s module or class docstring explicitly documents the provider-acceptance-to-local-write crash window (search for a relevant comment/docstring substring found during Step 1's read); if no such documentation exists, mark this a failing invariant (missing documentation) rather than skipping it.

- [ ] **Step 7: run and record**, then **Step 8: commit.**

---

## Task 14: `test_resume_equivalence.py` — spec §O (O1–O9)

**Files:**
- Create: `tests_conformance/test_resume_equivalence.py`

**Interfaces:**
- Consumes: the four `scripts/paper/run_*_rotations.py` scripts' `run()` functions (`run_two_stage_rotations.run`, and the text-extraction/visible-matcher equivalents — read `run_text_extraction_rotations.py` and `run_visible_llm_matcher_rotations.py` in Step 1), `choicebench.infra.resumable_csv.check_resume_compatible`, Task 11's `run_stochasticity_repeats.run`.

- [ ] **Step 1: read `scripts/paper/run_text_extraction_rotations.py` and `scripts/paper/run_visible_llm_matcher_rotations.py` in full** (not yet read this session) to confirm they follow the same resumable-CSV-append pattern as `run_two_stage_rotations.py` (per-question-id skip-if-completed, `check_resume_compatible` pre-flight) before writing O2/O4.

- [ ] **Step 2 (O1): cyclic interrupted-vs-uninterrupted equivalence.** There is no standalone "run cyclic rotations resumably" script (`PermutationRunner.run_one` is a single atomic in-process call over all N rotations, with no persisted partial state between rotations within one question) — re-scope O1 to what's actually resumable: run `PermutationRunner.run_one` fully (uninterrupted) on `diag_4a`, capture the result row; separately, simulate "interrupted after 2/4 rotations" by monkeypatching `PermutationRunner._generate_rotations` or `_call_backend_generate` to raise after 2 calls, catch the exception, then run `run_one` again fresh (full retry, since there's no partial-rotation checkpoint mechanism at this granularity) and assert the final result is identical to the uninterrupted run. Document in a comment that this tests "retry produces the same result" rather than genuine mid-question resume, since production has no mid-question checkpoint for cyclic — if this scoping feels like it undershoots spec intent, flag it explicitly in the suite's final report as a documented interpretation choice, not a silent narrowing.

- [ ] **Step 3 (O2/O3/O4): genuinely resumable scripts — interrupt after N/2 questions, resume, compare.** For each of `run_two_stage_rotations.run`, the text-extraction-rotations script, and the visible-matcher-rotations script (all confirmed or to-be-confirmed resumable via `_load_completed_question_ids`-style logic): run against a 4-question source CSV to completion in one call (`run_A`); separately, run against the first 2 questions only (`output_path` A), then call `run()` again with the full 4-question source and `resume=True` pointed at the same `output_path` (`run_B`); assert the resulting CSV (sorted by `question_id`) is row-for-row identical to `run_A`'s (excluding any wall-clock/timestamp columns, if present — check which columns those are and exclude them explicitly, don't guess).

- [ ] **Step 4 (O5): stochasticity resume — exactly obs0-3 once each.** Already substantially covered by Task 11 Step 4 (L6) — cross-reference it here with a one-line comment rather than duplicating, since it's the identical scenario under a different spec letter.

- [ ] **Step 5 (O6): schema mismatch hard failure.** `check_resume_compatible(existing_path_with_different_columns, exact_columns=[...])` raises `ValueError` (already partially exercised by Task 11's L5-series tests via `run_stochasticity_repeats`'s own `required_columns` call; add a direct unit-level call here against `exact_columns` mode specifically, which none of the L-series tests hit since stochasticity uses `required_columns` mode, not `exact_columns` — confirm which of the two-stage/text-extraction scripts use `exact_columns` mode, per this session's read of `run_two_stage_rotations.py:112-114`, and test that one directly).

- [ ] **Step 6 (O7): wrong model/config/source identity, same columns.** For a script whose `check_resume_compatible` call includes `expected_method_name`, write an existing file with the *same* column shape but a different `method_name` value in every row; assert `ValueError`.

- [ ] **Step 7 (O8): conflicting duplicate completed rows don't silently collapse.** Write an existing output file with two rows for the same `question_id` with *different* content (e.g. different `parsed_choice`); call the relevant script's `_load_completed_question_ids`-equivalent function and assert it either raises or surfaces both rows (never silently dedupes via `drop_duplicates(keep="first")` without flagging it) — read the actual dedup logic (if any) in each script before asserting; if none of the rotation scripts have this check (plausible, since `_load_completed_question_ids` in `run_two_stage_rotations.py` just does `set(existing["question_id"].astype(str))`, which is naturally dedup-blind to *conflicting* content, only tracking presence), record this as a failing/gap invariant per spec's explicit allowance.

- [ ] **Step 8 (O9): `--no-resume` semantics.** Read each script's CLI `--no-resume` handling (confirmed for `run_two_stage_rotations.py`: `resume=not args.no_resume`, and `resume=False` just makes `completed_ids = set()`, meaning a fresh full run with `--no-resume` against an *existing* output file will `open(output_path, "a")` and APPEND duplicate rows, not overwrite — confirmed by reading `run()`'s `open(output_path, "a", ...)` call this session). Assert this actual behavior (append-with-no-dedup-guard under `--no-resume` against a pre-existing file) — per spec O9's own escape hatch ("If current intended semantics explicitly overwrite/create a fresh artifact instead, assert that behavior"), if production does NOT overwrite (it doesn't, per this read), the frozen invariant "must not accidentally append duplicate scientifically identical rows" is likely VIOLATED by `--no-resume` against a stale file; write the test to assert the *frozen* invariant (no duplicate rows after `--no-resume` + existing file) and expect it to fail, recording the gap — do not assert current append behavior as if it were the frozen invariant.

- [ ] **Step 9: run and record**, then **Step 10: commit.**

---

## Task 15: `test_metrics_oracles.py` — spec §P (P1–P10)

**Files:**
- Create: `tests_conformance/test_metrics_oracles.py`

**Interfaces:**
- Consumes: `choicebench.metrics.accuracy.Accuracy`, `choicebench.metrics.recall_rstd.RecallRStd`, `choicebench.metrics.order_sensitivity.OrderSensitivity`, `choicebench.analysis.paired_tests.mcnemar_exact_test`, `choicebench.analysis.agreement.compute_agreement_rate` (all read this session except `mad.py` — read it in Step 1 for P5's flip-rate-adjacent MAD cross-check if needed, though P5's flip rate is actually `OrderSensitivity.order_flip_rate`, already fully understood).

- [ ] **Step 1 (P1/P2/P3): hand-built accuracy oracle.** Build a 5-row `results_df`: 2 correct (`parsed_choice == correct_option`), 1 wrong (`parsed_choice` set, `!= correct_option`), 2 unscorable (`parsed_choice = None`/NaN). Hand-compute: `accuracy = 2/5 = 0.4` (P1, unscorable counts as incorrect in the denominator — confirmed via `Accuracy.compute`'s `total = len(results_df)`), `accuracy_conditional = 2/3 ≈ 0.6667` (P2, `scored = 3`), `parse/unscorable rate = 2/5 = 0.4` (P3 — note `Accuracy` itself doesn't report this directly; compute it by hand as `results_df["parsed_choice"].isna().mean()` and just assert the arithmetic, not that `Accuracy` exposes a field named exactly this — check if it does first). Call `Accuracy().compute(results_df)` and assert its `accuracy`/`accuracy_conditional` match the hand-computed values exactly (`pytest.approx` for float safety).

- [ ] **Step 2 (P4): Clopper-Pearson against `scipy.stats.beta` directly, not `Accuracy`'s own helper.** Pick `k=2, n=5`; compute expected bounds via `scipy.stats.beta.ppf(0.025, 2, 4)` / `scipy.stats.beta.ppf(0.975, 3, 3)` directly in the test (this IS "a trusted mathematical primitive," per spec P4 — `scipy.stats.beta` is the primitive, `Accuracy._clopper_pearson` is the function under test); assert `Accuracy().compute(results_df)["accuracy_ci_low"/"accuracy_ci_high"]` match these independently-computed bounds to high precision.

- [ ] **Step 3 (P5): flip-rate hand computation.** Construct `per_rotation_choices_json` values by hand for 3 synthetic rows (`correct_index` known): one all-same-letter (no flip), one with 2 distinct canonical picks (flip), one with a failed rotation (`null` entry) mixed with otherwise-consistent picks (no flip, since only 1 unique non-null value). Hand-compute expected `order_flip_rate = 1/3`; call `OrderSensitivity().compute(pd.DataFrame(rows))` and assert `order_flip_rate == pytest.approx(1/3)`.

- [ ] **Step 4 (P6): RStd from known per-letter recalls.** Spec's own example: 4 letters with recalls `[1.0, 0.5, 0.0, 0.5]` (as fractions — but `RecallRStd` reports recall as a *percentage*, confirmed this session: `recall_pct = ... * 100`). Build a `results_df` engineered so each letter's `(gold_rows["parsed_choice"] == opt).sum() / n` equals `1.0, 0.5, 0.0, 0.5` respectively (e.g. n=2 per letter: letter A both correct, B one of two, C zero of two, D one of two). Hand-compute `population_std([100, 50, 0, 50]) = std of [100,50,0,50]` via `numpy.std` directly in the test (population, ddof=0) as the independent oracle; assert `RecallRStd().compute(results_df)["rstd"] == pytest.approx(that value)`.

- [ ] **Step 5 (P7): ARC RStd — only 4-option rows contribute.** Build a `results_df` mixing 3-, 4-, and 5-option rows (via a `_label_set`-driving column — read `metrics/base.py`'s `_label_set` helper to confirm how `RecallRStd` determines the active label set from `results_df`, since it iterates `options = _label_set(results_df)`, which likely derives from the *whole* frame's `n_choices`/`correct_option` values — if `_label_set` doesn't itself filter by option count, the *test* must pre-filter the input DataFrame to only 4-option rows before calling `RecallRStd().compute()`, matching how a real ARC pipeline consumer is expected to do this filtering upstream, per spec P7's phrasing "ARC RStd consumer... only 4-option rows contribute" — this describes a *consumer* responsibility, not necessarily `RecallRStd` itself). Read `metrics/base.py` before finalizing; write the test to filter to `n_choices == 4` rows exactly as the real ARC RStd computation must (find that call site, e.g. in a reporting script, to confirm this filtering actually happens in production — if no such filtering call site exists anywhere in the repo, record that as a gap).

- [ ] **Step 6 (P8): bootstrap determinism, resample count, question-level resampling.** Call `RecallRStd()._bootstrap_std(scored_df, options)` twice with the identical input; assert byte-identical output (determinism, seed=42 hardcoded internally per `_BOOTSTRAP_SEED = 42`, confirmed this session). Assert `_N_BOOTSTRAP == 10_000` (confirmed). For "question-level resampling rather than position-cell resampling": inspect the code (`idx = rng.integers(0, n, size=(_N_BOOTSTRAP, n))` then `gold_enc[idx]`/`pred_enc[idx]`, confirmed this session) — this resamples whole *rows* (questions), not per-letter cells, satisfying the invariant structurally; assert this by checking that a resampled row's `(gold, pred)` pair always co-occurs as they did in the original row (i.e. `gold_boot[i,j]` and `pred_boot[i,j]` come from the same original row index `idx[i,j]` for both arrays — true by construction since both use the same `idx`; write a small numeric example distinguishing "row-level" from "cell-level" resampling to make this concrete rather than asserting it from reading code alone: construct a 2-row `scored` frame where row-level resampling could never produce a (gold=A,pred=B) pair that didn't already exist in some real row, and confirm no such impossible pair appears across a bootstrap run).

- [ ] **Step 7 (P9): McNemar contingency counts.** Build `results_a`/`results_b`, each 4 rows over `question_id in {q1,q2,q3,q4}`: q1 both correct, q2 both wrong, q3 a-only-correct, q4 b-only-correct. Call `mcnemar_exact_test(results_a, results_b)`; assert `n_paired == 4`, `n_a_only == 0`, `n_b_only == 0`, `b == 1` (a correct, b incorrect = q3), `c == 1` (q4), `n_discordant == 2`, and `p_value == pytest.approx(1.0)` (since `binomtest(min(1,1), 2, 0.5).pvalue == 1.0` exactly — verify independently via `scipy.stats.binomtest(1, 2, 0.5).pvalue` in the test itself as the oracle, not by trusting the implementation).

- [ ] **Step 8 (P10): stochasticity agreement — already covered.** Cross-reference Task 11 Step 11 (L20) rather than duplicating; add one additional case here with a *known small example with a non-trivial fraction* (spec P10's own phrasing, beyond L20's binary examples): 3 groups of 4 observations each, 2 fully agreeing, 1 disagreeing → `agreement_rate == 2/3`; assert against `compute_agreement_rate`.

- [ ] **Step 9: run and record**, then **Step 10: commit.**

---

## Task 16: `test_provenance.py` — spec §Q (Q1–Q6)

**Files:**
- Create: `tests_conformance/test_provenance.py`

**Interfaces:**
- Consumes: `ExperimentRunner._build_result_row`'s full field list (already read in full this session, `methods/base.py:200-299`), `derive_baseline_from_cyclic`, `derive_matched_results`, `VisibleLLMMatcherRunner`.

- [ ] **Step 1 (Q1): observed row identity completeness.** Run `PermutationRunner.run_one` on `diag_4a` with a real `SpyBackend`; assert the result row contains non-null values for: `provider`, `model_name`, `benchmark_name`, `question_id`, `method_name`, `prompt_version`, `temperature`, `max_tokens`, `parsed_choice` (post-hoc, may be a real letter here), `per_rotation_choices_json` (display-mapping trace). This is a schema-completeness assertion, not a value-correctness one — every field must simply be *present and non-null* on a successful row.

- [ ] **Step 2 (Q2): baseline row lineage fields.** Already substantially covered by Task 4 Step 3 (D3) — cross-reference; add the one additional field D3 didn't explicitly check: assert `"per_rotation_choices_json" not in baseline_row` (or is `None`/absent) since direct_mcq "has no rotations of its own" per `derive_baseline_from_cyclic.py`'s own comment (confirmed this session, `.pop("per_rotation_choices_json", None)`).

- [ ] **Step 3 (Q3): semantic-derived row doesn't masquerade as a fresh generation.** From `derive_matched_results`'s output row, assert `derived_from_method_name` and `derived_from_run_id` are both present and non-null, and that the row carries no `transport_status` implying a fresh call happened when none did — check whether `derive_matched_results` sets `transport_status` at all (from the code read this session, it does NOT set `transport_status` — it inherits whatever the source row had, via `derived = dict(row)`); assert the derived row's `transport_status` equals the *source* row's `transport_status` unchanged (proving it's not fabricating a new transport outcome), which is the concrete form of "must not masquerade as a new target-model generation."

- [ ] **Step 4 (Q4): visible-matcher artifact retains text-extraction source relationship.** From `VisibleLLMMatcherRunner.run_one`'s result row, assert `"reused_extracted_text"` is present (confirmed this session, `row["reused_extracted_text"] = extracted_text`) and equals the input `question_row["extracted_text"]` exactly.

- [ ] **Step 5 (Q5): NaN/null free text remains missing.** Cross-reference Task 6 Step 4 (G4) — same test, same expected failure; add here only the `normalized_text` field check specifically (`derived["normalized_text"] = str(free_text) if free_text is not None else None` — confirmed this session this is where the NaN literally becomes the string `"nan"`), asserting `derived["normalized_text"]` is `None`/NaN rather than the string `"nan"` — expect this to FAIL too (same root cause as G4), record both.

- [ ] **Step 6 (Q6): internally coherent persisted status.** For a row where the source generation failed (`transport_status == "failure"`), assert `is_correct` is never `True` for that row (it should be `None`/NaN, since `score_result` is `None` when `model_response.is_success()` is `False` in every runner read this session) — build this via `SpyBackend(fail_on={1})` on a single-call method (`text_extraction`'s `run_one`) and check the resulting row's `is_correct`/`transport_status`/`answer_status` combination is internally consistent (`transport_status="failure"` ⇒ `answer_status="failure"` ⇒ `is_correct in (None,)`, never `True`).

- [ ] **Step 7: run and record**, then **Step 8: commit.**

---

## Task 17: `test_backward_compatibility.py` — spec §R (R1–R5)

**Files:**
- Create: `tests_conformance/test_backward_compatibility.py`

**Interfaces:**
- Consumes: `load_config`, a pre-v2 config file if one is committed anywhere findable (search `git show 549c9bcc2f9f4db3c74b3bc2394cfea145fb2c7b:<path>` for an old config, or check if any non-`config/paper/` config under `config/` predates the v2 schema), `client_extra_identity`, `PriDeConfig`/`PriDeRunner` defaults.

- [ ] **Step 1: find a genuine pre-v2 config.** Run `git show 549c9bcc2f9f4db3c74b3bc2394cfea145fb2c7b:<path> 2>/dev/null` is not directly useful without knowing a path — instead run `git ls-tree -r 549c9bcc2f9f4db3c74b3bc2394cfea145fb2c7b --name-only -- config/` to list what config files existed at that pre-v2 anchor commit, then `git show 549c9bcc:<one of those paths>` to check it out into a `tmp_path` file for R1 (never write it back into the working tree).

- [ ] **Step 2 (R1): old config still parses/runs through the offline fake path.** `load_config(str(old_config_tmp_path))` — if the schema rejects it outright (e.g. an old field name that's since been renamed/removed, or `_reject_unknown` failing on a since-removed key), that itself IS the R1 finding (record whether it currently passes or fails; don't modify the schema or the old config to make it pass). If it parses, additionally construct its first model via `build_backend(..., model_identity="t")` (with env-var monkeypatching as needed) and confirm no exception.

- [ ] **Step 3 (R2): `provider_seed` default None doesn't alter other seed behavior.** Build two otherwise-identical synthetic configs, one omitting `provider_seed` entirely and one with no models at all differing — actually the cleanest R2 test: construct `PermutationRunner`/any runner with `seed=42` (the *experiment* seed) and confirm `runner.seed == 42` regardless of whatever `provider_seed` was set to at the `ModelConfig`/backend-construction layer (these are simply different objects/layers — `ExperimentRunner.seed` never reads `ModelConfig.provider_seed` at all, confirmed via `methods/base.py`'s constructor, which takes its own `seed` kwarg independently). Assert this structural separation by `inspect.signature(ExperimentRunner.__init__)` containing `seed` but not `provider_seed` as a parameter name.

- [ ] **Step 4 (R3): legacy non-pinned cache keys unchanged.** Direct extension of Task 12 Step 3's "plain `OpenRouterClient` (no pinning) yields `client_extra_identity(...) is None`" finding — assert byte-identical `_cache_key(request, extra_identity=None)` output today matches what the pre-existing (pre-pinning) formula would have produced, by re-deriving the pre-pinning formula by hand (the same `key_data` dict literal minus any `extra_identity` key, hashed with `hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()`) and asserting equality — this proves R3's "should not unnecessarily alter legacy cache keys" for the non-pinned case, matching `cache.py`'s own docstring promise.

- [ ] **Step 5 (R4): PriDe strict calibration is opt-in.** Construct a `PriDeRunner` with `require_full_calibration` omitted (default `False`, confirmed this session) and a calibration pool smaller than `calibration_n`; assert it does NOT raise (falls back to a uniform prior / smaller calibration set, per `_compute_calibration_state`'s graceful-degrade path) — this is the "does not silently break generic historical callers" half of R4. Then repeat with `require_full_calibration=True` and confirm it DOES raise (already covered by Task 9 J8, cross-reference).

- [ ] **Step 6 (R5): conformance tests never touch real historical outputs.** This is a suite-hygiene meta-test: grep every file in `tests_conformance/` for hardcoded paths under `runs/`, `cache/`, `data/manifests/` used for *writing* (not reading) — assert none exist by construction (a `grep -rn "runs/\|cache/" tests_conformance/*.py` restricted to write-mode calls, or more simply: a code-review pass, not an automated assertion — write this as a documented manual-review checklist item in `README.md` rather than a flaky grep-based test, since "assert no test writes outside tmp_path" is better enforced by disciplined code review than a brittle static check).

- [ ] **Step 7: run and record**, then **Step 8: commit.**

---

## Task 18: `test_call_arithmetic.py` — spec §S (S1–S2)

**Files:**
- Create: `tests_conformance/test_call_arithmetic.py`

**Interfaces:**
- Consumes: everything from Tasks 4–11 (this file's job is to assert the *formulas* directly, parametrized over N=3/4/5, as pure arithmetic assertions cross-checked against actual call counts already proven in earlier tasks — not to re-run every runner again from scratch).

- [ ] **Step 1 (S1): encode the eight formulas as parametrized arithmetic tests.**
  ```python
  import pytest

  @pytest.mark.parametrize("n", [3, 4, 5])
  def test_s1_call_arithmetic_formulas(n):
      assert _cyclic_calls(n) == n
      assert _baseline_additional_calls(n) == 0
      assert _two_stage_calls(n) == (1, n)  # (stage1, stage2)
      assert _semantic_calls(n) == 0
      assert _text_extraction_calls(n) == n
      assert _visible_matcher_new_text_extraction_calls(n) == 0
      assert _ihs_calls(n) == n
      assert _reasoning_mcq_calls(n) == n
      assert _reasoning_two_stage_calls(n) == (1, n)
      assert _stochasticity_baseline_additional(n) == 3  # n-independent: obs1-3, 1 call/obs
      assert _stochasticity_two_stage_additional(n) == 6  # 3 * (1 stage1 + 1 stage2), also n-independent
      assert _stochasticity_reasoning_mcq_additional(n) == 3
      assert _stochasticity_reasoning_two_stage_additional(n) == 6
  ```
  Implement each `_xxx(n)` helper as a **one-line formula function** (e.g. `def _cyclic_calls(n): return n`) directly in this file — these are NOT calls into production code; they are the plan's own restatement of spec §S's formulas, existing so the *next* step can cross-check them against real call counts already measured in Tasks 4–11, catching any drift between "what the spec says the formula is" and "what earlier tasks actually measured."

- [ ] **Step 2: cross-check formulas against Tasks 4–11's actual measured call counts.** For each formula, either re-invoke the relevant runner here (cheap, since `SpyBackend`+diagnostic fixture setup is a 5-line helper by now) or, to avoid duplicate test execution cost, import the specific assertion helper functions Tasks 4–11 should have factored out (e.g. `_run_cyclic(question_id)` from `test_call_graphs.py`) and reuse them; assert `spy.call_count == _cyclic_calls(n)` etc. for at least one concrete N per formula (N=4 is sufficient here since Tasks 4-11 already covered N=3/5 exhaustively; this task's job is cross-checking the *formula*, not re-proving per-N coverage).

- [ ] **Step 3 (S2): derive production totals from final manifests/configs — first principles, report only.** Write a **non-assertion** reporting test (or a small script under `tests_conformance/` clearly marked as a report generator, e.g. `report_call_totals.py`, not collected by pytest) that:
  1. Loads `mmlu_eval_v2.csv` (1140 rows) and the ARC manifest (1172 rows, with 1165/4/3 by option count from Task 3).
  2. For each paper method + benchmark combination actually present across `config/paper/*.yaml` (enumerate from Task 2's config inventory), applies the Step 1 formulas using each question's real `n_choices` (not just the modal N=4) to compute total calls.
  3. Prints a per-method-per-benchmark table plus a grand total.
  Do not hardcode `69,353`/`72,953` or any other historical total (per spec's explicit prohibition) — the number that comes out is whatever the current frozen manifests+configs produce, and must be reported as-is in the final report, flagged as uncertain if any method's presence in the "production matrix" is ambiguous (e.g. which benchmark×model×method combinations actually run in the final paper — cross-reference `config/paper/*.yaml`'s method lists literally, don't guess).

- [ ] **Step 4: run and record**, then **Step 5: commit.**

---

## Task 19: Full-suite run, existing-suite regression check, and final report

**Files:** none created — this task runs and reports.

- [ ] **Step 1: (Phase 4 per spec) Read `tests/` for helper/fixture conventions only.** Skim `tests/conftest.py` and 2-3 files under `tests/runners/`/`tests/pipeline/` for: how existing tests construct prompts_dir/prompt_version, any existing fake backend classes that could have simplified `SpyBackend` (if one already exists and is more complete, note it in the final report as a possible future dedup, but do NOT replace `tests_conformance`'s independent fakes with it now — that would blur "independent" for no benefit at this late stage). Do not change any earlier task's test *expectations* based on this read — only note fixture-syntax learnings.

- [ ] **Step 2: run the full conformance suite.**
  ```bash
  cd /home/cotenthusiast/Projects/choicebench && python -m pytest tests_conformance/ -v --tb=short 2>&1 | tee /tmp/conformance_run.txt
  ```
  Record exact pass/fail/xfail counts and every failing test's ID.

- [ ] **Step 3: run the existing full test suite for regression.**
  ```bash
  cd /home/cotenthusiast/Projects/choicebench && python -m pytest tests/ -q 2>&1 | tail -50
  ```
  Confirm the new suite changed nothing here (it shouldn't — `tests_conformance/` doesn't touch `tests/` or production code). Record pass/fail counts.

- [ ] **Step 4: `git diff --check`.**
  ```bash
  git diff --check
  git status
  ```
  Confirm zero whitespace errors and that only `tests_conformance/` and `docs/superpowers/plans/` files are new/modified (no `src/`, `config/`, `scripts/`, `prompts/` changes).

- [ ] **Step 5: write the final report** in the exact structure spec's "FINAL REPORT" section demands (A–H), as a message to the user — NOT as a new committed file (spec doesn't ask for a report file, and CLAUDE.md says not to create unrequested docs). Use the accumulated pass/fail/xfail records from every task's "run and record" step, plus:
  - **E (suspicious behavior found while writing tests):** the NaN→"nan" bug in `derive_from_free_text.py:38/51` (spec §G4/§Q5), the `zip()`-silent-truncation in `visible_llm_matcher.py`'s `run_matching_rotations` (spec §H6), the `--no-resume` append-without-dedup behavior in `run_two_stage_rotations.py`/similar (spec §O9), any `derive_matched_results`/`VisibleLLMMatcherRunner.run_one` source-identity-validation gaps (spec §G5/§H7), and anything else Tasks 1–18 turned up while implementing (e.g. from Task 12's read of `batch_api_backend.py`, Task 3's manifest/data availability findings, Task 9's `pride_math.py` tie-break confirmation).
  - **D (live-validation-only properties):** everything spec's own "LIVE TESTS ARE OUT OF SCOPE" section lists, plus any offline-infeasible items discovered along the way (e.g. B2's MMLU HF-dataset regeneration, B4 if ARC normalized source isn't committed).

- [ ] **Step 6: final commit** (only if Steps 2-5 produced any last cleanup edits; otherwise this task has no commit of its own beyond the report message).

---

## Self-Review Notes (completed during plan authoring, not a task to execute)

**Spec coverage:** every lettered section A–S has a task (Tasks 2–18); Task 1 is shared infra; Task 19 is the run/report phase. Cross-references between overlapping items (D3/Q2, G4/Q5, I10/K7, L6/O5, L20/P10) are called out explicitly in the tasks themselves so no item is silently dropped for "already covered elsewhere" without a citation.

**Placeholder scan:** every task step names an exact production symbol (module path + function/class name) and either an exact expected value copied from `FROZEN_SPEC.md` or an exact currently-observed behavior (with file:line) that the test is expected to catch as a failure. Steps that depend on not-yet-read source (`batch_api_backend.py`, `pride_math.py`, `run_text_extraction_rotations.py`, `run_visible_llm_matcher_rotations.py`, `scoring/text_matcher.py`, `pipeline/options.py`, `metrics/base.py`) say so explicitly and name exactly what to read and why, rather than hand-waving "figure it out."

**Type/interface consistency:** `SpyBackend`/`RecordedCall`/`diagnostic_rows()`/`canonical_options_for()`/`gold_source_index_for()` (Task 1) are the only shared symbols reused across Tasks 2–18; every later task's Interfaces block names them consistently.

**Review Focus coverage:** the five items listed above each map to a specific task/step: NaN→"nan" (Task 6 Step 4 / Task 16 Step 5), `ModelRequest` seed default trap (Task 2 Step 4), PriDe eligibility-order adversarial fixture (Task 9 Step 3), duck-typed `client_extra_identity` requiring a real client class (Task 12 Steps 2-3), and obs0 identity checks needing a real CSV round-trip (Task 11 Step 5).
