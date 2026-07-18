# Method Comparison Walkthrough Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `examples/method_comparison.md` — the extensibility walkthrough the README promises ("in an afternoon you can add a new debiasing method... and run it against MMLU") — backed by a real, runnable demo method and a real captured run, not invented output.

**Architecture:** Add one new per-question method, `ShuffledBaselineRunner` (`src/choicebench/methods/library/shuffled_baseline.py`), registered exactly the way an external user would register a built-in method. Run it head-to-head with `direct_mcq` on the bundled toy dataset via a new example config (`examples/method_comparison.yaml`), capture the real terminal output, then write the walkthrough doc around that captured output.

**Tech Stack:** Python 3.10+, pytest, the existing ChoiceBench pipeline (no new dependencies).

## Global Constraints

- Do not restructure existing code — no unrelated refactors, no touching files outside what each task below lists.
- The only `src/`/`tests/` additions are the demo method and its test. `examples/method_comparison.yaml` and `examples/method_comparison.md` are the walkthrough deliverables (consistent with `examples/hpc/` already holding real runnable configs/scripts, not just docs).
- `ShuffledBaselineRunner` must be a per-question method: `requires_score_options = False` (inherited default) and `applies_modal_k_gate = False` (inherited default) — no modal-k gate involvement.
- Follow the conventions in `src/choicebench/methods/library/cyclic_logprob.py` / `direct_mcq.py`: module-level docstring (Method/Description/Reference/Backend requirements/Logprob support/Calls per question), `from __future__ import annotations`, type hints.
- Register the method exactly as a user following the README's "Add a new method (3 steps)" section would: `registry.py` + `methods/library/__init__.py` (+ `methods/__init__.py` re-export, matching the existing five).
- Every command and output shown in `examples/method_comparison.md` must be one you actually ran in this session — no invented output. If the walkthrough-as-specced doesn't work as planned, fix the walkthrough, not the framework, unless it's a genuine framework bug (stop and report before patching).
- Update `README.md`'s examples reference and `CHANGELOG.md`'s Unreleased section (no such section exists yet — add one above `## v0.1.1`).
- Full test suite must pass at the end with no regressions.

---

### Task 1: Implement `ShuffledBaselineRunner` and register it

**Files:**
- Create: `src/choicebench/methods/library/shuffled_baseline.py`
- Modify: `src/choicebench/methods/library/__init__.py`
- Modify: `src/choicebench/methods/__init__.py`
- Modify: `src/choicebench/registry.py`
- Test: `tests/runners/test_shuffled_baseline.py`

**Interfaces:**
- Produces: `ShuffledBaselineRunner` (subclass of `ExperimentRunner`), with `run_one(question_row, sample_index) -> dict` and a `staticmethod _shuffle_options(canonical_options: dict[str, str], rng: random.Random) -> tuple[dict[str, str], dict[str, str]]` (returns `(shuffled_options, label_map)` where `label_map` maps shuffled letter → canonical letter).
- Registered under method name `"shuffled_baseline"` in `METHOD_REGISTRY`.

- [ ] **Step 1: Write the failing test file**

Create `tests/runners/test_shuffled_baseline.py`:

```python
# tests/runners/test_shuffled_baseline.py

from __future__ import annotations

import random
from pathlib import Path

from choicebench.clients.types import ProviderTimeoutError
from choicebench.methods.library.shuffled_baseline import ShuffledBaselineRunner
from choicebench.scoring.types import SCORE_CORRECT, SCORE_UNSCORABLE
from choicebench.parsing.types import PARSE_OK

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend, seed=0, method_name="shuffled_baseline"):
    return ShuffledBaselineRunner(
        backend=backend,
        method_name=method_name,
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run",
        seed=seed,
    )


class TestShuffledBaselineRunnerClassAttributes:
    def test_requires_score_options_is_false(self):
        assert ShuffledBaselineRunner.requires_score_options is False

    def test_applies_modal_k_gate_is_false(self):
        assert ShuffledBaselineRunner.applies_modal_k_gate is False


class TestShuffleOptions:
    def test_shuffle_is_a_bijection_over_letters_and_text(self):
        canon = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        rng = random.Random(0)
        shuffled, label_map = ShuffledBaselineRunner._shuffle_options(canon, rng)

        assert set(shuffled.keys()) == set(canon.keys())
        assert sorted(shuffled.values()) == sorted(canon.values())
        assert set(label_map.keys()) == set(canon.keys())
        assert set(label_map.values()) == set(canon.keys())
        for shuffled_label, canonical_label in label_map.items():
            assert shuffled[shuffled_label] == canon[canonical_label]

    def test_shuffle_is_deterministic_for_same_rng_seed(self):
        canon = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        shuffled_1, map_1 = ShuffledBaselineRunner._shuffle_options(canon, random.Random(7))
        shuffled_2, map_2 = ShuffledBaselineRunner._shuffle_options(canon, random.Random(7))
        assert shuffled_1 == shuffled_2
        assert map_1 == map_2


class TestShuffledBaselineRunnerRunOne:
    def test_run_one_remaps_shuffled_letter_to_canonical(self, runner_question_row):
        """The model answers with a shuffled-prompt letter; run_one must score
        against the canonical letter it maps back to, not the raw letter."""
        canon = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        seed = 0
        rng = random.Random(f"{seed}:{runner_question_row['question_id']}")
        shuffled, label_map = ShuffledBaselineRunner._shuffle_options(canon, rng)
        shuffled_letter_for_correct = next(
            letter for letter, text in shuffled.items() if text == "HTTPS"
        )

        backend = MockBackend(responses=[shuffled_letter_for_correct])
        result = _make_runner(backend, seed=seed).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT
        assert result["parse_status"] == PARSE_OK

    def test_run_one_matches_answer_text_regardless_of_shuffle(self, runner_question_row):
        """Text-match fallback works against the shuffled options too."""
        backend = MockBackend(responses=["The answer is HTTPS."])
        result = _make_runner(backend, seed=1).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_failed_response(self, runner_question_row):
        backend = MockBackend(responses=[ProviderTimeoutError("Request timed out.")])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["error_type"] == "ProviderTimeoutError"

    def test_unparseable_response(self, runner_question_row):
        backend = MockBackend(responses=["I'm not sure about this question"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    def test_prompt_contains_all_option_texts(self, runner_question_row):
        backend = MockBackend(responses=["The answer is HTTPS."])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)

        assert "FTP" in result["prompt"]
        assert "HTTP" in result["prompt"]
        assert "HTTPS" in result["prompt"]
        assert "SMTP" in result["prompt"]

    def test_same_question_same_seed_produces_same_prompt(self, runner_question_row):
        """The shuffle is seeded per (run seed, question_id) — repeat calls
        for the same question in the same run must render an identical prompt."""
        backend = MockBackend(responses=["The answer is HTTPS.", "The answer is HTTPS."])
        runner = _make_runner(backend, seed=3)
        result_1 = runner.run_one(runner_question_row, sample_index=0)
        result_2 = runner.run_one(runner_question_row, sample_index=1)

        assert result_1["prompt"] == result_2["prompt"]
```

- [ ] **Step 2: Run the test file to confirm it fails on import**

Run: `.venv/bin/pytest tests/runners/test_shuffled_baseline.py -v`
Expected: `ModuleNotFoundError: No module named 'choicebench.methods.library.shuffled_baseline'` (collection error).

- [ ] **Step 3: Implement the runner**

Create `src/choicebench/methods/library/shuffled_baseline.py`:

```python
# src/choicebench/methods/library/shuffled_baseline.py

"""
Method: Shuffled Baseline
--------------------------
Description: Randomly permutes the option ordering for a question once
(seeded by the run seed and question_id, so the permutation is fixed across
repeated samples of the same question but reproducible across runs), then
issues a single direct-ask prompt against the shuffled order. The parsed
letter is mapped back from shuffled position to the canonical option
ordering before scoring. Unlike cyclic_permutation (N calls, majority vote
across every rotation), this makes exactly one backend call per question —
it isolates what a single random reordering does to the model's answer,
rather than averaging it out.
Reference: n/a (reference/demo method for the extensibility walkthrough;
mechanically the single-draw case of the reordering Zheng et al. exhaustively
enumerate in cyclic permutation, ICLR 2024, arXiv:2309.03882).
Backend requirements: generate only
Logprob support required: no
Calls per question: 1
"""

from __future__ import annotations

import random
from typing import Any

from choicebench.methods.base import ExperimentRunner
from choicebench.parsing.types import ParseResult
from choicebench.pipeline.prompt_builder import build_direct_mcq_prompt


class ShuffledBaselineRunner(ExperimentRunner):
    """Runner for the shuffled-baseline condition.

    Same single-call shape as DirectMCQRunner, but the option order is
    shuffled once per question before rendering. The shuffle is seeded from
    (self.seed, question_id) so run_one() is deterministic and reproducible;
    the parsed letter is looked up in the shuffle's label map to recover the
    canonical letter before _score() runs.
    """

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        canonical_options = self._build_options(question_row)
        rng = random.Random(f"{self.seed}:{question_row['question_id']}")
        shuffled_options, label_map = self._shuffle_options(canonical_options, rng)

        prompt = build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=shuffled_options,
        )
        model_response = self._call_backend_generate(prompt)

        parsed_result = None
        score_result = None

        if model_response.is_success():
            # Parse against shuffled_options — the letters and text the model
            # actually saw. Parsing against canonical_options here would pair
            # the wrong text with each letter wherever the shuffle moved it.
            shuffled_parse = self._parse(model_response.raw_text, shuffled_options)
            canonical_choice = (
                label_map[shuffled_parse.final_choice]
                if shuffled_parse.final_choice is not None
                else None
            )
            parsed_result = ParseResult(
                final_choice=canonical_choice,
                status=shuffled_parse.status,
                raw_text=shuffled_parse.raw_text,
                normalized_text=shuffled_parse.normalized_text,
                reason=shuffled_parse.reason,
            )
            score_result = self._score(parsed_result, question_row["correct_option"])

        return self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            model_response=model_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )

    @staticmethod
    def _shuffle_options(
            canonical_options: dict[str, str],
            rng: random.Random,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Permute option text across the canonical letter slots.

        Args:
            canonical_options: Canonical letter-to-text mapping, in order.
            rng: Seeded Random instance controlling the permutation.

        Returns:
            Tuple of (shuffled_options, label_map):
              - shuffled_options: the same letters, with text permuted — this
                is what gets rendered into the prompt.
              - label_map: shuffled letter -> canonical letter, so a letter
                parsed from the shuffled prompt can be mapped back to score.
        """
        labels = list(canonical_options.keys())
        order = list(range(len(labels)))
        rng.shuffle(order)
        shuffled_options = {
            labels[i]: canonical_options[labels[order[i]]] for i in range(len(labels))
        }
        label_map = {labels[i]: labels[order[i]] for i in range(len(labels))}
        return shuffled_options, label_map
```

- [ ] **Step 4: Register the method**

In `src/choicebench/methods/library/__init__.py`, add the import and export:

```python
from choicebench.methods.direct_mcq import DirectMCQRunner
from choicebench.methods.library.cyclic_logprob import CyclicLogprobRunner
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.methods.library.pride import PriDeRunner
from choicebench.methods.library.shuffled_baseline import ShuffledBaselineRunner
from choicebench.methods.library.two_stage import TwoStageRunner

__all__ = [
    "CyclicLogprobRunner",
    "DirectMCQRunner",
    "PermutationRunner",
    "PriDeRunner",
    "ShuffledBaselineRunner",
    "TwoStageRunner",
]
```

In `src/choicebench/methods/__init__.py`, add the same import/export pair (mirror the existing five entries).

In `src/choicebench/registry.py`, add the import and a `METHOD_REGISTRY` entry:

```python
from choicebench.methods import (
    CyclicLogprobRunner,
    DirectMCQRunner,
    PermutationRunner,
    PriDeRunner,
    ShuffledBaselineRunner,
    TwoStageRunner,
)

METHOD_REGISTRY: dict[str, type] = {
    "direct_mcq": DirectMCQRunner,
    "cyclic_permutation": PermutationRunner,
    "cyclic_logprob": CyclicLogprobRunner,
    "two_stage": TwoStageRunner,
    "pride": PriDeRunner,
    "shuffled_baseline": ShuffledBaselineRunner,
}
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `.venv/bin/pytest tests/runners/test_shuffled_baseline.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/choicebench/methods/library/shuffled_baseline.py \
        src/choicebench/methods/library/__init__.py \
        src/choicebench/methods/__init__.py \
        src/choicebench/registry.py \
        tests/runners/test_shuffled_baseline.py
git commit -m "feat(methods): add shuffled_baseline demo method"
```

---

### Task 2: Build the example config and run it for real

**Files:**
- Create: `examples/method_comparison.yaml`
- Output (not committed as source, only referenced): `runs/method_comparison_demo/`, `reports/method_comparison_demo_metrics.json`

**Interfaces:**
- Consumes: `shuffled_baseline` and `direct_mcq` from `METHOD_REGISTRY` (Task 1), the bundled `data/processed/toy_normalized.csv`.
- Produces: real captured terminal output to paste verbatim into `examples/method_comparison.md` (Task 3).

- [ ] **Step 1: Confirm the toy dataset exists (regenerate if not)**

Run: `ls data/processed/toy_normalized.csv || .venv/bin/python scripts/prepare_toy_data.py`
Expected: the file exists (10 rows) before proceeding — if it had to be generated, expect `Wrote 10 rows to data/processed/toy_normalized.csv`.

- [ ] **Step 2: Write the example config**

Create `examples/method_comparison.yaml`, modeled on `config/toy_experiment.yaml` and `config/synthetic_verify.yaml` but naming both methods under comparison:

```yaml
# examples/method_comparison.yaml
#
# Companion config for examples/method_comparison.md. Runs the new
# shuffled_baseline demo method head-to-head with the direct_mcq baseline on
# the bundled toy dataset, using DummyBackend so it needs no GPU or API key.

experiment:
  name: method_comparison_demo

models:
  - backend: dummy
    model_name_or_path: dummy_1
    device: cpu
    generation_kwargs:
      max_new_tokens: 16
      temperature: 0.0
      do_sample: false

benchmarks:
  - name: toy
    split: test
    n_samples: 10
    subject_filter: null

methods:
  - name: direct_mcq
  - name: shuffled_baseline

metrics:
  - accuracy
  - mad

run:
  seed: 42
  resume: false
  dry_run: false
  checkpoint_every_n: 50
  prompt_version: "v1"
  concurrency_limit: 10
```

- [ ] **Step 3: Run the experiment**

Run: `.venv/bin/python scripts/run_experiment.py --config examples/method_comparison.yaml --run-id method_comparison_demo --yes`
Record the full stdout verbatim (needed for Task 3). If it fails, diagnose against the real cause before touching the walkthrough text — do not paper over a real error with invented output.

- [ ] **Step 4: Evaluate the run**

Run: `.venv/bin/python scripts/evaluate_run.py --run-id method_comparison_demo`
Record the full stdout verbatim, and read the resulting `reports/method_comparison_demo_metrics.json`.

- [ ] **Step 5: Sanity-check the captured numbers**

Confirm in the JSON: both `direct_mcq` and `shuffled_baseline` have entries under `dummy_1` → `toy`, each with `accuracy`, `accuracy_ci_low/high`, `accuracy_conditional(_ci_low/high)`, and `mad`/`mad_std`. Note whether the two methods' accuracy/MAD differ — DummyBackend always returns `"The answer is A."`, so `shuffled_baseline`'s accuracy should track how often the shuffle happens to leave (or moves) the correct answer's text such that the fixed response still parses as correct via the letter or text-match path; this is expected to differ from direct_mcq's fixed 3/10 and is the actual point of the demo (a fixed model response is not fixed-answer once you reorder the options). Write down the real observed numbers — do not guess them for Task 3.

---

### Task 3: Write `examples/method_comparison.md`

**Files:**
- Create: `examples/method_comparison.md`

**Interfaces:**
- Consumes: the runner source from Task 1, the config from Task 2, and the real captured output from Task 2 Steps 3–5.

- [ ] **Step 1: Draft the doc using only verified content**

Write `examples/method_comparison.md` with these sections, in order:

1. **What we're building and why** (2-3 sentences) — comparing a new method against `direct_mcq` on the same questions is the framework's native motion; `shuffled_baseline` is the demo method.
2. **The method implementation** — the full contents of `src/choicebench/methods/library/shuffled_baseline.py` as a code block, with 1-2 sentences before it on what `run_one`'s contract requires and what `ExperimentRunner` handles for you (backend call wrapping via `_call_backend_generate`, result-row assembly via `_build_result_row`, checkpointing/resume in `run_experiment.py`, not the method's concern).
3. **Registration** — the three edits from Task 1 Step 4 (registry.py + library/__init__.py), plus a short note that an out-of-repo package only needs `name: my_package.methods:ShuffledBaselineRunner` in YAML and none of those edits.
4. **The experiment config** — the real contents of `examples/method_comparison.yaml`.
5. **Running it** — the two real commands from Task 2 Steps 3-4, each followed by its real (truncated if long — keep the meaningful lines) captured output.
6. **Reading the results** — the real `reports/method_comparison_demo_metrics.json` contents (or the relevant excerpt), one paragraph on what changed between the two methods and why, and a short note distinguishing the `mad` metric (marginal answer-letter skew) from causal order-sensitivity — pointing at the `order_sensitivity` metric for the latter, and noting it currently only populates for `cyclic_logprob`.
7. **Where to go next** — point at `scripts/prepare_data.py --hf-path cais/mmlu --hf-subset all` to run against real MMLU, and at `examples/hpc/` for running the same config shape on Slurm.

Match the README's tone: direct, no marketing language. Every code/command block must be copy-pasteable and exactly what was run/read in Task 1/2 — no paraphrasing of output.

- [ ] **Step 2: Verify every code block against its source**

For each code block in the new doc, diff it against the real file it claims to reproduce (`shuffled_baseline.py`, `method_comparison.yaml`, the registry edits, the metrics JSON). They must match verbatim (output blocks may be truncated with a `...` marker, but never altered).

---

### Task 4: Update README and CHANGELOG

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add an examples pointer to README**

In `README.md`'s "Extending the Framework" section (after the "External registration via..." block, before the `---` / `## Architecture` heading around line 311-313), add:

```markdown
### Worked example

`examples/method_comparison.md` walks through adding a new method
(`shuffled_baseline`) end-to-end — implementation, registration, a real
experiment config, and a real run against the toy dataset compared with
`direct_mcq`.
```

- [ ] **Step 2: Add a CHANGELOG Unreleased section**

In `CHANGELOG.md`, insert above the existing `## v0.1.1 — 2026-07-04` heading:

```markdown
## Unreleased

- Added `shuffled_baseline`, a demo evaluation method (single-call, seeded
  per-question option shuffle) used as the subject of a new extensibility
  walkthrough, `examples/method_comparison.md`.

```

- [ ] **Step 3: Commit**

```bash
git add README.md CHANGELOG.md examples/method_comparison.md examples/method_comparison.yaml
git commit -m "docs: add method_comparison extensibility walkthrough"
```

---

### Task 5: Full regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/pytest`
Expected: all tests pass, including the new `tests/runners/test_shuffled_baseline.py` and every pre-existing test (no regressions from the registry/`__init__.py` edits).

- [ ] **Step 2: Report**

Confirm pass/fail counts before/after in the session summary. If anything regressed, fix it as part of this task before declaring the plan complete (do not leave a broken suite for a later task).
