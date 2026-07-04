# Worked example: adding a method and comparing it against the baseline

This walks through adding a new evaluation method — `shuffled_baseline` — and
running it against `direct_mcq` on the same questions. Comparing a new method
against the baseline this way, on the same benchmark and model, is the
framework's native motion: every method in `src/choicebench/methods/library/`
exists to be measured against `direct_mcq`, not run in isolation.

`shuffled_baseline` randomly permutes a question's option order once (seeded,
so it's reproducible), asks it as a normal direct MCQ prompt, then maps the
model's answer back to the canonical option ordering before scoring. It's a
single backend call per question — the same shape as `direct_mcq` — so the
only thing that changes between the two methods is whether the options are in
their original order or a shuffled one.

## The method implementation

Every method is an `ExperimentRunner` subclass. The only method you're
required to implement is `run_one(question_row, sample_index) -> dict`; the
base class handles backend-call wrapping (`_call_backend_generate`, which
never raises — it turns exceptions into a failure `ModelResponse`), building
the option map from either schema (`_build_options`), and assembling the
final result row (`_build_result_row`, whose schema is load-bearing for
`evaluate_run.py`). `run_experiment.py` handles checkpointing, resume, and
concurrency around whatever `run_one` returns — none of that is this file's
concern.

This is the full, unmodified contents of
`src/choicebench/methods/library/shuffled_baseline.py`:

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

The key thing to notice: `run_one` parses the model's response against
`shuffled_options` (what was actually rendered into the prompt), then looks
the resulting letter up in `label_map` to recover the canonical letter before
scoring. Parsing against the canonical, unshuffled options would pair the
wrong text with each letter wherever the shuffle moved it.

## Registration

Built-in methods register in three places, matching the README's "Add a new
method" steps. For `shuffled_baseline`:

`src/choicebench/methods/library/__init__.py`:

```python
from choicebench.methods.library.shuffled_baseline import ShuffledBaselineRunner

__all__ = [
    # ...
    "ShuffledBaselineRunner",
]
```

`src/choicebench/registry.py`:

```python
from choicebench.methods import ShuffledBaselineRunner

METHOD_REGISTRY: dict[str, type] = {
    # ...
    "shuffled_baseline": ShuffledBaselineRunner,
}
```

`src/choicebench/methods/__init__.py` gets the same import/export pair, so
`from choicebench.methods import ShuffledBaselineRunner` also works.

If this method lived in your own package instead of this repo, none of the
above is needed — skip straight to the config and use
`name: my_package.methods:ShuffledBaselineRunner` in YAML.

## The experiment config

`examples/method_comparison.yaml`, run as-is:

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

## Running it

```
$ python scripts/run_experiment.py --config examples/method_comparison.yaml --run-id method_comparison_demo --yes
15:24:44  INFO      Loaded config: examples/method_comparison.yaml
15:24:44  INFO      Run ID: method_comparison_demo  |  Output: .../runs/method_comparison_demo
15:24:44  INFO      Loaded 10 questions from toy
15:24:44  INFO      ── Benchmark: toy  Method: direct_mcq  Model: dummy_1 (dummy) ──────────────────
15:24:44  INFO      Backend: DummyBackend
15:24:44  INFO      [direct_mcq] Starting fresh: 10 questions
15:24:44  INFO      [direct_mcq] Progress: 10 / 10 questions complete
15:24:44  INFO      [direct_mcq] Done → .../runs/method_comparison_demo/method_comparison_demo_direct_mcq_dummy_1_toy.csv
15:24:44  INFO      ── Benchmark: toy  Method: shuffled_baseline  Model: dummy_1 (dummy) ──────────────────
15:24:44  INFO      Backend: DummyBackend
15:24:44  INFO      [shuffled_baseline] Starting fresh: 10 questions
15:24:44  INFO      [shuffled_baseline] Progress: 10 / 10 questions complete
15:24:44  INFO      [shuffled_baseline] Done → .../runs/method_comparison_demo/method_comparison_demo_shuffled_baseline_dummy_1_toy.csv
15:24:44  INFO      ── Run complete ─────────────────────────────────
15:24:44  INFO        Run ID:     method_comparison_demo
15:24:44  INFO        Results in: .../runs/method_comparison_demo
15:24:44  INFO        Failed jobs: 0
15:24:44  INFO      ─────────────────────────────────────────────────
```

```
$ python scripts/evaluate_run.py --run-id method_comparison_demo
15:24:50  INFO      Loading run: method_comparison_demo
15:24:50  INFO      Loaded 20 rows
15:24:50  INFO      Metrics to compute: ['accuracy', 'mad']
15:24:50  INFO      Metrics saved to .../reports/method_comparison_demo_metrics.json
15:24:50  INFO      ── Eval complete ────────────────────────────────
15:24:50  INFO        [direct_mcq | dummy_1 | toy]  {'accuracy': 0.3, 'accuracy_ci_low': 0.0667, 'accuracy_ci_high': 0.6525, 'accuracy_conditional': 0.3, 'accuracy_conditional_ci_low': 0.0667, 'accuracy_conditional_ci_high': 0.6525, 'mad': 46.6667, 'mad_std': 9.6976}
15:24:50  INFO        [shuffled_baseline | dummy_1 | toy]  {'accuracy': 0.2, 'accuracy_ci_low': 0.0252, 'accuracy_ci_high': 0.5561, 'accuracy_conditional': 0.2, 'accuracy_conditional_ci_low': 0.0252, 'accuracy_conditional_ci_high': 0.5561, 'mad': 20.0, 'mad_std': 7.5855}
15:24:50  INFO      ─────────────────────────────────────────────────
```

## Reading the results

`reports/method_comparison_demo_metrics.json`:

```json
{
    "direct_mcq": {
        "dummy_1": {
            "toy": {
                "accuracy": 0.3,
                "accuracy_ci_low": 0.06673951117773447,
                "accuracy_ci_high": 0.6524528500599973,
                "accuracy_conditional": 0.3,
                "accuracy_conditional_ci_low": 0.06673951117773447,
                "accuracy_conditional_ci_high": 0.6524528500599973,
                "mad": 46.666666666666664,
                "mad_std": 9.697566338462908
            }
        }
    },
    "shuffled_baseline": {
        "dummy_1": {
            "toy": {
                "accuracy": 0.2,
                "accuracy_ci_low": 0.02521072632683336,
                "accuracy_ci_high": 0.5560954623076415,
                "accuracy_conditional": 0.2,
                "accuracy_conditional_ci_low": 0.02521072632683336,
                "accuracy_conditional_ci_high": 0.5560954623076415,
                "mad": 20.0,
                "mad_std": 7.585523037338954
            }
        }
    }
}
```

`DummyBackend` always returns the fixed text `"The answer is A."`, so it
always parses to letter `A` in whatever prompt it's shown — shuffled or not.
`direct_mcq` scores 3/10 because `scripts/prepare_toy_data.py` deliberately
pins exactly 3 of the 10 toy questions' correct answer to canonical option A
(so a fixed "always say A" response gets exactly those 3 right). Under
`shuffled_baseline`, the per-question shuffle moves each question's correct
answer to a different canonical slot, so "always say A" now lands correctly
only on whichever questions happen to have their correct answer shuffled
*into* slot A for this seed — 2 different questions, not the same 3. This is
the point of the demo: a fixed or positionally-biased response pattern isn't
fixed *correctness* once you stop guaranteeing the correct answer sits in the
position the bias favors.

The 10-question toy set is far too small for the accuracy difference itself
to mean anything (look at the confidence intervals — both `[0.03, 0.66]`-ish
and heavily overlapping); what's real and reproducible here is the mechanism,
not the specific 0.3 vs. 0.2 split. The `mad` drop (46.7 → 20.0) is more
informative at even this sample size: `direct_mcq`'s "always A" response
concentrates its predicted-letter mass on a single canonical letter (A),
which the toy set's authored answer distribution doesn't share — that's a
large marginal skew. `shuffled_baseline`'s per-question shuffle spreads that
same "always pick whatever's in slot A" behavior across different canonical
letters, so the predicted-letter distribution looks less skewed relative to
the gold distribution, even though nothing about the model's actual behavior
changed.

Don't over-read `mad` as "less biased," though — it's a marginal
answer-letter-skew indicator (predicted-letter-% vs. correct-letter-% over
the same scored subset), not a causal order-bias measure. It can't tell you
whether reordering a *given* question's options would flip that question's
answer. That's what the `order_sensitivity` metric (`order_rstd`,
`order_flip_rate`) is for — it currently only populates for `cyclic_logprob`,
the one bundled method that persists its per-permutation logprob
distributions (`option_distributions_json`), which is what that metric reads.

## Where to go next

- Run the same comparison against a real benchmark instead of the toy set:
  `python scripts/prepare_data.py --hf-path cais/mmlu --hf-subset all`, then
  point a copy of `examples/method_comparison.yaml` at `benchmarks: - name:
  mmlu` and a real backend/model instead of `dummy`.
- To run the same config shape on a Slurm cluster instead of locally, see
  `examples/hpc/` — `setup_hpc.sh`, `env_hpc.sh`, and `run_choicebench.sbatch`
  are editable templates for exactly this.
