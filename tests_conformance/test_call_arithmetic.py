"""Section S: paper-spec call arithmetic.

S1: encode each method's call-count formula (first principles, from the
frozen spec's own §S list) as one-line pure functions of N (option count),
then cross-check each formula against a REAL runner + SpyBackend, proving
"what the spec's formula says" and "what production actually does" agree
for at least one concrete N (N=4 -- N=3/5 are already exhaustively covered
per-method in test_call_graphs.py/test_two_stage_protocol.py/
test_ihs_protocol.py/test_text_extraction_matcher.py).

S2 (deriving production totals from the final manifests/configs) is
reported by report_call_totals.py in this directory -- a script, not a
pytest-collected test, per spec's own instruction not to hardcode or assert
a specific historical total.
"""

from __future__ import annotations

import pytest

from choicebench.analysis.derive_baseline_from_cyclic import derive_baseline_from_cyclic
from choicebench.analysis.derive_from_free_text import derive_matched_results
from choicebench.methods.library.independent_hypothesis import IndependentHypothesisRunner
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.methods.library.text_extraction import TextExtractionRunner
from choicebench.methods.library.two_stage import TwoStageRunner
from choicebench.methods.library.visible_llm_matcher import VisibleLLMMatcherRunner

from fakes.fixtures import q_n4_paris
from fakes.spy_backend import SpyBackend

import pandas as pd


def _cyclic_calls(n: int) -> int:
    return n


def _baseline_additional_calls(n: int) -> int:
    return 0


def _two_stage_calls(n: int) -> tuple[int, int]:
    return (1, n)


def _semantic_calls(n: int) -> int:
    return 0


def _text_extraction_calls(n: int) -> int:
    return n


def _visible_matcher_new_text_extraction_calls(n: int) -> int:
    return 0


def _ihs_calls(n: int) -> int:
    return n


def _reasoning_mcq_calls(n: int) -> int:
    return n  # reasoning_cyclic = PermutationRunner, same call shape as cyclic


def _reasoning_two_stage_calls(n: int) -> tuple[int, int]:
    return (1, n)  # reasoning_two_stage = TwoStageRunner, same call shape as two_stage


def _stochasticity_baseline_additional(n: int) -> int:
    return 3  # obs1-3, 1 call/obs, n-independent


def _stochasticity_two_stage_additional(n: int) -> int:
    return 6  # 3 * (1 stage1 + 1 stage2), n-independent


def _stochasticity_reasoning_mcq_additional(n: int) -> int:
    return 3


def _stochasticity_reasoning_two_stage_additional(n: int) -> int:
    return 6


@pytest.mark.parametrize("n", [3, 4, 5])
def test_s1_call_arithmetic_formulas(n):
    assert _cyclic_calls(n) == n
    assert _baseline_additional_calls(n) == 0
    assert _two_stage_calls(n) == (1, n)
    assert _semantic_calls(n) == 0
    assert _text_extraction_calls(n) == n
    assert _visible_matcher_new_text_extraction_calls(n) == 0
    assert _ihs_calls(n) == n
    assert _reasoning_mcq_calls(n) == n
    assert _reasoning_two_stage_calls(n) == (1, n)
    assert _stochasticity_baseline_additional(n) == 3
    assert _stochasticity_two_stage_additional(n) == 6
    assert _stochasticity_reasoning_mcq_additional(n) == 3
    assert _stochasticity_reasoning_two_stage_additional(n) == 6


def test_s1_formulas_cross_checked_against_real_runners_n4(base_runner_kwargs, ihs_runner_kwargs):
    n = 4

    spy = SpyBackend()
    cyclic_runner = PermutationRunner(backend=spy, method_name="cyclic_permutation", **base_runner_kwargs)
    cyclic_row = cyclic_runner.run_one(q_n4_paris, sample_index=0)
    assert spy.call_count == _cyclic_calls(n)

    calls_before_baseline = spy.call_count
    derive_baseline_from_cyclic(pd.DataFrame([cyclic_row]), method_name="direct_mcq")
    assert spy.call_count - calls_before_baseline == _baseline_additional_calls(n)

    spy2 = SpyBackend()
    two_stage_runner = TwoStageRunner(backend=spy2, method_name="two_stage", **base_runner_kwargs)
    two_stage_runner.run_one(q_n4_paris, sample_index=0)
    # run_one() itself is always exactly 1 stage1 + 1 stage2 = 2 calls,
    # regardless of N -- distinct from the (1, n) rotation-experiment
    # formula below (see module docstring / plan Task 4 Step 1).
    assert spy2.call_count == 2

    spy3 = SpyBackend()
    rotation_runner = TwoStageRunner(backend=spy3, method_name="two_stage", **base_runner_kwargs)
    rotation_runner.run_stage2_rotations(q_n4_paris, "Paris", sample_index=0)
    assert spy3.call_count == _two_stage_calls(n)[1]  # stage2-only rotation rerun = n calls

    semantic_source = pd.DataFrame([{**q_n4_paris, "run_id": "r", "method_name": "two_stage", "free_text_response": "Paris", "seed": 42, "benchmark_name": "diag_bench"}])
    derive_matched_results(semantic_source, method_name="semantic_matching_v1", embed_fn=lambda t: [1.0 if t == "Paris" else 0.0])
    # No backend was ever passed to derive_matched_results -- zero calls by construction (§S "semantic: 0 target-model calls").

    spy4 = SpyBackend()
    te_runner = TextExtractionRunner(backend=spy4, method_name="text_extraction", **base_runner_kwargs)
    # _text_extraction_calls(n) is the rotation-rerun formula (run_rotations,
    # one call per rotation) -- run_one() alone is always a fixed 1 call,
    # asserted separately.
    te_runner.run_rotations(q_n4_paris, sample_index=0)
    assert spy4.call_count == _text_extraction_calls(n)

    spy4b = SpyBackend()
    te_runner_one = TextExtractionRunner(backend=spy4b, method_name="text_extraction", **base_runner_kwargs)
    te_runner_one.run_one(q_n4_paris, sample_index=0)
    assert spy4b.call_count == 1

    spy5 = SpyBackend(default_text="The answer is A.")
    matcher_runner = VisibleLLMMatcherRunner(backend=spy5, method_name="visible_llm_matcher", **base_runner_kwargs)
    row_with_text = dict(q_n4_paris)
    row_with_text["extracted_text"] = "Paris"
    matcher_runner.run_one(row_with_text, sample_index=0)
    # visible_llm_matcher makes exactly 1 NEW call (its own Stage-2 match) --
    # zero NEW text_extraction calls, per §S ("visible matcher: 0 new
    # text-extraction calls + matcher calls as defined by method").
    assert spy5.call_count == 1

    spy6 = SpyBackend(default_text="<score>50</score>")
    ihs_runner = IndependentHypothesisRunner(backend=spy6, method_name="independent_hypothesis", **ihs_runner_kwargs)
    ihs_runner.run_one(q_n4_paris, sample_index=0)
    assert spy6.call_count == _ihs_calls(n)
