"""Section H: text extraction + visible LLM matcher.

Frozen invariant: text_extraction makes N target-model calls (visible
options, free-text answer, cascade-matched back); visible_llm_matcher
reuses a prior text_extraction artifact's extracted text and makes zero
NEW text-extraction calls (its own LLM-match call is separately accounted
for, see section S).
"""

from __future__ import annotations

import pytest

from choicebench.methods.library.text_extraction import TextExtractionRunner
from choicebench.methods.library.visible_llm_matcher import VisibleLLMMatcherRunner

from fakes.fixtures import q_n3_primary, q_n4_paris, q_n5_wide
from fakes.spy_backend import SpyBackend


def _text_extraction_embed_fn(texts: list[str]):
    import numpy as np
    return np.zeros((len(texts), 2))


@pytest.mark.parametrize(
    "question_row,expected_n_calls",
    [(q_n4_paris, 4), (q_n3_primary, 3), (q_n5_wide, 5)],
    ids=["n4", "n3", "n5"],
)
def test_text_extraction_rotations_call_count(question_row, expected_n_calls, base_runner_kwargs):
    """H1/H2/H3: text_extraction over N options makes exactly N target-model
    calls (the rotations variant, which is the one section S's arithmetic
    and the rest of this section build on)."""
    spy = SpyBackend(default_text="Paris")
    runner = TextExtractionRunner(backend=spy, method_name="text_extraction", embed_fn=_text_extraction_embed_fn, **base_runner_kwargs)
    runner.run_rotations(question_row, sample_index=0)
    assert spy.call_count == expected_n_calls


def test_text_extraction_run_one_is_single_call(base_runner_kwargs):
    """The plain (non-rotation) run_one() path makes exactly 1 call,
    distinct from the N-call rotations path above -- both shapes matter for
    section S's arithmetic."""
    spy = SpyBackend(default_text="Paris")
    runner = TextExtractionRunner(backend=spy, method_name="text_extraction", embed_fn=_text_extraction_embed_fn, **base_runner_kwargs)
    runner.run_one(q_n4_paris, sample_index=0)
    assert spy.call_count == 1


def test_visible_llm_matcher_makes_zero_new_text_extraction_calls(base_runner_kwargs):
    """H4: given a question_row already carrying extracted_text (as if
    joined from a prior text_extraction run), VisibleLLMMatcherRunner.run_one
    makes exactly 1 NEW call (its own LLM-match call) -- zero of that call
    is a text_extraction call; text_extraction itself is never re-invoked."""
    row_with_extracted_text = dict(q_n4_paris)
    row_with_extracted_text["extracted_text"] = "Paris"

    spy = SpyBackend(default_text="The answer is A.")
    runner = VisibleLLMMatcherRunner(backend=spy, method_name="visible_llm_matcher", **base_runner_kwargs)
    row = runner.run_one(row_with_extracted_text, sample_index=0)

    assert spy.call_count == 1
    assert row["reused_extracted_text"] == "Paris"


def test_visible_llm_matcher_missing_extracted_text_fails_loudly(base_runner_kwargs):
    """Precondition guard: VisibleLLMMatcherRunner must not silently invent
    an extracted_text value if the upstream join was never done."""
    spy = SpyBackend()
    runner = VisibleLLMMatcherRunner(backend=spy, method_name="visible_llm_matcher", **base_runner_kwargs)
    with pytest.raises(KeyError):
        runner.run_one(dict(q_n4_paris), sample_index=0)
    assert spy.call_count == 0


def test_visible_llm_matcher_rotations_consume_same_rotation_extracted_text(base_runner_kwargs):
    """H5: each matcher invocation must consume the SAME-ROTATION
    text_extraction artifact -- proved by tagging each rotation's extracted
    text distinctively and checking rotation r's prompt contains ONLY
    rotation r's own tag."""
    tags = [f"rot{i}_paris_answer" for i in range(4)]
    spy = SpyBackend(default_text="The answer is A.")
    runner = VisibleLLMMatcherRunner(backend=spy, method_name="visible_llm_matcher", **base_runner_kwargs)
    runner.run_matching_rotations(q_n4_paris, per_rotation_extracted_text=tags, sample_index=0)

    assert spy.call_count == 4
    for i, call in enumerate(spy.calls):
        assert tags[i] in call.prompt
        for j, other_tag in enumerate(tags):
            if j != i:
                assert other_tag not in call.prompt


def test_trace_length_mismatch_fails_loudly(base_runner_kwargs):
    """H6: trace length != n_choices must fail loudly, not silently zip/truncate.

    FIXED (verified against commit 651137768dcad640de28f124cff3d50837fe7d7c):
    run_matching_rotations now checks len(per_rotation_extracted_text) ==
    len(rotations) before the zip and raises ValueError on mismatch.
    Previously xfail."""
    spy = SpyBackend(default_text="The answer is A.")
    runner = VisibleLLMMatcherRunner(backend=spy, method_name="visible_llm_matcher", **base_runner_kwargs)
    short_trace = ["only_three", "entries", "here"]  # q_n4_paris has 4 options
    with pytest.raises(ValueError):
        runner.run_matching_rotations(q_n4_paris, per_rotation_extracted_text=short_trace, sample_index=0)


def test_cross_model_extracted_text_is_rejected(base_runner_kwargs):
    """H7: a wrong-model producer artifact must not silently pass, where
    source identity is available.

    FIXED (verified against commit 651137768dcad640de28f124cff3d50837fe7d7c):
    VisibleLLMMatcherRunner._validate_extracted_text_source checks the
    OPTIONAL extracted_text_source_model/_benchmark/_method fields, when
    present, against this run's own identity. Previously xfail."""
    row = dict(q_n4_paris)
    row["extracted_text"] = "Paris"
    row["extracted_text_source_model"] = "some-other-model-entirely"
    spy = SpyBackend(default_text="The answer is A.")
    runner = VisibleLLMMatcherRunner(backend=spy, method_name="visible_llm_matcher", **base_runner_kwargs)
    with pytest.raises(ValueError):
        runner.run_one(row, sample_index=0)


def test_cross_benchmark_extracted_text_is_rejected(base_runner_kwargs):
    """H7 (benchmark half): extracted_text_source_benchmark mismatch is
    also rejected, exercising the second branch the fix added."""
    row = dict(q_n4_paris)
    row["extracted_text"] = "Paris"
    row["extracted_text_source_benchmark"] = "some_other_benchmark"
    spy = SpyBackend(default_text="The answer is A.")
    runner = VisibleLLMMatcherRunner(backend=spy, method_name="visible_llm_matcher", **base_runner_kwargs)
    with pytest.raises(ValueError):
        runner.run_one(row, sample_index=0)


def test_extracted_text_without_source_fields_is_unvalidated_backward_compat(base_runner_kwargs):
    """The source-identity fields are optional/additive: a question_row
    predating this fix (no extracted_text_source_* fields at all) must
    still run exactly as before -- this is the fix's own stated backward-
    compatibility guarantee, not just an absence-of-crash check."""
    row = dict(q_n4_paris)
    row["extracted_text"] = "Paris"
    spy = SpyBackend(default_text="The answer is A.")
    runner = VisibleLLMMatcherRunner(backend=spy, method_name="visible_llm_matcher", **base_runner_kwargs)
    result = runner.run_one(row, sample_index=0)
    assert spy.call_count == 1
    assert result["reused_extracted_text"] == "Paris"
