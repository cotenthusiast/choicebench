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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "VisibleLLMMatcherRunner.run_matching_rotations zips "
        "rotations with per_rotation_extracted_text without checking their "
        "lengths match -- a length mismatch silently truncates to the "
        "shorter length instead of raising (see visible_llm_matcher.py's "
        "zip(rotations, per_rotation_extracted_text) call)."
    ),
)
def test_trace_length_mismatch_fails_loudly(base_runner_kwargs):
    """H6: trace length != n_choices must fail loudly, not silently zip/truncate."""
    spy = SpyBackend(default_text="The answer is A.")
    runner = VisibleLLMMatcherRunner(backend=spy, method_name="visible_llm_matcher", **base_runner_kwargs)
    short_trace = ["only_three", "entries", "here"]  # q_n4_paris has 4 options
    with pytest.raises(ValueError):
        runner.run_matching_rotations(q_n4_paris, per_rotation_extracted_text=short_trace, sample_index=0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Neither TextExtractionRunner nor VisibleLLMMatcherRunner validates "
        "that question_row['extracted_text'] actually came from the same "
        "model/benchmark/method this run is about to use -- confirmed by "
        "reading both modules, no such validation code exists."
    ),
)
def test_cross_model_extracted_text_is_rejected(base_runner_kwargs):
    """H7: a wrong-model/-benchmark/-method producer artifact must not
    silently pass, where source identity is available."""
    row = dict(q_n4_paris)
    row["extracted_text"] = "Paris"
    row["extracted_text_source_model"] = "some-other-model-entirely"
    spy = SpyBackend(default_text="The answer is A.")
    runner = VisibleLLMMatcherRunner(backend=spy, method_name="visible_llm_matcher", **base_runner_kwargs)
    with pytest.raises(ValueError):
        runner.run_one(row, sample_index=0)
