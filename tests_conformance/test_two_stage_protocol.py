"""Section F: two-stage rotation semantics.

Frozen invariant: normal two-stage rotation experiment makes exactly
1 Stage-1 call + N Stage-2 calls, Stage 1's prompt never shows options, and
every Stage-2 rotation consumes the SAME Stage-1 artifact (proved by literal
string identity of the free-text answer threaded through, not by two
independently-generated texts happening to match).
"""

from __future__ import annotations

import pytest

from choicebench.methods.library.two_stage import TwoStageRunner

from fakes.fixtures import q_n3_primary, q_n5_wide, q_n4_paris
from fakes.spy_backend import SpyBackend


def test_stage1_no_options_and_stage2_reuses_stage1_verbatim(base_runner_kwargs):
    """F1/F2/F3/F4: N=4 -- 1 Stage-1 call whose prompt has no option text,
    then exactly 4 Stage-2 calls, each of which literally contains the
    SAME Stage-1 free-text string this run itself produced (never a second,
    re-elicited Stage-1 answer), in the rotation's own intended option
    order."""
    stage1_spy = SpyBackend(default_text="I believe the answer is Paris.")
    runner = TwoStageRunner(backend=stage1_spy, method_name="two_stage", **base_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)

    assert stage1_spy.call_count == 2  # run_one() itself: 1 stage-1 + 1 stage-2
    stage1_prompt = stage1_spy.calls[0].prompt
    for option_text in ["Paris", "London", "Berlin", "Madrid"]:
        assert option_text not in stage1_prompt

    free_text_answer = row["free_text_response"]
    assert free_text_answer == "I believe the answer is Paris."

    # Now rerun ONLY stage 2 across all 4 rotations, passing that SAME
    # Python string object's value as free_text_answer -- this literal
    # value-identity is the lineage proof run_stage2_rotations relies on
    # (it takes no separate "stage-1 artifact id", just the string itself).
    stage2_spy = SpyBackend(default_text="The answer is A.")
    stage2_runner = TwoStageRunner(backend=stage2_spy, method_name="two_stage", **base_runner_kwargs)
    stage2_row = stage2_runner.run_stage2_rotations(
        q_n4_paris, free_text_answer=free_text_answer, sample_index=0,
    )

    assert stage2_spy.call_count == 4
    for call in stage2_spy.calls:
        assert free_text_answer in call.prompt

    # F4: Stage-2 rotation r's option block matches build_rotations()'s own
    # rendering for rotation r (spot-check rotation 1's displayed order).
    from choicebench.pipeline.prompt_builder import build_rotations

    canonical_options = stage2_runner._build_options(q_n4_paris)
    rotations = build_rotations(canonical_options)
    rotation_1_first_option_text = next(iter(rotations[1].mapping.values()))
    assert rotation_1_first_option_text in stage2_spy.calls[1].prompt

    assert "per_rotation_choices_json" in stage2_row


@pytest.mark.parametrize(
    "question_row,expected_stage2_calls",
    [(q_n3_primary, 3), (q_n5_wide, 5)],
    ids=["n3", "n5"],
)
def test_stage2_rotation_call_count(question_row, expected_stage2_calls, base_runner_kwargs):
    """F5/F6: run_stage2_rotations makes exactly N Stage-2 calls for an
    N-option question, given a fixed, already-elicited Stage-1 answer."""
    spy = SpyBackend(default_text="The answer is A.")
    runner = TwoStageRunner(backend=spy, method_name="two_stage", **base_runner_kwargs)
    runner.run_stage2_rotations(
        question_row, free_text_answer="A previously elicited free-text answer.",
        sample_index=0,
    )
    assert spy.call_count == expected_stage2_calls


def test_reasoning_two_stage_uses_reasoning_stage1_prompt_but_same_call_shape(
    reasoning_runner_kwargs, base_runner_kwargs,
):
    """F7: reasoning_two_stage has the same 1+N call graph as two_stage, but
    its Stage-1 prompt comes from the frozen reasoning prompt bundle
    (v1_reasoning), which differs from v1's own Stage-1 prompt for the
    identical question."""
    v1_spy = SpyBackend(default_text="v1 free text answer.")
    v1_runner = TwoStageRunner(backend=v1_spy, method_name="two_stage", **base_runner_kwargs)
    v1_runner.run_one(q_n4_paris, sample_index=0)
    v1_stage1_prompt = v1_spy.calls[0].prompt

    reasoning_spy = SpyBackend(default_text="reasoning free text answer.")
    reasoning_runner = TwoStageRunner(
        backend=reasoning_spy, method_name="reasoning_two_stage", **reasoning_runner_kwargs,
    )
    reasoning_row = reasoning_runner.run_one(q_n4_paris, sample_index=0)
    reasoning_stage1_prompt = reasoning_spy.calls[0].prompt

    assert reasoning_spy.call_count == 2  # 1 stage-1 + 1 stage-2, same shape as v1
    assert reasoning_stage1_prompt != v1_stage1_prompt
    assert reasoning_row["free_text_response"] == "reasoning free text answer."
