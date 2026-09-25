"""Section E: permutation / source_index semantics.

Frozen invariant: displayed labels change under rotation, but canonical
source_index identity never does; a model that tracks semantic content
across rotations produces flip_rate == 0, while a model that tracks display
position produces a flip for every rotation that moves the correct option.
"""

from __future__ import annotations

import json
import re

import pytest

from choicebench.methods.library.permutation import PermutationRunner
from choicebench.pipeline.prompt_builder import build_rotations

from fakes.fixtures import (
    N4_TEXTS,
    q_n3_primary,
    q_n3_rotation_oracle,
    q_n4_paris,
    q_n4_paris_rotation_oracle,
    q_n5_rotation_oracle,
    q_n5_wide,
)
from fakes.spy_backend import SpyBackend

_OPTION_LINE_RE = re.compile(r"^([A-Z])\. (.+)$", re.MULTILINE)


def _letter_for_text(prompt: str, target_text: str) -> str:
    """Scan a rendered options block for the display letter holding
    target_text -- test-side scaffolding, never production logic."""
    for letter, text in _OPTION_LINE_RE.findall(prompt):
        if text.strip() == target_text:
            return letter
    raise AssertionError(f"text {target_text!r} not found in prompt:\n{prompt}")


@pytest.mark.parametrize(
    "question_row,oracle",
    [
        (q_n4_paris, q_n4_paris_rotation_oracle),
        (q_n3_primary, q_n3_rotation_oracle),
        (q_n5_wide, q_n5_rotation_oracle),
    ],
    ids=["n4", "n3", "n5"],
)
def test_rotation_mapping_matches_independent_oracle(question_row, oracle):
    """E1/E4/E5: build_rotations()'s displayed mapping matches the
    independently hand-computed oracle table, and canonical source_index
    identity is stable across every rotation of the same question."""
    from choicebench.methods.base import ExperimentRunner

    canonical_options = ExperimentRunner._build_options(question_row)
    label_to_source_index = ExperimentRunner._build_label_to_source_index(question_row)
    rotations = build_rotations(canonical_options)

    assert len(rotations) == len(oracle)
    for rotation, expected_mapping in zip(rotations, oracle):
        assert rotation.mapping == expected_mapping

    # source_index identity is a property of the question, not the rotation:
    # re-deriving it after rotating changes nothing (it's never re-derived
    # from the rotated mapping in the first place).
    for _ in rotations:
        assert ExperimentRunner._build_label_to_source_index(question_row) == label_to_source_index


def test_semantic_tracking_yields_zero_flip_rate(base_runner_kwargs):
    """E2: a model that answers the same semantic option ("Paris") under
    every rotation produces identical canonical predictions across
    rotations -- flip rate 0."""
    def responder(prompt: str) -> str:
        letter = _letter_for_text(prompt, "Paris")
        return f"The answer is {letter}."

    spy = SpyBackend(responder=responder)
    runner = PermutationRunner(backend=spy, method_name="cyclic_permutation", **base_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)

    per_rotation = json.loads(row["per_rotation_choices_json"])
    assert per_rotation == ["A", "A", "A", "A"]
    n_flips = sum(1 for choice in per_rotation if choice != per_rotation[0])
    assert n_flips == 0


def test_positional_tracking_yields_flips(base_runner_kwargs):
    """E3: a model that always answers displayed letter "A" (positional,
    not semantic) tracks a DIFFERENT canonical option under every rotation
    of a 4-option question with all-distinct option texts -- the resulting
    canonical choices vary, and the hand-computed flip count against
    rotation 0's own answer is non-zero."""
    spy = SpyBackend(default_text="The answer is A.")
    runner = PermutationRunner(backend=spy, method_name="cyclic_permutation", **base_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)

    per_rotation = json.loads(row["per_rotation_choices_json"])
    # Rotation r's displayed "A" is canonical text N4_TEXTS[(0 + r) % 4], so
    # the canonical LETTER "always displayed-A" resolves to is letter
    # chr(ord('A') + r) for rotation r (canonical position r -> letter r).
    expected = [chr(ord("A") + r) for r in range(4)]
    assert per_rotation == expected
    n_flips_vs_rotation0 = sum(1 for c in per_rotation if c != per_rotation[0])
    assert n_flips_vs_rotation0 == 3  # rotations 1, 2, 3 disagree with rotation 0


def test_rotation_never_mutates_gold_canonical_fields():
    """E6: changing only display labels/order must never mutate the row's
    own gold canonical source_index / correct_option / correct_index."""
    from choicebench.methods.base import ExperimentRunner

    before = dict(q_n4_paris)
    canonical_options = ExperimentRunner._build_options(q_n4_paris)
    _ = build_rotations(canonical_options)
    _ = build_rotations(canonical_options)
    assert q_n4_paris["correct_option"] == before["correct_option"]
    assert q_n4_paris["correct_index"] == before["correct_index"]
    assert q_n4_paris["choices_json"] == before["choices_json"]


def test_rotation_processing_order_does_not_affect_semantic_result(base_runner_kwargs):
    """E7: processing rotations in a different execution order must produce
    the same canonical semantic result. Scripted by CONTENT (which text is
    shown), not by call order, so reversing rotation order cannot change
    what each individual call returns -- only the order results arrive in."""
    def responder(prompt: str) -> str:
        letter = _letter_for_text(prompt, "Paris")
        return f"The answer is {letter}."

    spy_forward = SpyBackend(responder=responder)
    runner_forward = PermutationRunner(
        backend=spy_forward, method_name="cyclic_permutation", **base_runner_kwargs,
    )
    row_forward = runner_forward.run_one(q_n4_paris, sample_index=0)

    # PermutationRunner.run_one() always issues rotations in forward order
    # internally; to prove order-independence of the SEMANTIC result (not
    # just re-run the same code path twice), directly exercise the module's
    # own rotation list in reverse and re-derive the majority vote by hand
    # using the same content-keyed responder and the shared tiebreak utility
    # PermutationRunner itself uses.
    from choicebench.scoring.tiebreak import majority_vote_with_tiebreak

    canonical_options = runner_forward._build_options(q_n4_paris)
    canonical_letters = list(canonical_options.keys())
    rotations = build_rotations(canonical_options)

    reversed_choices: list[str | None] = []
    for rotation in reversed(rotations):
        prompt = f"Q\n{chr(10).join(f'{k}. {v}' for k, v in rotation.mapping.items())}"
        letter = _letter_for_text(prompt, "Paris")
        display_index = canonical_letters.index(letter)
        reversed_choices.append(canonical_letters[rotation.slot_to_canonical[display_index]])
    reversed_choices.reverse()  # back to rotation-index order for comparison

    forward_choices = json.loads(row_forward["per_rotation_choices_json"])
    assert set(reversed_choices) == set(forward_choices)

    voted_reversed = majority_vote_with_tiebreak(
        list(reversed(reversed_choices)),
        label_to_source_index=runner_forward._build_label_to_source_index(q_n4_paris),
        seed=42, benchmark_id="diag_bench", question_id=q_n4_paris["question_id"],
        method_name="cyclic_permutation",
    )
    assert voted_reversed == row_forward["parsed_choice"]
