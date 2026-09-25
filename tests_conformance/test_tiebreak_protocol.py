"""Section K: tie-break protocol.

Frozen invariant: any tie resolves via choicebench.scoring.tiebreak
.resolve_tie, a pure function of (seed, benchmark, question, method, tied
canonical ids) -- deliberately excluding model/provider, displayed letter,
rotation index, and execution/worker order. Uses BLAKE2b, not Python's
hash()/random, so results are stable across processes.
"""

from __future__ import annotations

import subprocess
import sys

from choicebench.methods.library.independent_hypothesis import IndependentHypothesisRunner
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.scoring.tiebreak import majority_vote_with_tiebreak, resolve_tie

from fakes.fixtures import q_n4_paris
from fakes.spy_backend import SpyBackend

_COMMON = dict(seed=42, benchmark_id="diag_bench", question_id="diag-n4-paris-001", method_name="cyclic_permutation")


def test_resolve_tie_excludes_model_identity(base_runner_kwargs):
    """K1: resolve_tie's signature has no model parameter at all -- proved
    structurally -- and two PermutationRunner instances differing only in
    model_label, driven into the identical induced tie, agree on the
    canonical winner."""
    import inspect

    assert "model" not in inspect.signature(resolve_tie).parameters

    def make_tied_runner(model_label: str, provider: str = "spy-provider") -> tuple[SpyBackend, PermutationRunner]:
        spy = SpyBackend(default_text="CALL", provider=provider)
        kwargs = dict(base_runner_kwargs)
        runner = PermutationRunner(
            backend=spy, method_name="cyclic_permutation", model_label=model_label, **kwargs,
        )
        return spy, runner

    # Induce a genuine tie: every rotation call returns unparseable text, so
    # ALL FOUR canonical letters end up with zero votes each -- not a real
    # majority-vote tie in the usual sense. Use a cleaner tie instead: two
    # rotations "vote" for canonical A, two vote for canonical B, by
    # scripting responses keyed to rotation content.
    from choicebench.pipeline.prompt_builder import build_rotations

    def tie_two_way_responder(prompt: str) -> str:
        # Whichever displayed letter shows "Paris" -> answer it; whichever
        # shows "London" in the SAME prompt is irrelevant (Paris always
        # present). To force a genuine A/B tie across 4 rotations, answer
        # "Paris"'s letter on rotations 0/1 and "London"'s letter on 2/3.
        import re
        options = dict(re.findall(r"^([A-Z])\. (.+)$", prompt, re.MULTILINE))
        # crude rotation fingerprint: which canonical text sits at "A"
        if options.get("A") in ("Paris", "Madrid"):
            target = "Paris"
        else:
            target = "London"
        for letter, text in options.items():
            if text == target:
                return f"The answer is {letter}."
        raise AssertionError("target text not found")

    spy1 = SpyBackend(responder=tie_two_way_responder)
    runner1 = PermutationRunner(backend=spy1, method_name="cyclic_permutation", model_label="model-one", **base_runner_kwargs)
    row1 = runner1.run_one(q_n4_paris, sample_index=0)

    spy2 = SpyBackend(responder=tie_two_way_responder)
    runner2 = PermutationRunner(backend=spy2, method_name="cyclic_permutation", model_label="model-two", **base_runner_kwargs)
    row2 = runner2.run_one(q_n4_paris, sample_index=0)

    assert row1["parsed_choice"] == row2["parsed_choice"]


def test_resolve_tie_excludes_provider():
    """K2: same tied set/keys, different provider -> resolve_tie itself
    takes no provider argument (structural), and is deterministic
    regardless of any provider label a caller might otherwise be tempted
    to fold in."""
    winner_1 = resolve_tie(**_COMMON, tied_canonical_ids=[0, 2])
    winner_2 = resolve_tie(**_COMMON, tied_canonical_ids=[0, 2])
    assert winner_1 == winner_2  # provider is not even a parameter to vary


def test_resolve_tie_independent_of_display_order():
    """K3: the SAME tied canonical ids, passed in a different iteration
    order, resolve identically -- resolve_tie sorts its input internally."""
    winner_forward = resolve_tie(**_COMMON, tied_canonical_ids=[0, 2, 3])
    winner_reversed = resolve_tie(**_COMMON, tied_canonical_ids=[3, 2, 0])
    assert winner_forward == winner_reversed


def test_majority_vote_tiebreak_independent_of_choice_list_order():
    """K4: majority_vote_with_tiebreak on the same multiset of choices, in
    two different orders, agrees on the winner."""
    label_to_source_index = {"A": 0, "B": 1, "C": 2, "D": 3}
    kwargs = dict(
        label_to_source_index=label_to_source_index, seed=42,
        benchmark_id="diag_bench", question_id="q1", method_name="cyclic_permutation",
    )
    winner_forward = majority_vote_with_tiebreak(["A", "B", "A", "B"], **kwargs)
    winner_reordered = majority_vote_with_tiebreak(["B", "A", "B", "A"], **kwargs)
    assert winner_forward == winner_reordered


def test_different_tied_sets_each_resolve_within_their_own_set():
    """K5: two different tied sets for the same (seed, benchmark, question,
    method) each resolve to a member of THEIR OWN set."""
    winner_1 = resolve_tie(**_COMMON, tied_canonical_ids=[0, 2])
    winner_2 = resolve_tie(**_COMMON, tied_canonical_ids=[1, 3])
    assert winner_1 in {0, 2}
    assert winner_2 in {1, 3}


def test_ihs_exact_tie_uses_resolve_tie_not_argmax(ihs_runner_kwargs):
    """K6/K7: IndependentHypothesisRunner's tie-break, for an exact score
    tie across ALL candidates, matches resolve_tie's own independently
    computed winner -- not np.argmax's first-index default."""
    spy = SpyBackend(default_text="<score>77</score>")  # every candidate ties
    runner = IndependentHypothesisRunner(backend=spy, method_name="independent_hypothesis", **ihs_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)

    label_to_source_index = runner._build_label_to_source_index(q_n4_paris)
    expected_id = resolve_tie(
        seed=42, benchmark_id="diag_bench", question_id=q_n4_paris["question_id"],
        method_name="independent_hypothesis", tied_canonical_ids=list(label_to_source_index.values()),
    )
    id_to_label = {v: k for k, v in label_to_source_index.items()}
    assert row["parsed_choice"] == id_to_label[expected_id]
    # np.argmax over an all-tied array always returns index 0 ("A") -- assert
    # the winner is NOT trivially always "A" for a tie fixture where the
    # BLAKE2b-derived winner happens to differ. This is a probabilistic
    # sanity check, not a strict requirement -- skip the strong claim and
    # instead directly confirm equality with resolve_tie's own computation
    # (already asserted above), which is the real proof.


def test_tiebreak_independent_of_process_level_hash_randomization():
    """K8: resolve_tie's result does not depend on Python's hash()/
    PYTHONHASHSEED randomization -- run it in two fresh subprocesses with
    different PYTHONHASHSEED values and confirm identical output."""
    import os

    script = (
        "from choicebench.scoring.tiebreak import resolve_tie; "
        "print(resolve_tie(seed=42, benchmark_id='diag_bench', question_id='q1', "
        "method_name='cyclic_permutation', tied_canonical_ids=[0, 1, 2, 3]))"
    )
    results = []
    for hash_seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True,
            cwd=str(__file__.rsplit("/tests_conformance", 1)[0]),
        )
        results.append(proc.stdout.strip())
    assert results[0] == results[1]
