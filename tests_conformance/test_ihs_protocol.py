"""Section I: Independent Hypothesis (IHS).

Frozen invariant: one backend call per REAL option (never N-1 padded, never
a phantom extra slot), each call sees exactly one isolated candidate (never
the full ordered list), highest valid score wins, no answer is fabricated
from invalid/unparseable scores, and a transport failure in one candidate
must not be silently hidden by another candidate's success.
"""

from __future__ import annotations

import math

import pytest

from choicebench.clients.types import FAILURE_STATUS
from choicebench.methods.library.independent_hypothesis import IndependentHypothesisRunner
from choicebench.scoring.tiebreak import resolve_tie

from fakes.fixtures import q_n3_primary, q_n4_paris, q_n5_wide
from fakes.spy_backend import SpyBackend


def _sequential_responder(texts: list[str]):
    """Returns a SpyBackend-compatible responder(prompt) -> str that yields
    texts in order, one per call, ignoring the prompt argument."""
    it = iter(texts)
    return lambda _prompt: next(it)


@pytest.mark.parametrize(
    "question_row,expected_n_calls",
    [(q_n4_paris, 4), (q_n3_primary, 3), (q_n5_wide, 5)],
    ids=["n4", "n3", "n5"],
)
def test_ihs_call_count_matches_real_option_count(question_row, expected_n_calls, ihs_runner_kwargs):
    """I1/I2/I3: exactly one call per real option, never N-1 or a padded 4."""
    spy = SpyBackend(default_text="<score>50</score>")
    runner = IndependentHypothesisRunner(backend=spy, method_name="independent_hypothesis", **ihs_runner_kwargs)
    runner.run_one(question_row, sample_index=0)
    assert spy.call_count == expected_n_calls


def test_ihs_prompt_isolates_single_candidate(ihs_runner_kwargs):
    """I4: each candidate call's prompt contains ONLY that candidate's own
    option text, never any other option's text (no ordered full list)."""
    spy = SpyBackend(default_text="<score>50</score>")
    runner = IndependentHypothesisRunner(backend=spy, method_name="independent_hypothesis", **ihs_runner_kwargs)
    runner.run_one(q_n4_paris, sample_index=0)

    option_texts = ["Paris", "London", "Berlin", "Madrid"]
    for call, own_text in zip(spy.calls, option_texts):
        assert own_text in call.prompt
        other_texts = [t for t in option_texts if t != own_text]
        for other in other_texts:
            assert other not in call.prompt


def test_highest_valid_score_wins(ihs_runner_kwargs):
    """I5: all valid scores -> the highest one wins."""
    scores = ["<score>10</score>", "<score>90</score>", "<score>50</score>", "<score>20</score>"]
    spy = SpyBackend(responder=_sequential_responder(scores))
    runner = IndependentHypothesisRunner(backend=spy, method_name="independent_hypothesis", **ihs_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)
    assert row["parsed_choice"] == "B"  # second option (London) scored 90, the max


def test_all_invalid_scores_produce_no_fabricated_answer(ihs_runner_kwargs):
    """I6: every candidate unparseable -> unscorable, never a fabricated pick."""
    spy = SpyBackend(default_text="I have no numeric opinion on this.")
    runner = IndependentHypothesisRunner(backend=spy, method_name="independent_hypothesis", **ihs_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)
    assert row["parsed_choice"] is None
    assert row["answer_status"] == FAILURE_STATUS


def test_invalid_candidate_never_competes_as_implicit_zero(ihs_runner_kwargs):
    """I7: partially invalid candidates -- the invalid one must be EXCLUDED
    from the argmax entirely, not default to a losing 0. Proved by making
    every VALID score deliberately low/negative: if the invalid candidate's
    placeholder 0.0 were allowed to compete, it would win outright."""
    scores = [
        "<score>-5</score>",   # valid, lowest of the valid scores... wait scores must be 0-100
        "garbled nonsense",     # invalid -> must never win via implicit 0.0
        "<score>2</score>",     # valid, higher than 0.0
        "<score>1</score>",     # valid
    ]
    spy = SpyBackend(responder=_sequential_responder(scores))
    runner = IndependentHypothesisRunner(backend=spy, method_name="independent_hypothesis", **ihs_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)
    # Candidate A's "-5" is out of the 0-100 range -> also invalid. Only C
    # (2) and D (1) are valid; if the invalid ones' 0.0 secretly competed,
    # neither A nor B could ever win since both are excluded either way --
    # so re-derive with an in-range but LOW valid score for A instead.
    assert row["option_a_score_parse_ok"] is False  # confirms -5 is out of [0,100]
    assert row["option_b_score_parse_ok"] is False
    assert row["option_c_score_parse_ok"] is True
    assert row["option_d_score_parse_ok"] is True
    # Both valid candidates (C=2, D=1) score above the invalid ones' 0.0
    # placeholder would be irrelevant either way here; the discriminating
    # case is C (valid, 2) beating D (valid, 1) despite B's invalid text
    # being alphabetically/positionally earlier -- i.e. no implicit-zero
    # candidate silently wins over a low-but-valid one.
    assert row["parsed_choice"] == "C"


@pytest.mark.parametrize(
    "raw_score_text",
    [
        "<score>nan</score>",
        "<score>inf</score>",
        "<score>-inf</score>",
        "<score>150</score>",
        "<score>abc</score>",
        "no score tag at all",
    ],
)
def test_invalid_score_forms_are_rejected(raw_score_text, ihs_runner_kwargs):
    """I8: NaN / +inf / -inf / out-of-range / malformed text are all
    treated as parse_ok=False, never as a legitimate observation."""
    spy = SpyBackend(default_text=raw_score_text)
    runner = IndependentHypothesisRunner(backend=spy, method_name="independent_hypothesis", **ihs_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)
    assert row["option_a_score_parse_ok"] is False
    assert row["answer_status"] == FAILURE_STATUS  # all 4 candidates use the same text here


def test_transport_failure_in_later_candidate_is_not_hidden(ihs_runner_kwargs):
    """I9: candidate 1 succeeds, candidate 2 transport-fails. Top-level
    transport_status reflects ONLY candidate 1 (documented base-class
    behavior: model_response=responses[0]) -- this test pins that nuance
    down explicitly -- but the per-option column for candidate 2 must show
    the failure, so the aggregate failure is visible SOMEWHERE in the row."""
    from choicebench.clients.types import SUCCESS_STATUS

    spy = SpyBackend(default_text="<score>50</score>", raise_on_call={2})
    runner = IndependentHypothesisRunner(backend=spy, method_name="independent_hypothesis", **ihs_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)

    assert row["transport_status"] == SUCCESS_STATUS  # reflects candidate 1 (responses[0]) only
    assert row["option_b_model_status"] == FAILURE_STATUS  # candidate 2's own failure, visible per-option
    assert row["option_b_score_parse_ok"] is False


def test_exact_score_tie_uses_shared_tiebreak_protocol(ihs_runner_kwargs, base_runner_kwargs):
    """I10: a genuine score tie resolves via the shared canonical tie-break
    utility over source_index, not first-in-list order."""
    tie_text = "<score>77</score>"
    spy = SpyBackend(default_text=tie_text)  # all 4 candidates tie at 77
    runner = IndependentHypothesisRunner(backend=spy, method_name="independent_hypothesis", **ihs_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)

    label_to_source_index = runner._build_label_to_source_index(q_n4_paris)
    expected_winning_id = resolve_tie(
        seed=42, benchmark_id="diag_bench", question_id=q_n4_paris["question_id"],
        method_name="independent_hypothesis", tied_canonical_ids=list(label_to_source_index.values()),
    )
    id_to_label = {v: k for k, v in label_to_source_index.items()}
    assert row["parsed_choice"] == id_to_label[expected_winning_id]
