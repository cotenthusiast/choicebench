# tests/scripts/test_evaluate_run.py
#
# Regression for FSF-2 / PF-1: --reparse must only touch direct_mcq rows. For
# every other method the stored raw_text is not the call the reported answer
# came from, so reparsing it silently corrupts the result. This test builds a
# mixed-method run and asserts only the direct_mcq row is recomputed.

import importlib.util
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_evaluate_run():
    spec = importlib.util.spec_from_file_location(
        "evaluate_run_under_test", _REPO_ROOT / "scripts" / "evaluate_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(method, parse_reason, raw_text, parsed_choice, transport_status="success"):
    return {
        "method_name": method,
        "parse_reason": parse_reason,
        "raw_text": raw_text,
        "parsed_choice": parsed_choice,
        "parse_status": "parse_ok",
        "normalized_text": "",
        "score_status": "scored",
        "is_correct": parsed_choice == "C",
        "transport_status": transport_status,
        "correct_option": "C",
        "choice_a": "alpha",
        "choice_b": "bravo",
        "choice_c": "charlie",
        "choice_d": "delta",
        "question_id": f"{method}_q",
    }


def test_reparse_only_touches_direct_mcq_rows():
    mod = _load_evaluate_run()

    # Each non-direct row stores a stale parsed_choice="A" and a raw_text that
    # WOULD reparse to "B" — so any reparse is detectable as corruption.
    df = pd.DataFrame(
        [
            _row("direct_mcq", "Answer successfully parsed", "The answer is B", "A"),
            _row("cyclic_permutation", "majority_vote", None, "A"),
            _row("pride", "pride_eq8", None, "A"),
            _row("cyclic_logprob", "eq1_averaging", None, "A"),
            _row("two_stage", "Answer successfully parsed", "The answer is B", "A"),
        ]
    )

    out = mod.reparse_run(df)

    # direct_mcq row recomputed: stale "A" → "B".
    direct = out[out["method_name"] == "direct_mcq"].iloc[0]
    assert direct["parsed_choice"] == "B"

    # Every other method left exactly as it was.
    for method in ["cyclic_permutation", "pride", "cyclic_logprob", "two_stage"]:
        untouched = out[out["method_name"] == method].iloc[0]
        assert untouched["parsed_choice"] == "A", method
        assert untouched["parse_reason"] == df[df["method_name"] == method].iloc[0]["parse_reason"]


def test_reparse_logs_skipped_methods(caplog):
    mod = _load_evaluate_run()
    df = pd.DataFrame(
        [
            _row("direct_mcq", "Answer successfully parsed", "The answer is B", "A"),
            _row("pride", "pride_eq8", None, "A"),
            _row("two_stage", "Answer successfully parsed", "The answer is B", "A"),
        ]
    )
    import logging

    with caplog.at_level(logging.WARNING):
        mod.reparse_run(df)

    msg = caplog.text
    assert "pride" in msg
    assert "two_stage" in msg
