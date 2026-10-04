# tests/analysis/test_derive_text_extraction.py

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from choicebench.analysis.derive_text_extraction import PARSER_VERSION, reparse_text_extraction
from choicebench.backends.dummy_backend import DummyBackend
from choicebench.methods.library.text_extraction import TextExtractionRunner
from choicebench.scoring.types import SCORE_CORRECT, SCORE_UNSCORABLE


def _none(v):
    return None if pd.isna(v) else v

REPO_ROOT = Path(__file__).resolve().parents[2]


def _no_match_embed_fn(texts):
    return np.eye(len(texts), dtype=np.float64)


def _row(qid="q1", raw="HTTPS", **overrides):
    choices = [{"text": t, "source_index": i} for i, t in enumerate(["FTP", "HTTP", "HTTPS", "SMTP"])]
    row = dict(
        run_id="prod_run", experiment_id="exp1", condition_id="cond_abc", benchmark_name="mmlu", question_id=qid,
        split_name="test", subject="computer_security", method_name="text_extraction", prompt_version="v1",
        provider="openai", model_name="gpt-4.1-mini", temperature=0.0, max_tokens=1024, seed=42,
        question_text="Which protocol is primarily used to securely browse websites?",
        choices_json=json.dumps(choices), correct_index=2, correct_option="C", n_choices=4,
        prompt="p", transport_status="success", answer_status="success", raw_text=raw,
        # what the buggy production path stored (letter parser)
        parsed_choice="B" if raw == "B" else "C", parse_status="ok", parse_reason="free_form_parser",
        normalized_text="x", is_correct=False, score_status="incorrect",
    )
    row.update(overrides)
    return row


def _runner():
    return TextExtractionRunner(
        backend=DummyBackend(), method_name="text_extraction", split_name="test", prompt_version="v1",
        prompts_dir=REPO_ROOT / "prompts", run_id="chk", seed=42, benchmark_name="mmlu", embed_fn=_no_match_embed_fn,
    )


RAWS = ["HTTPS", "B", "ftp", "The best answer is HTTPS", "completely unrelated gibberish response"]


def test_derivation_has_no_backend_parameter():
    import inspect
    assert not any("backend" in p or "client" in p for p in inspect.signature(reparse_text_extraction).parameters)


def test_derived_answers_equal_the_runners_own_cascade():
    df = pd.DataFrame([_row(f"q{i}", raw) for i, raw in enumerate(RAWS)])
    derived = reparse_text_extraction(df, embed_fn=_no_match_embed_fn)
    runner = _runner()
    expected = [runner._match_and_score(raw, _row(f"q{i}", raw))[0].final_choice for i, raw in enumerate(RAWS)]
    assert [_none(v) for v in derived["parsed_choice"]] == expected
    assert _none(derived.loc[1, "parsed_choice"]) is None  # raw "B" is not option text: the letter parser's answer is gone


def test_scoring_fields_are_recomputed_and_consistent():
    df = pd.DataFrame([_row("q1", "HTTPS"), _row("q2", "B")])
    d = reparse_text_extraction(df, embed_fn=_no_match_embed_fn)
    assert bool(d.loc[0, "is_correct"]) is True and d.loc[0, "score_status"] == SCORE_CORRECT
    assert d.loc[0, "answer_status"] == "success" and d.loc[0, "parse_reason"] == "text_matcher_cascade"
    assert _none(d.loc[1, "parsed_choice"]) is None and d.loc[1, "answer_status"] == "failure"
    assert d.loc[1, "score_status"] == SCORE_UNSCORABLE and not bool(d.loc[1, "is_correct"])


def test_raw_outputs_and_identity_are_untouched_and_originals_are_preserved():
    df = pd.DataFrame([_row("q1", "B")])
    d = reparse_text_extraction(df, embed_fn=_no_match_embed_fn)
    for col in ("raw_text", "prompt", "question_id", "model_name", "run_id", "transport_status"):
        assert d.loc[0, col] == df.loc[0, col]
    assert d.loc[0, "original_parsed_choice"] == "B" and d.loc[0, "original_parse_reason"] == "free_form_parser"
    assert d.loc[0, "derived_from_run_id"] == "prod_run" and d.loc[0, "derived_from_condition_id"] == "cond_abc"
    assert d.loc[0, "parser_version"] == PARSER_VERSION


def test_no_rotation_invariance_trace_is_fabricated():
    """Unlike semantic matching, text_extraction's options are VISIBLE: its answer is not rotation-invariant, so the
    derivation must not write a per_rotation_choices_json that would claim a zero flip rate."""
    d = reparse_text_extraction(pd.DataFrame([_row()]), embed_fn=_no_match_embed_fn)
    assert "per_rotation_choices_json" not in d.columns


def test_missing_raw_text_is_unscorable_not_the_string_nan():
    d = reparse_text_extraction(pd.DataFrame([_row("q1", None, transport_status="failure")]), embed_fn=_no_match_embed_fn)
    assert _none(d.loc[0, "parsed_choice"]) is None and d.loc[0, "answer_status"] == "failure"


def test_refuses_a_source_that_is_not_text_extraction_or_mixes_models():
    with pytest.raises(ValueError, match="text_extraction"):
        reparse_text_extraction(pd.DataFrame([_row(method_name="two_stage")]), embed_fn=_no_match_embed_fn)
    with pytest.raises(ValueError, match="model_name"):
        reparse_text_extraction(pd.DataFrame([_row("q1"), _row("q2", model_name="other")]), embed_fn=_no_match_embed_fn)


def test_cli_writes_a_new_file_with_source_sha256_and_never_overwrites(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("derive_te", REPO_ROOT / "scripts/paper/derive_text_extraction_corrected.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src, out = tmp_path / "cond_x.csv", tmp_path / "out" / "corrected.csv"
    pd.DataFrame([_row("q1", "HTTPS"), _row("q2", "B")]).to_csv(src, index=False)
    before = src.read_bytes()
    mod.derive(src, out, embed_fn=_no_match_embed_fn)
    got = pd.read_csv(out)
    assert set(got["derived_from_sha256"]) == {hashlib.sha256(before).hexdigest()}
    assert src.read_bytes() == before                      # the original is byte-identical
    assert len(got) == 2 and got["derived_from_file"].iloc[0].endswith("cond_x.csv")
    with pytest.raises(FileExistsError):
        mod.derive(src, out, embed_fn=_no_match_embed_fn)
    with pytest.raises(ValueError):
        mod.derive(src, src, embed_fn=_no_match_embed_fn)  # output must never be the source
