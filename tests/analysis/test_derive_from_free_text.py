# tests/analysis/test_derive_from_free_text.py

import json

import numpy as np
import pandas as pd
import pytest

from choicebench.analysis.derive_from_free_text import derive_matched_results


def _no_match_embed_fn(texts: list[str]) -> np.ndarray:
    return np.eye(len(texts), dtype=np.float64)


def _source_row(**overrides) -> dict:
    choices_json = json.dumps([
        {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
        {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
    ])
    row = dict(
        run_id="src_run_001", benchmark_name="mmlu", question_id="q1",
        subject="computer_security", method_name="two_stage_v1", prompt_version="v1",
        model_name="gpt-4.1-mini", provider="openai", seed=42,
        question_text="Which protocol is primarily used to securely browse websites?",
        choices_json=choices_json, correct_option="C", n_choices=4,
        free_text_response="HTTPS", answer_status="success", transport_status="success",
    )
    row.update(overrides)
    return row


class TestDeriveMatchedResults:
    def test_zero_new_model_calls(self):
        """Structural guarantee: the derivation function's signature has no
        backend/client parameter at all -- there is nothing to call."""
        import inspect
        params = inspect.signature(derive_matched_results).parameters
        assert not any("backend" in p or "client" in p for p in params)

    def test_exact_match_derives_correct(self):
        df = pd.DataFrame([_source_row(free_text_response="HTTPS")])
        derived = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_no_match_embed_fn)
        assert derived.iloc[0]["parsed_choice"] == "C"
        assert bool(derived.iloc[0]["is_correct"]) is True

    def test_incorrect_match(self):
        df = pd.DataFrame([_source_row(free_text_response="FTP")])
        derived = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_no_match_embed_fn)
        assert derived.iloc[0]["parsed_choice"] == "A"
        assert bool(derived.iloc[0]["is_correct"]) is False

    def test_unmatched_free_text_is_unscorable(self):
        df = pd.DataFrame([_source_row(free_text_response="gibberish unrelated text")])
        derived = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_no_match_embed_fn)
        assert derived.iloc[0]["parsed_choice"] is None

    def test_method_name_is_overwritten_on_derived_rows(self):
        df = pd.DataFrame([_source_row(method_name="two_stage_v1")])
        derived = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_no_match_embed_fn)
        assert derived.iloc[0]["method_name"] == "semantic_matching_v1"

    def test_provenance_back_to_source_is_preserved(self):
        df = pd.DataFrame([_source_row(run_id="src_run_001", method_name="two_stage_v1")])
        derived = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_no_match_embed_fn)
        assert derived.iloc[0]["derived_from_run_id"] == "src_run_001"
        assert derived.iloc[0]["derived_from_method_name"] == "two_stage_v1"

    def test_source_row_without_free_text_response_column_raises(self):
        df = pd.DataFrame([{"question_id": "q1"}])
        with pytest.raises(ValueError, match="free_text_response"):
            derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_no_match_embed_fn)

    def test_source_rows_that_failed_transport_are_carried_through_as_unscorable(self):
        """A row whose Stage-1 call itself failed has no free_text_response
        to match against -- must not crash, must come through as unscorable."""
        df = pd.DataFrame([_source_row(free_text_response=None, transport_status="failure")])
        derived = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_no_match_embed_fn)
        assert derived.iloc[0]["parsed_choice"] is None

    def test_multiple_rows_derive_independently(self):
        df = pd.DataFrame([
            _source_row(question_id="q1", free_text_response="HTTPS"),
            _source_row(question_id="q2", free_text_response="FTP"),
        ])
        derived = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_no_match_embed_fn)
        assert list(derived["parsed_choice"]) == ["C", "A"]


class TestDerivedRotationTrace:
    """semantic_matching_v1's match is a pure function of TEXT content
    (exact/containment/cosine over option strings) -- it never looks at
    which display letter or rotation position a text currently occupies.
    So its answer is provably invariant to rotation: the per-rotation trace
    is the same canonical choice repeated once per option, derived without
    looping over synthetic rotations or making any new calls."""

    def test_per_rotation_choices_json_is_the_matched_letter_repeated_n_times(self):
        df = pd.DataFrame([_source_row(free_text_response="HTTPS")])  # 4 options
        derived = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_no_match_embed_fn)
        per_rotation = json.loads(derived.iloc[0]["per_rotation_choices_json"])
        assert per_rotation == ["C", "C", "C", "C"]

    def test_unmatched_free_text_yields_all_none_trace(self):
        df = pd.DataFrame([_source_row(free_text_response="gibberish unrelated text")])
        derived = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_no_match_embed_fn)
        per_rotation = json.loads(derived.iloc[0]["per_rotation_choices_json"])
        assert per_rotation == [None, None, None, None]

    def test_trace_length_matches_this_questions_own_option_count(self):
        """A 3-option question's trace has 3 entries, not a hardcoded 4."""
        choices_json = json.dumps([
            {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
            {"text": "HTTPS", "source_index": 2},
        ])
        df = pd.DataFrame([_source_row(
            free_text_response="HTTPS", choices_json=choices_json, correct_option="C", n_choices=3,
        )])
        derived = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_no_match_embed_fn)
        per_rotation = json.loads(derived.iloc[0]["per_rotation_choices_json"])
        assert len(per_rotation) == 3
