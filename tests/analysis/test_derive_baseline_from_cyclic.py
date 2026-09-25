# tests/analysis/test_derive_baseline_from_cyclic.py

import json

import pandas as pd
import pytest

from choicebench.analysis.derive_baseline_from_cyclic import derive_baseline_from_cyclic
from choicebench.clients.types import FAILURE_STATUS, SUCCESS_STATUS


def _source_row(**overrides) -> dict:
    """A saved cyclic_permutation output row -- majority-voted parsed_choice/
    is_correct/answer_status, but rotation 0's OWN raw_text/transport_status
    (per PermutationRunner: model_response=responses[0] for both its sync
    and async paths)."""
    choices_json = json.dumps([
        {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
        {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
    ])
    row = dict(
        run_id="src_run_001", benchmark_name="mmlu", question_id="q1",
        subject="computer_security", method_name="cyclic_permutation", prompt_version="v1",
        model_name="gpt-4.1-mini", provider="openai", seed=42, sample_index=0,
        question_text="Which protocol is primarily used to securely browse websites?",
        choices_json=choices_json, correct_option="C", n_choices=4,
        # Rotation 0's own raw call -- "C" (correct), but the source row's
        # own parsed_choice/is_correct/answer_status below deliberately
        # reflect a DIFFERENT (majority-vote) outcome, to prove the
        # derivation recomputes from raw_text rather than inheriting them.
        raw_text="C", transport_status=SUCCESS_STATUS, finish_reason="stop",
        latency_seconds=0.4, timestamp_utc="2026-09-25T00:00:00Z",
        error_type=None, error_message=None, error_stage=None, error_retryable=None,
        prompt="Answer the following...",
        parsed_choice="A", parse_status="ok", normalized_text="a", parse_reason="majority_vote",
        is_correct=False, score_status="scored", answer_status=SUCCESS_STATUS,
        per_rotation_choices_json=json.dumps(["C", "A", "A", "A"]),
    )
    row.update(overrides)
    return row


class TestDeriveBaselineFromCyclic:
    def test_zero_new_model_calls(self):
        """Structural guarantee: no backend/client parameter exists to call."""
        import inspect
        params = inspect.signature(derive_baseline_from_cyclic).parameters
        assert not any("backend" in p or "client" in p for p in params)

    def test_uses_rotation0_raw_text_not_the_majority_vote_answer(self):
        """The core correctness property: the derived baseline answer must
        come from rotation 0's OWN raw_text, re-parsed against the CANONICAL
        (unpermuted) options -- never copied from the source row's own
        parsed_choice, which reflects the majority vote across ALL rotations,
        a materially different observation."""
        df = pd.DataFrame([_source_row(raw_text="C", parsed_choice="A")])
        derived = derive_baseline_from_cyclic(df, method_name="direct_mcq")
        assert derived.iloc[0]["parsed_choice"] == "C"
        assert bool(derived.iloc[0]["is_correct"]) is True

    def test_incorrect_rotation0_answer(self):
        df = pd.DataFrame([_source_row(raw_text="A", correct_option="C")])
        derived = derive_baseline_from_cyclic(df, method_name="direct_mcq")
        assert derived.iloc[0]["parsed_choice"] == "A"
        assert bool(derived.iloc[0]["is_correct"]) is False

    def test_method_name_is_overwritten_on_derived_rows(self):
        df = pd.DataFrame([_source_row(method_name="cyclic_permutation")])
        derived = derive_baseline_from_cyclic(df, method_name="direct_mcq")
        assert derived.iloc[0]["method_name"] == "direct_mcq"

    def test_provenance_back_to_source_is_preserved(self):
        df = pd.DataFrame([_source_row(run_id="src_run_001", method_name="cyclic_permutation")])
        derived = derive_baseline_from_cyclic(df, method_name="direct_mcq")
        assert derived.iloc[0]["derived_from_run_id"] == "src_run_001"
        assert derived.iloc[0]["derived_from_method_name"] == "cyclic_permutation"

    def test_rotation0_provenance_fields_are_preserved_unchanged(self):
        """prompt/raw_text/transport_status/latency/timestamp are ALREADY
        rotation-0-specific in the source row (PermutationRunner always uses
        responses[0]/prompts[0] for these) -- the derivation must carry them
        through verbatim, not regenerate or blank them."""
        df = pd.DataFrame([_source_row(
            raw_text="C", prompt="the exact rotation-0 prompt", latency_seconds=0.77,
            timestamp_utc="2026-09-25T01:02:03Z",
        )])
        derived = derive_baseline_from_cyclic(df, method_name="direct_mcq")
        row = derived.iloc[0]
        assert row["raw_text"] == "C"
        assert row["prompt"] == "the exact rotation-0 prompt"
        assert row["latency_seconds"] == 0.77
        assert row["timestamp_utc"] == "2026-09-25T01:02:03Z"

    def test_answer_status_reflects_rotation0_not_the_inherited_majority_vote_status(self):
        """The source row's own answer_status reflects the MAJORITY VOTE's
        success, not rotation 0 alone -- must be recomputed from the
        freshly-derived parse, not inherited via dict(row)."""
        # Majority vote succeeded (answer_status=success in the source row),
        # but rotation 0's own raw_text fails to parse against any option.
        df = pd.DataFrame([_source_row(
            raw_text="xyz completely unrelated response 12345", answer_status=SUCCESS_STATUS,
        )])
        derived = derive_baseline_from_cyclic(df, method_name="direct_mcq")
        assert derived.iloc[0]["answer_status"] == FAILURE_STATUS
        assert derived.iloc[0]["parsed_choice"] is None

    def test_rotation0_transport_failure_is_unscorable_not_crashed(self):
        """Rotation 0's own call failed transport -- no raw_text to parse.
        Must come through as unscorable, matching DirectMCQRunner's own
        model_response.is_success() gate, never crash."""
        df = pd.DataFrame([_source_row(raw_text=None, transport_status=FAILURE_STATUS)])
        derived = derive_baseline_from_cyclic(df, method_name="direct_mcq")
        assert derived.iloc[0]["parsed_choice"] is None
        assert derived.iloc[0]["answer_status"] == FAILURE_STATUS

    def test_per_rotation_choices_json_is_dropped_from_the_derived_row(self):
        """direct_mcq has no rotations of its own -- carrying cyclic's trace
        column through could wrongly suggest direct_mcq's OWN flip-rate
        should be computed from it, when the documented relationship is that
        direct_mcq's flip-rate is read from the cyclic_permutation artifact
        directly, never from direct_mcq's own file."""
        df = pd.DataFrame([_source_row()])
        derived = derive_baseline_from_cyclic(df, method_name="direct_mcq")
        assert "per_rotation_choices_json" not in derived.columns

    def test_source_row_without_per_rotation_choices_json_column_raises(self):
        df = pd.DataFrame([{"question_id": "q1"}])
        with pytest.raises(ValueError, match="per_rotation_choices_json"):
            derive_baseline_from_cyclic(df, method_name="direct_mcq")

    def test_multiple_rows_derive_independently(self):
        df = pd.DataFrame([
            _source_row(question_id="q1", raw_text="C"),
            _source_row(question_id="q2", raw_text="A"),
        ])
        derived = derive_baseline_from_cyclic(df, method_name="direct_mcq")
        assert list(derived["parsed_choice"]) == ["C", "A"]

    def test_three_option_question(self):
        choices_json = json.dumps([
            {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
            {"text": "HTTPS", "source_index": 2},
        ])
        df = pd.DataFrame([_source_row(
            raw_text="C", choices_json=choices_json, correct_option="C", n_choices=3,
            per_rotation_choices_json=json.dumps(["C", "A", "B"]),
        )])
        derived = derive_baseline_from_cyclic(df, method_name="direct_mcq")
        assert derived.iloc[0]["parsed_choice"] == "C"
        assert bool(derived.iloc[0]["is_correct"]) is True
