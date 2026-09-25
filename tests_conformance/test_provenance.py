"""Section Q: provenance / derived artifacts.

Frozen invariant: every observed row retains enough identity to distinguish
model/provider/benchmark/question/method/prompt-version/generation-settings;
a derived row (baseline-from-cyclic, semantic-matching) carries explicit
derived_from_* lineage and never masquerades as a fresh generation; a
visible-matcher row retains its text-extraction source relationship;
NaN/missing free text stays missing, never the literal string "nan"; and a
persisted row's status fields stay internally coherent (never is_correct=True
alongside a recorded transport failure).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from choicebench.analysis.derive_baseline_from_cyclic import derive_baseline_from_cyclic
from choicebench.analysis.derive_from_free_text import derive_matched_results
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.methods.library.text_extraction import TextExtractionRunner

from fakes.fixtures import q_n4_paris
from fakes.spy_backend import SpyBackend


def _exact_embed_fn(text: str):
    return [1.0 if text == q_n4_paris["choices_json"][0]["text"] else 0.0]


# --- Q1 ---------------------------------------------------------------

def test_q1_observed_row_identity_completeness(base_runner_kwargs):
    spy = SpyBackend(default_text="The answer is A.")
    runner = PermutationRunner(backend=spy, method_name="cyclic_permutation", **base_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)

    for field in (
        "provider", "model_name", "benchmark_name", "question_id", "method_name",
        "prompt_version", "temperature", "max_tokens", "per_rotation_choices_json",
    ):
        assert field in row and row[field] is not None, f"missing/null identity field: {field}"
    # Q1 is a schema-completeness assertion, not a value-correctness one --
    # parsed_choice legitimately depends on the (here, position-fixed)
    # responder's interaction with majority-vote tie-breaking across
    # rotations, not asserted here.
    assert row["parsed_choice"] in {"A", "B", "C", "D"}


# --- Q2 -------------------------------------------------------------------

def test_q2_baseline_row_lineage_and_no_rotation_trace(base_runner_kwargs):
    spy = SpyBackend(default_text="The answer is A.")
    runner = PermutationRunner(backend=spy, method_name="cyclic_permutation", **base_runner_kwargs)
    cyclic_row = runner.run_one(q_n4_paris, sample_index=0)

    baseline_df = derive_baseline_from_cyclic(pd.DataFrame([cyclic_row]), method_name="direct_mcq")
    baseline_row = baseline_df.iloc[0]

    assert baseline_row["derived_from_run_id"] == cyclic_row["run_id"]
    assert baseline_row["derived_from_method_name"] == "cyclic_permutation"
    assert baseline_row["method_name"] == "direct_mcq"
    # direct_mcq has no rotations of its own -- the column must not be
    # carried over from the cyclic source (would misleadingly imply
    # direct_mcq itself has a flip-rate trace).
    assert "per_rotation_choices_json" not in baseline_row or pd.isna(baseline_row.get("per_rotation_choices_json"))


# --- Q3 -------------------------------------------------------------------

def _two_stage_source_row(question_row: dict, free_text: str) -> dict:
    return {
        **question_row,
        "run_id": "source-run-1", "method_name": "two_stage",
        "free_text_response": free_text, "transport_status": "success",
        "seed": 42, "benchmark_name": "diag_bench",
    }


def test_q3_semantic_derived_row_does_not_masquerade_as_fresh_generation():
    row = _two_stage_source_row(q_n4_paris, "Paris")
    df = pd.DataFrame([row])
    derived_df = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_exact_embed_fn)
    derived_row = derived_df.iloc[0]

    assert derived_row["derived_from_run_id"] == "source-run-1"
    assert derived_row["derived_from_method_name"] == "two_stage"
    # Must inherit the SOURCE row's own transport_status unchanged -- never
    # fabricate a fresh transport outcome for a call that never happened.
    assert derived_row["transport_status"] == row["transport_status"]


# --- Q4 -------------------------------------------------------------------

def test_q4_visible_matcher_retains_text_extraction_source_relationship(base_runner_kwargs):
    spy = SpyBackend(default_text="The answer is A.")
    from choicebench.methods.library.visible_llm_matcher import VisibleLLMMatcherRunner

    row_with_extracted_text = dict(q_n4_paris)
    row_with_extracted_text["extracted_text"] = "Paris"
    runner = VisibleLLMMatcherRunner(backend=spy, method_name="visible_llm_matcher", **base_runner_kwargs)
    result_row = runner.run_one(row_with_extracted_text, sample_index=0)

    assert "reused_extracted_text" in result_row
    assert result_row["reused_extracted_text"] == "Paris"


# --- Q5 ---------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "derive_from_free_text.py's _derive_one_row sets "
        "normalized_text=str(free_text) if free_text is not None else None -- "
        "a pandas NaN (float('nan'), not Python None) IS 'not None', so "
        "str(float('nan')) produces the literal string 'nan', which becomes "
        "the persisted normalized_text value instead of staying missing. "
        "Same root cause as spec G4."
    ),
)
def test_q5_nan_free_text_normalized_text_stays_missing_not_literal_nan():
    row = _two_stage_source_row(q_n4_paris, float("nan"))
    df = pd.DataFrame([row])
    derived_df = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_exact_embed_fn)
    normalized_text = derived_df.iloc[0]["normalized_text"]
    assert normalized_text is None or (isinstance(normalized_text, float) and math.isnan(normalized_text))
    assert normalized_text != "nan"


# --- Q6 -------------------------------------------------------------------

def test_q6_internally_coherent_status_on_transport_failure(base_runner_kwargs):
    spy = SpyBackend(raise_on_call={1})
    runner = TextExtractionRunner(backend=spy, method_name="text_extraction", **base_runner_kwargs)
    row = runner.run_one(q_n4_paris, sample_index=0)

    assert row["transport_status"] == "failure"
    assert row["answer_status"] == "failure"
    assert row["is_correct"] is not True
