"""Section O: resume equivalence.

Frozen invariant: an uninterrupted run and an interrupted-then-resumed run
must produce a semantically identical final artifact; a schema mismatch or
wrong-method resume attempt must fail loudly; conflicting duplicate
"completed" rows must never be silently collapsed to one; --no-resume
against a pre-existing output must not silently duplicate rows.

O5 (stochasticity resume) is the identical scenario as spec L6 under a
different letter -- covered in test_stochasticity_protocol.py, cross-
referenced here rather than duplicated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PAPER_DIR = REPO_ROOT / "scripts" / "paper"
if str(SCRIPTS_PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PAPER_DIR))

import run_text_extraction_rotations as ter  # noqa: E402
import run_two_stage_rotations as tsr  # noqa: E402
import run_visible_llm_matcher_rotations as vmr  # noqa: E402

from choicebench.config.schema import GenerationKwargsConfig, ModelConfig
from choicebench.infra.resumable_csv import check_resume_compatible
from choicebench.methods.library.permutation import PermutationRunner

from fakes.fixtures import q_n4_paris
from fakes.spy_backend import SpyBackend

PROMPTS_DIR = REPO_ROOT / "prompts"


def _model_config() -> ModelConfig:
    return ModelConfig(
        backend="api", provider="spy-provider", model_name_or_path="spy-model",
        generation_kwargs=GenerationKwargsConfig(temperature=0.0, max_new_tokens=1024),
    )


def _fixed_letter_responder(target_text: str):
    import re

    def responder(prompt: str) -> str:
        options = dict(re.findall(r"^([A-Z])\. (.+)$", prompt, re.MULTILINE))
        for letter, text in options.items():
            if text == target_text:
                return f"The answer is {letter}."
        raise AssertionError(f"{target_text!r} not found in prompt options: {options}")

    return responder


# --- O1 -------------------------------------------------------------------

def test_o1_cyclic_interrupted_then_retried_matches_uninterrupted(base_runner_kwargs):
    """PermutationRunner.run_one() has no persisted mid-question checkpoint
    (no standalone 'resume cyclic rotations' script exists), and a single
    rotation's transport failure does NOT raise out of run_one() at all --
    _call_backend_generate() catches it and folds it into the row as one
    failed/None vote among the majority-vote rotations (confirmed directly:
    raise_on_call={3} below does not raise). There is therefore no
    "interrupted, caught exception, then resumed" scenario to exercise at
    this granularity; the closest faithful test of O1's intent is: a FRESH,
    FULL retry (the only recovery production actually offers at the
    per-question level) reproduces the same result as an uninterrupted run,
    given a deterministic (content-keyed, not call-count-keyed) responder."""
    responder = _fixed_letter_responder("Paris")

    spy_uninterrupted = SpyBackend(responder=responder)
    runner_uninterrupted = PermutationRunner(backend=spy_uninterrupted, method_name="cyclic_permutation", **base_runner_kwargs)
    uninterrupted_row = runner_uninterrupted.run_one(q_n4_paris, sample_index=0)

    # A degraded run where one rotation's call fails -- production tolerates
    # this gracefully (majority vote over the remaining 3 successful votes)
    # rather than raising, which is itself worth recording as observed
    # behavior distinct from "resume."
    spy_degraded = SpyBackend(responder=responder, raise_on_call={3})
    runner_degraded = PermutationRunner(backend=spy_degraded, method_name="cyclic_permutation", **base_runner_kwargs)
    degraded_row = runner_degraded.run_one(q_n4_paris, sample_index=0)
    assert spy_degraded.call_count == 4  # all 4 rotations were attempted despite the failure
    assert degraded_row["parsed_choice"] == "A"  # majority vote still recovers the right answer from 3/4 rotations

    spy_retried = SpyBackend(responder=responder)
    runner_retried = PermutationRunner(backend=spy_retried, method_name="cyclic_permutation", **base_runner_kwargs)
    retried_row = runner_retried.run_one(q_n4_paris, sample_index=0)

    assert retried_row["parsed_choice"] == uninterrupted_row["parsed_choice"]
    assert retried_row["is_correct"] == uninterrupted_row["is_correct"]
    assert retried_row["per_rotation_choices_json"] == uninterrupted_row["per_rotation_choices_json"]


# --- O2/O3/O4: genuinely resumable rotation scripts -------------------------

def _questions_csv(tmp_path: Path, filename: str = "questions.csv") -> Path:
    rows = []
    for i in range(1, 5):
        rows.append({
            "question_id": f"q{i}", "subject": "geography",
            "question_text": f"Question {i}: capital of France?",
            "choices_json": '[{"text": "Paris", "source_index": 0}, {"text": "London", "source_index": 1}, {"text": "Berlin", "source_index": 2}, {"text": "Madrid", "source_index": 3}]',
            "correct_option": "A", "benchmark_name": "diag_bench",
        })
    path = tmp_path / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_o2_text_extraction_rotations_resume_matches_uninterrupted(tmp_path, monkeypatch):
    responder = _fixed_letter_responder("Paris")

    def fake_build_backend(model_config, run_id, run_seed, execution_mode=None):
        return SpyBackend(responder=responder)

    monkeypatch.setattr(ter, "build_backend", fake_build_backend)

    questions_csv = _questions_csv(tmp_path)
    full_output = tmp_path / "full.csv"
    ter.run(questions_csv, _model_config(), run_id="t", output_path=full_output, run_seed=42, resume=True, execution_mode="sync")
    full_df = pd.read_csv(full_output).sort_values("question_id").reset_index(drop=True)

    partial_questions_csv = _questions_csv(tmp_path, filename="partial_input.csv")
    partial_df_source = pd.read_csv(partial_questions_csv).iloc[:2]
    partial_input_path = tmp_path / "partial_input_2rows.csv"
    partial_df_source.to_csv(partial_input_path, index=False)

    resumed_output = tmp_path / "resumed.csv"
    ter.run(partial_input_path, _model_config(), run_id="t", output_path=resumed_output, run_seed=42, resume=True, execution_mode="sync")
    ter.run(questions_csv, _model_config(), run_id="t", output_path=resumed_output, run_seed=42, resume=True, execution_mode="sync")
    resumed_df = pd.read_csv(resumed_output).sort_values("question_id").reset_index(drop=True)

    # sample_index is a per-invocation-local counter (enumerate() over
    # whatever's pending IN THIS CALL, not a stable global index) -- a
    # resumed run's second call restarts it from 0 for the remaining rows,
    # so it differs from an uninterrupted single-call run by construction.
    # It is execution-partitioning metadata, not scientific content, exactly
    # like latency_seconds/timestamp_utc -- excluded from the equivalence
    # check on the same basis.
    compare_cols = [c for c in full_df.columns if c not in {"latency_seconds", "timestamp_utc", "sample_index"}]
    pd.testing.assert_frame_equal(full_df[compare_cols], resumed_df[compare_cols])


def test_o3_two_stage_rotations_resume_matches_uninterrupted(tmp_path, monkeypatch):
    responder = _fixed_letter_responder("Paris")

    def fake_build_backend(model_config, run_id, run_seed):
        return SpyBackend(responder=responder)

    monkeypatch.setattr(tsr, "build_backend", fake_build_backend)

    rows = []
    for i in range(1, 5):
        rows.append({
            "question_id": f"q{i}", "subject": "geography",
            "question_text": f"Question {i}?",
            "choices_json": '[{"text": "Paris", "source_index": 0}, {"text": "London", "source_index": 1}, {"text": "Berlin", "source_index": 2}, {"text": "Madrid", "source_index": 3}]',
            "correct_option": "A", "benchmark_name": "diag_bench",
            "free_text_response": "I think it is Paris.",
        })
    two_stage_csv = tmp_path / "two_stage.csv"
    pd.DataFrame(rows).to_csv(two_stage_csv, index=False)

    full_output = tmp_path / "full_ts.csv"
    tsr.run(two_stage_csv, _model_config(), run_id="t", output_path=full_output, run_seed=42, resume=True)
    full_df = pd.read_csv(full_output).sort_values("question_id").reset_index(drop=True)

    resumed_output = tmp_path / "resumed_ts.csv"
    partial_csv = tmp_path / "two_stage_partial.csv"
    pd.DataFrame(rows[:2]).to_csv(partial_csv, index=False)
    tsr.run(partial_csv, _model_config(), run_id="t", output_path=resumed_output, run_seed=42, resume=True)
    tsr.run(two_stage_csv, _model_config(), run_id="t", output_path=resumed_output, run_seed=42, resume=True)
    resumed_df = pd.read_csv(resumed_output).sort_values("question_id").reset_index(drop=True)

    # sample_index is a per-invocation-local counter (enumerate() over
    # whatever's pending IN THIS CALL, not a stable global index) -- a
    # resumed run's second call restarts it from 0 for the remaining rows,
    # so it differs from an uninterrupted single-call run by construction.
    # It is execution-partitioning metadata, not scientific content, exactly
    # like latency_seconds/timestamp_utc -- excluded from the equivalence
    # check on the same basis.
    compare_cols = [c for c in full_df.columns if c not in {"latency_seconds", "timestamp_utc", "sample_index"}]
    pd.testing.assert_frame_equal(full_df[compare_cols], resumed_df[compare_cols])


def test_o4_visible_matcher_rotations_resume_matches_uninterrupted(tmp_path, monkeypatch):
    def fake_build_backend(model_config, run_id, run_seed):
        return SpyBackend(default_text="Paris.")

    monkeypatch.setattr(vmr, "build_backend", fake_build_backend)

    import json as _json
    rows = []
    for i in range(1, 4):
        rows.append({
            "question_id": f"q{i}", "subject": "geography",
            "question_text": f"Question {i}?",
            "choices_json": '[{"text": "Paris", "source_index": 0}, {"text": "London", "source_index": 1}, {"text": "Berlin", "source_index": 2}, {"text": "Madrid", "source_index": 3}]',
            "correct_option": "A", "benchmark_name": "diag_bench",
            "per_rotation_raw_text_json": _json.dumps(["Paris.", "Paris.", "Paris.", "Paris."]),
        })
    source_csv = tmp_path / "text_extraction_rotations.csv"
    pd.DataFrame(rows).to_csv(source_csv, index=False)

    full_output = tmp_path / "full_vm.csv"
    vmr.run(source_csv, _model_config(), run_id="t", output_path=full_output, run_seed=42, resume=True)
    full_df = pd.read_csv(full_output).sort_values("question_id").reset_index(drop=True)

    resumed_output = tmp_path / "resumed_vm.csv"
    partial_csv = tmp_path / "vm_partial.csv"
    pd.DataFrame(rows[:1]).to_csv(partial_csv, index=False)
    vmr.run(partial_csv, _model_config(), run_id="t", output_path=resumed_output, run_seed=42, resume=True)
    vmr.run(source_csv, _model_config(), run_id="t", output_path=resumed_output, run_seed=42, resume=True)
    resumed_df = pd.read_csv(resumed_output).sort_values("question_id").reset_index(drop=True)

    # sample_index is a per-invocation-local counter (enumerate() over
    # whatever's pending IN THIS CALL, not a stable global index) -- a
    # resumed run's second call restarts it from 0 for the remaining rows,
    # so it differs from an uninterrupted single-call run by construction.
    # It is execution-partitioning metadata, not scientific content, exactly
    # like latency_seconds/timestamp_utc -- excluded from the equivalence
    # check on the same basis.
    compare_cols = [c for c in full_df.columns if c not in {"latency_seconds", "timestamp_utc", "sample_index"}]
    pd.testing.assert_frame_equal(full_df[compare_cols], resumed_df[compare_cols])


# --- O6 -------------------------------------------------------------------

def test_o6_schema_mismatch_hard_failure(tmp_path):
    path = tmp_path / "existing.csv"
    pd.DataFrame([{"question_id": "q1", "some_other_column": 1}]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        check_resume_compatible(path, exact_columns=["question_id", "method_name", "parsed_choice"])


# --- O7 -------------------------------------------------------------------

def test_o7_wrong_method_name_same_columns_hard_failure(tmp_path):
    path = tmp_path / "existing.csv"
    pd.DataFrame([{"question_id": "q1", "method_name": "some_other_method"}]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        check_resume_compatible(path, required_columns=["question_id", "method_name"], expected_method_name="two_stage")


# --- O8 -------------------------------------------------------------------

def test_o8_conflicting_duplicate_completed_rows_not_silently_collapsed(tmp_path):
    """FIXED (verified against commit 651137768dcad640de28f124cff3d50837fe7d7c):
    _load_completed_question_ids() now delegates to
    infra.resumable_csv.load_completed_ids_excluding_conflicts(), which
    excludes any question_id with disagreeing rows from the "completed" set
    entirely, rather than silently picking whichever row loaded first.
    Previously xfail."""
    path = tmp_path / "existing.csv"
    pd.DataFrame([
        {"question_id": "q1", "parsed_choice": "A", "method_name": "text_extraction"},
        {"question_id": "q1", "parsed_choice": "B", "method_name": "text_extraction"},
    ]).to_csv(path, index=False)
    completed = ter._load_completed_question_ids(path)
    # Frozen invariant: a conflicting duplicate must not be silently folded
    # into a single "completed" verdict for q1.
    assert "q1" not in completed


def test_o8_run_hard_fails_loudly_on_conflicting_rows(tmp_path, monkeypatch):
    """Stronger form of O8, exercised through the real run() entry point:
    production now fails LOUDLY (not just silently-excludes) when resuming
    against a file with conflicting duplicate rows -- verified via
    infra.resumable_csv.detect_conflicting_ids(), called before any
    "completed" decision is made."""
    def fake_build_backend(model_config, run_id, run_seed, execution_mode=None):
        return SpyBackend(default_text="Paris.")

    monkeypatch.setattr(ter, "build_backend", fake_build_backend)
    questions_csv = _questions_csv(tmp_path)
    output_path = tmp_path / "conflicting.csv"
    pd.DataFrame([
        {"question_id": "q1", "parsed_choice": "A", "method_name": "text_extraction"},
        {"question_id": "q1", "parsed_choice": "B", "method_name": "text_extraction"},
    ]).to_csv(output_path, index=False)

    with pytest.raises(ValueError, match="conflicting duplicate"):
        ter.run(questions_csv, _model_config(), run_id="t", output_path=output_path, run_seed=42, resume=True, execution_mode="sync")


# --- O9 -------------------------------------------------------------------

def test_o9_no_resume_against_existing_output_does_not_duplicate_rows(tmp_path, monkeypatch):
    """FIXED (verified against commit 651137768dcad640de28f124cff3d50837fe7d7c):
    run() with resume=False now unlinks a pre-existing output_path before
    proceeding, so a --no-resume run always starts genuinely fresh instead
    of appending onto stale content. Previously xfail."""
    responder = _fixed_letter_responder("Paris")

    def fake_build_backend(model_config, run_id, run_seed, execution_mode=None):
        return SpyBackend(responder=responder)

    monkeypatch.setattr(ter, "build_backend", fake_build_backend)
    questions_csv = _questions_csv(tmp_path)
    output_path = tmp_path / "out.csv"

    ter.run(questions_csv, _model_config(), run_id="t", output_path=output_path, run_seed=42, resume=True, execution_mode="sync")
    ter.run(questions_csv, _model_config(), run_id="t", output_path=output_path, run_seed=42, resume=False, execution_mode="sync")

    out_df = pd.read_csv(output_path)
    assert not out_df["question_id"].duplicated().any(), (
        f"--no-resume against an existing output duplicated rows: "
        f"{out_df['question_id'].value_counts().to_dict()}"
    )
