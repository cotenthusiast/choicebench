import json
from pathlib import Path

import pandas as pd
import pytest
import asyncio
import importlib.util

from choicebench.datasets import (
    DatasetArtifactError, DatasetSpec, artifact_csv_path, dataset_content_digest,
    load_prepared_dataset, write_prepared_dataset,
)
from choicebench.identity import canonical_json, integrity_digest, stable_digest
from choicebench.infra.checkpoint import CheckpointManager
from choicebench.io.readers import ResultSetError, read_manifest_results
from choicebench.manifest import (
    ManifestCompatibilityError, RUN_STATE_SCHEMA_VERSION, ensure_manifest, make_manifest,
)


def _df(label="A"):
    return pd.DataFrame([{
        "question_id": "q1", "subject": "s", "question_text": "Q?",
        "choice_a": "x", "choice_b": "y", "correct_option": label,
        "correct_answer_text": "x" if label == "A" else "y",
    }])


def test_split_artifacts_have_distinct_paths_and_contents(tmp_path):
    test = DatasetSpec("mmlu", "test", "org/data", "all", output_name="mmlu")
    validation = DatasetSpec("mmlu", "validation", "org/data", "all", output_name="mmlu")
    a = write_prepared_dataset(_df("A"), tmp_path, test)
    b = write_prepared_dataset(_df("B"), tmp_path, validation)
    assert a.path != b.path
    assert a.content_digest != b.content_digest
    assert load_prepared_dataset(tmp_path, test).dataframe.iloc[0]["correct_option"] == "A"
    assert load_prepared_dataset(tmp_path, validation).dataframe.iloc[0]["correct_option"] == "B"


def test_dataset_digest_is_semantic_and_detects_mutation(tmp_path):
    spec = DatasetSpec("mmlu", "test", "org/data", output_name="mmlu")
    artifact = write_prepared_dataset(_df(), tmp_path, spec)
    assert dataset_content_digest(_df()[list(reversed(_df().columns))]) == artifact.content_digest
    mutated = pd.read_csv(artifact.path)
    mutated.loc[0, "correct_option"] = "B"
    mutated.to_csv(artifact.path, index=False)
    with pytest.raises(DatasetArtifactError, match="no longer matches"):
        load_prepared_dataset(tmp_path, spec)


def test_fresh_artifact_roundtrips_empty_strings_and_leading_zero_ids(tmp_path):
    spec = DatasetSpec("edge", "test", output_name="edge")
    frame = _df()
    frame.loc[0, "question_id"] = "001"
    frame.loc[0, "subject"] = ""
    written = write_prepared_dataset(frame, tmp_path, spec)
    loaded = load_prepared_dataset(tmp_path, spec)
    assert loaded.content_digest == written.content_digest
    assert str(loaded.dataframe.iloc[0]["question_id"]) == "001"


def test_dataset_source_metadata_is_integrity_bound(tmp_path):
    spec = DatasetSpec("mmlu", "test", output_name="mmlu")
    artifact = write_prepared_dataset(_df(), tmp_path, spec, source_metadata={"hf_fingerprint": "abc"})
    metadata = json.loads(artifact.metadata_path.read_text())
    metadata["source"]["hf_fingerprint"] = "changed"
    artifact.metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(DatasetArtifactError, match="integrity"):
        load_prepared_dataset(tmp_path, spec)


def test_legacy_generic_dataset_is_not_silently_claimed(tmp_path):
    (tmp_path / "mmlu_normalized.csv").write_text("question_id\nq1\n")
    with pytest.raises(FileNotFoundError, match="legacy unverified"):
        load_prepared_dataset(tmp_path, DatasetSpec("mmlu", "test", output_name="mmlu"))


def test_canonicalization_refuses_credentials_and_normalizes_order():
    # CB-4: credential-named keys are refused outright (fail closed) rather
    # than redacted, so two configs can never collapse into one identity
    # because a credential value was blanked before hashing.
    with pytest.raises(ValueError, match="Credential-named key"):
        canonical_json({"b": 2, "api_key": "top-secret", "a": 1})

    # Embedded URL credentials are sanitized; key order and URL spelling are
    # normalized, so these two spellings share one identity with no secrets.
    left = {"b": 2, "base_url": "https://user:pass@HOST/v1?token=x", "a": 1}
    right = {"a": 1, "base_url": "https://host/v1", "b": 2}
    rendered = canonical_json(left)
    assert "pass" not in rendered and "token=x" not in rendered
    assert stable_digest(left) == stable_digest(right)


@pytest.mark.parametrize("changed", [
    {"seed": 2}, {"prompt": "p2"}, {"method": {"name": "m", "alpha": 0.2}},
    {"model": {"backend": "api", "provider": "other"}}, {"dataset": "digest2"},
])
def test_manifest_refuses_every_scientific_identity_change(tmp_path, changed):
    base = {"seed": 1, "prompt": "p1", "method": {"name": "m", "alpha": 0.1},
            "model": {"backend": "api", "provider": "p"}, "dataset": "digest1"}
    ensure_manifest(tmp_path, make_manifest(base))
    candidate = dict(base)
    candidate.update(changed)
    with pytest.raises(ManifestCompatibilityError, match="belongs to experiment"):
        ensure_manifest(tmp_path, make_manifest(candidate))


def test_identical_manifest_is_resumable(tmp_path):
    manifest = make_manifest({"seed": 1})
    first, reused = ensure_manifest(tmp_path, manifest)
    second, reused_again = ensure_manifest(tmp_path, make_manifest({"seed": 1}))
    assert reused is False and reused_again is True
    assert first["experiment_id"] == second["experiment_id"]


def test_tampered_existing_manifest_is_rejected(tmp_path):
    manifest = make_manifest({"seed": 1})
    ensure_manifest(tmp_path, manifest)
    stored = json.loads((tmp_path / "manifest.json").read_text())
    stored["payload"]["seed"] = 999
    (tmp_path / "manifest.json").write_text(json.dumps(stored))
    with pytest.raises(ManifestCompatibilityError, match="integrity"):
        ensure_manifest(tmp_path, manifest)


def test_checkpoint_is_bound_to_condition_and_manifest(tmp_path):
    good = CheckpointManager(tmp_path, condition_id="cond_a", experiment_id="exp", selection_id="sel")
    good.save(["q1"], [{"question_id": "q1"}], "now")
    wrong = CheckpointManager(tmp_path, condition_id="cond_a", experiment_id="other", selection_id="sel")
    with pytest.raises(RuntimeError, match="belongs to"):
        wrong.load()


def test_identity_bound_interrupted_run_resumes_exact_remaining_rows(tmp_path):
    root = Path(__file__).resolve().parents[1]
    import choicebench.cli.run_experiment as mod
    mod = importlib.reload(mod)
    manager = CheckpointManager(
        tmp_path / "checkpoints",
        condition_id="cond_resume", experiment_id="exp_resume", selection_id="sel_resume",
    )
    metadata = {
        "condition_id": "cond_resume", "experiment_id": "exp_resume",
        "dataset_selection_id": "sel_resume", "dataset_artifact_id": "ds_resume",
        "model_id": "model_resume", "method_id": "method_resume",
        "prompt_id": "prompt_resume", "benchmark_split": "test",
    }
    manager.save(["q1"], [{"question_id": "q1", **metadata}], "now")

    class Runner:
        run_id = "r"
        backend = object()

        def run_many(self, rows):
            assert rows["question_id"].tolist() == ["q2"]
            return [{"question_id": "q2"}]

    questions = pd.DataFrame([{"question_id": "q1"}, {"question_id": "q2"}])
    asyncio.run(mod.run_method(
        "direct_mcq", Runner(), questions, manager, tmp_path, 1, resume=True,
        model_name="m", benchmark="toy", condition_metadata=metadata,
    ))
    result = pd.read_csv(tmp_path / "results" / "cond_resume.csv")
    assert result["question_id"].tolist() == ["q1", "q2"]


def _result_manifest(run_dir: Path, status="completed"):
    condition = {"condition_id": "cond_1", "result_path": "results/cond_1.csv",
                 "checkpoint_path": "checkpoints/cond_1.json",
                 "result_metadata_path": "results/cond_1.artifact.json",
                 "selection_id": "sel_1", "model_id": "model_1", "method_id": "method_1"}
    manifest = make_manifest({"conditions": [condition], "datasets": [
        {
            "selection_id": "sel_1", "selected_question_ids": ["q1"], "row_count": 1,
            "selected_content_digest": dataset_content_digest(pd.DataFrame([{"question_id": "q1"}])),
            "run_snapshot_path": "artifacts/datasets/sel_1.csv",
        }
    ]})
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "run_state.json").write_text(json.dumps({
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"], "conditions": {"cond_1": {"status": status}},
    }))
    snapshot = run_dir / "artifacts" / "datasets" / "sel_1.csv"
    snapshot.parent.mkdir(parents=True)
    pd.DataFrame([{"question_id": "q1"}]).to_csv(snapshot, index=False)


def test_manifest_reader_reports_missing_and_rejects_stale_results(tmp_path):
    _result_manifest(tmp_path)
    with pytest.raises(ResultSetError, match="missing"):
        read_manifest_results(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    (results / "stale.csv").write_text("x\n1\n")
    with pytest.raises(ResultSetError, match="Unexpected"):
        read_manifest_results(tmp_path)


def test_manifest_reader_revalidates_dataset_snapshot(tmp_path):
    _result_manifest(tmp_path)
    snapshot = tmp_path / "artifacts" / "datasets" / "sel_1.csv"
    snapshot.write_text("question_id\nchanged\n")
    with pytest.raises(ResultSetError, match="snapshot"):
        read_manifest_results(tmp_path)


def test_gated_condition_requires_identity_bound_gate_evidence(tmp_path):
    condition = {
        "condition_id": "cond_gate", "result_path": "results/cond_gate.csv",
        "checkpoint_path": "checkpoints/cond_gate.json",
        "result_metadata_path": "results/cond_gate.artifact.json",
        "gate_path": "artifacts/cond_gate/modal_k_gate.json",
        "selection_id": "sel", "model_id": "model", "method_id": "method",
    }
    manifest = make_manifest({"conditions": [condition]})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "run_state.json").write_text(json.dumps({
        "schema_version": RUN_STATE_SCHEMA_VERSION, "experiment_id": manifest["experiment_id"],
        "conditions": {"cond_gate": {"status": "gated"}},
    }))
    with pytest.raises(ResultSetError, match="gate artifact"):
        read_manifest_results(tmp_path)

    gate = {"condition_id": "cond_gate", "experiment_id": manifest["experiment_id"]}
    gate["artifact_digest"] = integrity_digest(gate)
    path = tmp_path / condition["gate_path"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(gate))
    frame, _ = read_manifest_results(tmp_path)
    assert frame.empty


# ---------------------------------------------------------------------------
# CB-1 (reader side): failed conditions are legal accounting states; anything
# unfinished is refused. CB-6: null row-ownership identities fail closed.
# ---------------------------------------------------------------------------

def test_manifest_reader_accounts_failed_condition_without_artifact(tmp_path):
    _result_manifest(tmp_path, status="failed")
    frame, manifest = read_manifest_results(tmp_path)
    assert frame.empty
    assert manifest["payload"]["conditions"][0]["condition_id"] == "cond_1"


def test_manifest_reader_rejects_failed_condition_with_result_artifact(tmp_path):
    _result_manifest(tmp_path, status="failed")
    results = tmp_path / "results"
    results.mkdir()
    (results / "cond_1.csv").write_text("question_id\nq1\n")
    with pytest.raises(ResultSetError, match="failed condition has a result artifact"):
        read_manifest_results(tmp_path)


def test_manifest_reader_refuses_unfinished_run(tmp_path):
    _result_manifest(tmp_path, status="pending")
    with pytest.raises(ResultSetError, match="has not finished"):
        read_manifest_results(tmp_path)


def _completed_run_with_rows(run_dir: Path, rows: list[dict]):
    """A fully valid completed run: manifest, state, snapshot, result artifact."""
    from choicebench.io.writers import validate_result_artifact, write_run_results

    question_ids = [row["question_id"] for row in rows]
    selection_df = pd.DataFrame([{"question_id": qid} for qid in question_ids])
    condition = {
        "condition_id": "cond_1", "result_path": "results/cond_1.csv",
        "checkpoint_path": "checkpoints/cond_1.json",
        "result_metadata_path": "results/cond_1.artifact.json",
        "selection_id": "sel_1", "model_id": "model_1", "method_id": "method_1",
        "prompt_id": "prompt_1", "artifact_id": "ds_1",
        "split": "test", "benchmark_name": "toy",
    }
    manifest = make_manifest({
        "conditions": [condition],
        "datasets": [{
            "selection_id": "sel_1", "selected_question_ids": question_ids,
            "row_count": len(question_ids),
            "selected_content_digest": dataset_content_digest(selection_df),
            "run_snapshot_path": "artifacts/datasets/sel_1.csv",
        }],
        "models": [{"model_id": "model_1", "config": {"model_name_or_path": "dummy_model"}}],
        "methods": [{"method_id": "method_1", "config": {"name": "direct_mcq"}}],
    })
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    snapshot = run_dir / "artifacts" / "datasets" / "sel_1.csv"
    snapshot.parent.mkdir(parents=True)
    selection_df.to_csv(snapshot, index=False)
    rows = [{**row, "experiment_id": manifest["experiment_id"]} for row in rows]
    result_path = write_run_results(rows, run_dir, "cond_1", "toy")
    metadata = validate_result_artifact(result_path)
    (run_dir / "run_state.json").write_text(json.dumps({
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "conditions": {"cond_1": {"status": "completed",
                                  "result_path": "results/cond_1.csv",
                                  "result_sha256": metadata["file_sha256"]}},
    }))
    return manifest, result_path


def _identity_row(question_id: str, experiment_id: str) -> dict:
    return {
        "question_id": question_id, "is_correct": True,
        "experiment_id": experiment_id, "condition_id": "cond_1",
        "dataset_artifact_id": "ds_1", "dataset_selection_id": "sel_1",
        "model_id": "model_1", "method_id": "method_1", "prompt_id": "prompt_1",
        "benchmark_split": "test", "model_name": "dummy_model",
        "method_name": "direct_mcq",
    }


def test_null_row_ownership_is_rejected_at_read_even_with_consistent_digests(tmp_path):
    """CB-6: a result row whose condition_id is null must fail validation even
    when the artifact sidecar digests are recomputed to be self-consistent —
    corrupt rows are an error, never a silent shrink of the metric sample."""
    manifest, result_path = _completed_run_with_rows(
        tmp_path, [_identity_row("q1", "placeholder"), _identity_row("q2", "placeholder")],
    )
    baseline, _ = read_manifest_results(tmp_path)
    assert len(baseline) == 2  # fixture is valid before corruption

    frame = pd.read_csv(result_path)
    frame.loc[0, "condition_id"] = None
    frame.to_csv(result_path, index=False)
    _rewrite_artifact(result_path, tmp_path)
    with pytest.raises(ResultSetError, match="invalid condition_id"):
        read_manifest_results(tmp_path)


def _rewrite_artifact(result_path: Path, run_dir: Path) -> None:
    """Recompute a self-consistent artifact sidecar + run-state digest after
    editing the CSV (simulates corruption that keeps every digest valid)."""
    from choicebench.identity import file_digest
    from choicebench.io.writers import result_artifact_path

    sidecar = result_artifact_path(result_path)
    metadata = json.loads(sidecar.read_text())
    metadata.pop("metadata_digest")
    metadata["file_sha256"] = file_digest(result_path)
    metadata["row_count"] = len(pd.read_csv(result_path))
    metadata["metadata_digest"] = integrity_digest(metadata)
    sidecar.write_text(json.dumps(metadata))
    state_path = run_dir / "run_state.json"
    state = json.loads(state_path.read_text())
    state["conditions"]["cond_1"]["result_sha256"] = metadata["file_sha256"]
    state_path.write_text(json.dumps(state))


def test_null_row_ownership_is_rejected_at_write(tmp_path):
    from choicebench.io.writers import write_run_results

    rows = [_identity_row("q1", "exp_x"), {**_identity_row("q2", "exp_x"), "condition_id": None}]
    with pytest.raises(RuntimeError, match="missing condition_id identity"):
        write_run_results(rows, tmp_path, "cond_1", "toy")
