from choicebench.cli import evaluate_run
from choicebench.io.readers import read_manifest_results
from choicebench.manifest import validate_manifest


def test_v2_fixture_validates_without_rewrite(synthetic_v2_run):
    run_dir, manifest, _ = synthetic_v2_run
    before = {
        p.relative_to(run_dir): p.read_bytes()
        for p in run_dir.rglob("*")
        if p.is_file()
    }
    validate_manifest(manifest)
    frame, loaded = read_manifest_results(run_dir)
    after = {
        p.relative_to(run_dir): p.read_bytes()
        for p in run_dir.rglob("*")
        if p.is_file()
    }
    assert loaded["schema_version"] == "choicebench.manifest.v2"
    assert frame["question_id"].astype(str).tolist() == ["q1", "q2"]
    assert after == before


def test_v2_evaluation_identity_and_shape_are_frozen(
    synthetic_v2_run, monkeypatch
):
    run_dir, manifest, expected_evaluation_id = synthetic_v2_run
    monkeypatch.setattr(evaluate_run, "RUNS_DIR", run_dir.parent)
    frame, _ = read_manifest_results(run_dir)
    report = evaluate_run.build_evaluation_report(
        run_dir.name, frame, manifest, reparse=False
    )
    assert report["schema_version"] == "choicebench.evaluation.v1"
    assert report["evaluation_id"] == expected_evaluation_id
    assert set(report["conditions"]) == {"cond_legacyfixture"}


def test_v2_missing_result_origin_implies_native_only_in_compat_view(
    synthetic_v2_run,
):
    _, manifest, _ = synthetic_v2_run
    assert "result_origin" not in manifest["payload"]["conditions"][0]
