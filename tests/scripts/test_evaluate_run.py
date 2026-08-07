# tests/scripts/test_evaluate_run.py
#
# Regression for FSF-2 / PF-1: --reparse must only touch direct_mcq rows. For
# every other method the stored raw_text is not the call the reported answer
# came from, so reparsing it silently corrupts the result. This test builds a
# mixed-method run and asserts only the direct_mcq row is recomputed.

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from choicebench.identity import integrity_digest
from choicebench.manifest import RUN_STATE_SCHEMA_VERSION, make_manifest
from choicebench.provenance import implementation_identity

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_evaluate_run():
    import choicebench.cli.evaluate_run as module
    return importlib.reload(module)


def test_json_safe_converts_nonfinite_metric_values_to_null():
    mod = _load_evaluate_run()
    assert mod._json_safe({"nan": float("nan"), "inf": float("inf"), "ok": 1.0}) == {
        "nan": None, "inf": None, "ok": 1.0,
    }


_ROW_CHOICES_JSON = json.dumps(
    [{"text": t, "source_index": i} for i, t in enumerate(["alpha", "bravo", "charlie", "delta"])]
)


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
        "choices_json": _ROW_CHOICES_JSON,
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


def _gated_evaluation_manifest(tmp_path, mod):
    metric = mod._load_metric("accuracy")
    condition = {
        "condition_id": "cond_gate", "result_path": "results/cond_gate.csv",
        "checkpoint_path": "checkpoints/cond_gate.json",
        "result_metadata_path": "results/cond_gate.artifact.json",
        "selection_id": "sel", "model_id": "model", "method_id": "method",
        "model_display_name": "dummy", "benchmark_name": "toy", "split": "test",
    }
    manifest = make_manifest({
        "conditions": [condition],
        "methods": [{"method_id": "method", "config": {"name": "pride"}}],
        "evaluation": [{"name": "accuracy", "implementation": implementation_identity(type(metric))}],
    })
    run_dir = tmp_path / "gated"
    run_dir.mkdir()
    (run_dir / "run_state.json").write_text(json.dumps({
        "schema_version": RUN_STATE_SCHEMA_VERSION, "experiment_id": manifest["experiment_id"],
        "conditions": {"cond_gate": {"status": "gated"}},
    }))
    return run_dir, manifest


def test_evaluation_rejects_metric_code_not_bound_by_manifest():
    mod = _load_evaluate_run()
    manifest = make_manifest({"evaluation": [{"name": "accuracy", "implementation": {"wrong": True}}]})
    with pytest.raises(RuntimeError, match="no longer matches"):
        mod._verified_metric_records(manifest)


def test_all_gated_run_has_accounting_report_and_reparse_is_collision_safe(tmp_path, monkeypatch):
    mod = _load_evaluate_run()
    run_dir, manifest = _gated_evaluation_manifest(tmp_path, mod)
    monkeypatch.setattr(mod, "RUNS_DIR", tmp_path)
    ordinary = mod.build_evaluation_report("gated", pd.DataFrame(), manifest, False)
    reparsed = mod.build_evaluation_report("gated", pd.DataFrame(), manifest, True)
    assert ordinary["conditions"]["cond_gate"]["status"] == "gated"
    assert ordinary["conditions"]["cond_gate"]["metrics"] == {}
    assert ordinary["evaluation_id"] != reparsed["evaluation_id"]
    assert ordinary["source"]["experiment_id"] == manifest["experiment_id"]
    assert ordinary["report_digest"] == integrity_digest({
        key: value for key, value in ordinary.items() if key != "report_digest"
    })


# ---------------------------------------------------------------------------
# CB-1: failed conditions are accounted, never block evaluation of the rest.
# CB-3: evaluation identity binds the exact consumed result contents.
# ---------------------------------------------------------------------------

def _mixed_run(tmp_path, mod, statuses: dict[str, dict]):
    """Build a manifest + run_state for conditions with the given state entries."""
    metric = mod._load_metric("accuracy")
    conditions = [
        {
            "condition_id": condition_id, "result_path": f"results/{condition_id}.csv",
            "checkpoint_path": f"checkpoints/{condition_id}.json",
            "result_metadata_path": f"results/{condition_id}.artifact.json",
            "selection_id": "sel", "model_id": "model", "method_id": "method",
            "model_display_name": "dummy", "benchmark_name": "toy", "split": "test",
        }
        for condition_id in sorted(statuses)
    ]
    manifest = make_manifest({
        "conditions": conditions,
        "methods": [{"method_id": "method", "config": {"name": "direct_mcq"}}],
        "evaluation": [{"name": "accuracy", "implementation": implementation_identity(type(metric))}],
    })
    run_dir = tmp_path / "mixed"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "run_state.json").write_text(json.dumps({
        "schema_version": RUN_STATE_SCHEMA_VERSION, "experiment_id": manifest["experiment_id"],
        "conditions": statuses,
    }))
    return run_dir, manifest


def _result_df(condition_id: str, correct: int, total: int) -> pd.DataFrame:
    rows = []
    for index in range(total):
        row = _row("direct_mcq", "Answer successfully parsed", "text",
                   "C" if index < correct else "A")
        row.update({
            "condition_id": condition_id, "model_name": "dummy",
            "benchmark_name": "toy", "benchmark_split": "test",
            "question_id": f"q{index}",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def test_partial_run_with_failed_condition_is_accounted(tmp_path, monkeypatch):
    mod = _load_evaluate_run()
    run_dir, manifest = _mixed_run(tmp_path, mod, {
        "cond_bad": {"status": "failed", "error": "backend exploded"},
        "cond_ok": {"status": "completed", "result_sha256": "aa" * 32},
    })
    monkeypatch.setattr(mod, "RUNS_DIR", tmp_path)
    report = mod.build_evaluation_report("mixed", _result_df("cond_ok", 1, 2), manifest, False)

    assert report["run_status"] == "partial"
    assert report["condition_counts"] == {"completed": 1, "gated": 0, "failed": 1, "infra_failure": 0}
    assert report["conditions"]["cond_bad"]["status"] == "failed"
    assert report["conditions"]["cond_bad"]["metrics"] == {}
    assert report["conditions"]["cond_bad"]["error"] == "backend exploded"
    assert report["conditions"]["cond_ok"]["metrics"]["accuracy"] == 0.5
    assert report["report_digest"] == integrity_digest({
        key: value for key, value in report.items() if key != "report_digest"
    })


def test_all_failed_run_produces_accounting_report(tmp_path, monkeypatch):
    mod = _load_evaluate_run()
    run_dir, manifest = _mixed_run(tmp_path, mod, {
        "cond_a": {"status": "failed"},
        "cond_b": {"status": "failed"},
    })
    monkeypatch.setattr(mod, "RUNS_DIR", tmp_path)
    report = mod.build_evaluation_report("mixed", pd.DataFrame(), manifest, False)

    assert report["run_status"] == "partial"
    assert report["condition_counts"] == {"completed": 0, "gated": 0, "failed": 2, "infra_failure": 0}
    assert all(item["status"] == "failed" and item["metrics"] == {}
               for item in report["conditions"].values())


def _transport_failure_df(condition_id: str, total: int) -> pd.DataFrame:
    rows = []
    for index in range(total):
        row = _row("direct_mcq", "transport_error", None, None, transport_status="failure")
        row.update({
            "condition_id": condition_id, "model_name": "dummy",
            "benchmark_name": "toy", "benchmark_split": "test",
            "question_id": f"q{index}",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def test_all_transport_failure_condition_is_flagged_infra_failure(tmp_path, monkeypatch):
    """A condition whose result CSV exists (run-state status: completed) but
    whose rows are entirely backend transport failures must not be reported
    as an indistinguishable 'accuracy: 0.0, status: completed' — that reads
    identically to a genuinely bad model producing all-wrong answers."""
    mod = _load_evaluate_run()
    run_dir, manifest = _mixed_run(tmp_path, mod, {
        "cond_ok": {"status": "completed", "result_sha256": "aa" * 32},
    })
    monkeypatch.setattr(mod, "RUNS_DIR", tmp_path)
    report = mod.build_evaluation_report("mixed", _transport_failure_df("cond_ok", 4), manifest, False)

    condition = report["conditions"]["cond_ok"]
    assert condition["status"] == "infra_failure"
    assert condition["transport_failure_count"] == 4
    assert condition["transport_failure_fraction"] == 1.0
    assert condition["metrics"]["accuracy"] == 0.0
    assert report["condition_counts"] == {"completed": 0, "gated": 0, "failed": 0, "infra_failure": 1}
    assert report["run_status"] == "partial"


def test_unfinished_condition_refuses_evaluation(tmp_path, monkeypatch):
    mod = _load_evaluate_run()
    run_dir, manifest = _mixed_run(tmp_path, mod, {"cond_a": {"status": "pending"}})
    monkeypatch.setattr(mod, "RUNS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="has not finished"):
        mod.build_evaluation_report("mixed", pd.DataFrame(), manifest, False)


def test_different_result_contents_get_distinct_evaluation_ids_and_coexist(tmp_path, monkeypatch):
    """The audit's reset/rerun scenario: same experiment, materially different
    results (accuracy 0.5 vs 1.0) must yield different evaluation IDs so both
    reports coexist instead of one silently replacing the other."""
    mod = _load_evaluate_run()
    monkeypatch.setattr(mod, "RUNS_DIR", tmp_path)

    run_dir, manifest = _mixed_run(tmp_path, mod, {
        "cond_ok": {"status": "completed", "result_sha256": "aa" * 32},
    })
    first = mod.build_evaluation_report("mixed", _result_df("cond_ok", 1, 2), manifest, False)

    run_dir, manifest = _mixed_run(tmp_path, mod, {
        "cond_ok": {"status": "completed", "result_sha256": "bb" * 32},
    })
    second = mod.build_evaluation_report("mixed", _result_df("cond_ok", 2, 2), manifest, False)

    assert first["conditions"]["cond_ok"]["metrics"]["accuracy"] == 0.5
    assert second["conditions"]["cond_ok"]["metrics"]["accuracy"] == 1.0
    assert first["evaluation_id"] != second["evaluation_id"]

    reports = tmp_path / "reports"
    reports.mkdir()
    for report in (first, second):
        mod.write_evaluation_report(reports / f"mixed_{report['evaluation_id']}_metrics.json", report)
    written = sorted(path.name for path in reports.glob("*.json"))
    assert len(written) == 2

    # Identical result contents keep the same evaluation identity.
    run_dir, manifest = _mixed_run(tmp_path, mod, {
        "cond_ok": {"status": "completed", "result_sha256": "aa" * 32},
    })
    repeat = mod.build_evaluation_report("mixed", _result_df("cond_ok", 1, 2), manifest, False)
    assert repeat["evaluation_id"] == first["evaluation_id"]


def test_write_evaluation_report_refuses_divergent_overwrite(tmp_path):
    mod = _load_evaluate_run()
    from choicebench.io.readers import ResultSetError

    path = tmp_path / "run_eval_x_metrics.json"
    report = {"evaluation_id": "eval_x", "value": 1}
    report["report_digest"] = integrity_digest(report)
    mod.write_evaluation_report(path, report)
    # Idempotent re-write of the identical report is allowed.
    mod.write_evaluation_report(path, report)

    divergent = {"evaluation_id": "eval_x", "value": 2}
    divergent["report_digest"] = integrity_digest(divergent)
    with pytest.raises(ResultSetError, match="Refusing to overwrite"):
        mod.write_evaluation_report(path, divergent)
    assert json.loads(path.read_text())["value"] == 1

    # A tampered existing file that kept its stale report_digest is refused
    # too: the guard compares full content, not the recorded digest line.
    tampered = dict(report)
    tampered["value"] = 99
    path.write_text(json.dumps(tampered))
    with pytest.raises(ResultSetError, match="Refusing to overwrite"):
        mod.write_evaluation_report(path, report)
